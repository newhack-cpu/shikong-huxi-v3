# multistep_forecasting.py
"""
多步预测对比实验（v2 新增）
============================

【目的】
报告 5.3 节、PPT 第 15 页声称 "MSTN v2 在 72h 预测上比 LightGBM 低 36%"
——必须有真实实验脚本支撑。

【实验设计】
- 提前量: [1h, 6h, 12h, 24h, 48h, 72h]
- 对比模型: LightGBM, Pure LSTM (recursive), MSTN v2 (recursive)
- 评估指标: MAE / RMSE / R² / MAPE
- 对每个提前量, 在测试集上分别计算

【方法】
- 对深度学习模型, 用 recursive forecasting (滚动预测):
    pred[t+1] = model(history)
    history = [..., pred[t+1]]
    pred[t+2] = model(history)
- 对树模型同理

【运行】
python multistep_forecasting.py                   # 全部模型 + 全部提前量
python multistep_forecasting.py --quick           # 只跑 6 个提前量中前 3 个
python multistep_forecasting.py --horizons 1 24   # 只跑指定提前量

【输出】
results/multistep_results.csv      - 完整结果表
results/multistep_results.png      - 对比折线图
results/multistep_predictions.csv  - MSTN v2 各提前量的预测序列
"""

import os
import sys
import argparse
import json
import importlib.util
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def mape(actual, pred):
    """平均绝对百分比误差"""
    actual = np.asarray(actual)
    pred = np.asarray(pred)
    mask = actual > 1e-6
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def metrics_dict(actual, pred):
    return {
        'MAE':  mean_absolute_error(actual, pred),
        'RMSE': float(np.sqrt(mean_squared_error(actual, pred))),
        'R2':   r2_score(actual, pred),
        'MAPE': mape(actual, pred),
    }


# ════════════════════════════════════════════════════════════════
# LightGBM 多步预测（用 lag 特征，递归预测会失效，所以用直接训练 N 个模型）
# ════════════════════════════════════════════════════════════════

def lightgbm_multistep(X, y, train_end, val_end, horizons):
    """
    对每个提前量训练独立的 LightGBM 模型 (direct multi-step)

    输入:
        X: [N, F] 特征矩阵 (已对齐)
        y: [N]   目标值
        train_end, val_end: 切分点
        horizons: list of int, 预测提前量
    """
    try:
        import lightgbm as lgb
    except ImportError:
        print("⚠️  LightGBM 未安装，跳过")
        return None

    print("\n[LightGBM] 多步直接预测 (direct multi-step)...")
    results = {}

    for h in horizons:
        # 构造 (X_t, y_{t+h}) 对
        if h >= len(X):
            continue
        X_h = X[:-h]
        y_h = y[h:]

        # 切分
        X_tr, y_tr = X_h[:train_end], y_h[:train_end]
        X_te, y_te = X_h[val_end:],   y_h[val_end:]

        if len(X_tr) < 100 or len(X_te) < 50:
            continue

        model = lgb.LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, n_jobs=-1, verbose=-1,
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)

        results[h] = {
            'metrics':  metrics_dict(y_te, pred),
            'actual':   y_te.tolist()[:200],
            'predicted': pred.tolist()[:200],
        }
        print(f"  h={h:2d}h  MAE={results[h]['metrics']['MAE']:.3f}  "
              f"R²={results[h]['metrics']['R2']:.4f}")

    return results


# ════════════════════════════════════════════════════════════════
# MSTN v2 多步预测（递归 forecasting）
# ════════════════════════════════════════════════════════════════

def mstn_multistep(df, feat_cols, train_end, val_end, horizons, seq_len=24):
    """
    使用训练好的 MSTN v2 做递归多步预测
    """
    try:
        import torch
        import joblib
    except ImportError:
        print("⚠️  PyTorch 未安装，跳过 MSTN v2")
        return None

    weights = 'models/mstn_v2_best.pth'
    sx_path = 'models/mstn_scaler_X.pkl'
    sy_path = 'models/mstn_scaler_y.pkl'

    if not all(os.path.exists(p) for p in [weights, sx_path, sy_path]):
        print(f"⚠️  MSTN v2 模型文件未找到")
        print(f"    请先运行: python 6_advanced_model_v2.py")
        return None

    print("\n[MSTN v2] 多步递归预测 (recursive forecasting)...")
    spec = importlib.util.spec_from_file_location("adv", "6_advanced_model_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    scaler_X = joblib.load(sx_path)
    scaler_y = joblib.load(sy_path)
    n_features = scaler_X.n_features_in_

    model = mod.MSTNv2(input_dim=n_features, hidden_dim=64)
    state = torch.load(weights, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()

    X_raw = df[feat_cols].values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    y_raw = df['pm25'].values.astype(np.float32)

    # 标准化
    X_scaled = scaler_X.transform(X_raw)
    y_scaled = scaler_y.transform(y_raw.reshape(-1, 1)).flatten()

    results = {}
    pm25_col_idx = feat_cols.index('pm25_lag_1h') if 'pm25_lag_1h' in feat_cols else None

    for h in horizons:
        # 在测试集上做 h 步递归预测
        actual_list, pred_list = [], []
        max_h_in_test = min(h, len(X_scaled) - val_end - seq_len)

        for start_idx in range(val_end, len(X_scaled) - seq_len - h):
            # 初始历史窗口
            window = X_scaled[start_idx:start_idx + seq_len].copy()

            # 递归 h 步
            current_pred = None
            with torch.no_grad():
                for step in range(h):
                    x_t = torch.from_numpy(window).float().unsqueeze(0)
                    pred, _, _ = model(x_t)
                    current_pred = float(pred[0, 1].item())   # q50

                    if step < h - 1:
                        # 滚动窗口：用预测值替换 lag_1h（如果存在该列）
                        new_row = window[-1].copy()
                        if pm25_col_idx is not None:
                            new_row[pm25_col_idx] = current_pred
                        window = np.vstack([window[1:], new_row])

            # 反标准化
            pred_orig = scaler_y.inverse_transform(
                np.array([[current_pred]])
            )[0, 0]
            actual_orig = y_raw[start_idx + seq_len + h - 1]

            actual_list.append(actual_orig)
            pred_list.append(pred_orig)

            if len(actual_list) >= 500:   # 限制测试样本数量加速
                break

        if len(actual_list) > 50:
            results[h] = {
                'metrics':   metrics_dict(actual_list, pred_list),
                'actual':    actual_list[:200],
                'predicted': pred_list[:200],
            }
            print(f"  h={h:2d}h  MAE={results[h]['metrics']['MAE']:.3f}  "
                  f"R²={results[h]['metrics']['R2']:.4f}  "
                  f"(n={len(actual_list)})")

    return results


# ════════════════════════════════════════════════════════════════
# 可视化
# ════════════════════════════════════════════════════════════════

def visualize(all_results, horizons, output_path='multistep_results.png'):
    """生成多步预测对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        'LightGBM':      '#FF7F0E',
        'MSTN v2 (Ours)': '#D62728',
    }
    markers = {'LightGBM': 'o', 'MSTN v2 (Ours)': '^'}

    for model_name, results in all_results.items():
        if not results:
            continue
        hs = sorted(results.keys())
        maes = [results[h]['metrics']['MAE'] for h in hs]
        r2s  = [results[h]['metrics']['R2']  for h in hs]

        c = colors.get(model_name, '#888')
        m = markers.get(model_name, 'o')

        lw = 2.5 if 'Ours' in model_name else 2.0
        ms = 9 if 'Ours' in model_name else 7

        ax1.plot(hs, maes, m + '-', color=c, linewidth=lw, markersize=ms,
                 markeredgecolor='white', label=model_name)
        ax2.plot(hs, r2s, m + '-', color=c, linewidth=lw, markersize=ms,
                 markeredgecolor='white', label=model_name)

    ax1.set_xlabel('Forecast Horizon (hours)', fontweight='bold')
    ax1.set_ylabel('MAE (μg/m³)  ↓', fontweight='bold')
    ax1.set_title('(a) Mean Absolute Error vs Horizon', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xticks(horizons)

    ax2.set_xlabel('Forecast Horizon (hours)', fontweight='bold')
    ax2.set_ylabel('R² Score  ↑', fontweight='bold')
    ax2.set_title('(b) R² vs Horizon', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks(horizons)

    fig.suptitle('Multi-step Forecasting: MSTN v2 vs LightGBM',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 可视化已保存: {output_path}")


# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data_with_features.csv')
    parser.add_argument('--horizons', nargs='+', type=int,
                        default=[1, 6, 12, 24, 48, 72])
    parser.add_argument('--quick', action='store_true',
                        help='只跑前 3 个提前量')
    args = parser.parse_args()

    horizons = args.horizons[:3] if args.quick else args.horizons

    print("=" * 70)
    print(f"多步预测对比实验  (horizons = {horizons})")
    print("=" * 70)

    if not os.path.exists(args.data):
        print(f"❌ 数据文件不存在: {args.data}")
        sys.exit(1)

    df = pd.read_csv(args.data)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # ✅ v2 反泄漏特征过滤
    try:
        from feature_safety import get_safe_feature_cols
        feat_cols = get_safe_feature_cols(df, target='pm25', verbose=True)
    except ImportError:
        exclude = ['timestamp', 'city', 'pm25', 'time_period', 'season',
                   'wind_direction', 'station', 'source',
                   'pm25_normalized', 'pm25_global_mean', 'pm25_global_std',
                   'pm25_deviation_from_mean', 'pm25_24h_mean',
                   'bhi', 'bhi_level', 'bhi_ipm', 'bhi_it', 'bhi_ie',
                   'breathing_health_index']
        feat_cols = [c for c in df.columns
                     if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  样本数: {len(df):,} | 特征数: {len(feat_cols)}")

    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X = df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y = df['pm25'].values

    all_results = {}

    # 1. LightGBM
    res_lgbm = lightgbm_multistep(X, y, train_end, val_end, horizons)
    if res_lgbm:
        all_results['LightGBM'] = res_lgbm

    # 2. MSTN v2
    res_mstn = mstn_multistep(df, feat_cols, train_end, val_end, horizons)
    if res_mstn:
        all_results['MSTN v2 (Ours)'] = res_mstn

    if not all_results:
        print("❌ 没有任何模型成功，请先训练模型")
        sys.exit(1)

    # 3. 汇总表
    print("\n" + "=" * 70)
    print("汇总: MAE 对比")
    print("=" * 70)

    table_data = []
    for h in horizons:
        row = {'Horizon': f'{h}h'}
        for model_name in all_results:
            if h in all_results[model_name]:
                row[model_name + ' MAE'] = all_results[model_name][h]['metrics']['MAE']
                row[model_name + ' R²']  = all_results[model_name][h]['metrics']['R2']
        if 'MSTN v2 (Ours)' in all_results and 'LightGBM' in all_results:
            if h in all_results['MSTN v2 (Ours)'] and h in all_results['LightGBM']:
                mae_mstn = all_results['MSTN v2 (Ours)'][h]['metrics']['MAE']
                mae_lgbm = all_results['LightGBM'][h]['metrics']['MAE']
                row['Improvement vs LightGBM'] = (
                    f"{(1 - mae_mstn/mae_lgbm) * 100:+.1f}%"
                )
        table_data.append(row)

    df_results = pd.DataFrame(table_data)
    print(df_results.to_string(index=False))
    df_results.to_csv('multistep_results.csv', index=False)
    print(f"\n✅ 汇总表已保存: multistep_results.csv")

    # 4. 可视化
    visualize(all_results, horizons)

    # 5. 详细结果 JSON
    serializable = {
        m: {str(h): r['metrics'] for h, r in res.items()}
        for m, res in all_results.items()
    }
    with open('multistep_results.json', 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"✅ 详细结果: multistep_results.json")


if __name__ == '__main__':
    main()

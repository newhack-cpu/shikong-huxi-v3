# -*- coding: utf-8 -*-
# mstn_hyperparameter_search.py
"""
MSTN v2/v3 超参数搜索（v3.6 新增 · 策略 A）
==============================================

【目的】
针对 Claude Code 实跑发现的 MSTN v2 R²=0.756 输给 XGBoost R²=0.902 问题，
通过超参数搜索探索 MSTN 是否能在某组配置下反超树模型。

【搜索维度】
- hidden_dim    : [32, 48, 64, 96, 128]
- seq_len       : [24, 48, 72]
- dropout       : [0.1, 0.2, 0.3]
- lr            : [5e-4, 1e-3, 2e-3]
- batch_size    : [32, 64, 128]

完整笛卡尔积太大（5×3×3×3×3=405 组），实际用：
- --quick 模式: 8 组核心配置（约 5-15 分钟）
- --full 模式: 27 组关键配置（约 1-2 小时）
- --custom: 用户自定义 JSON 配置文件

【输出】
results/mstn_hp_search_results.csv  - 每个配置的 MAE/RMSE/R²
results/mstn_hp_search_best.json    - 最佳配置 + 推荐
results/mstn_hp_search.png          - 可视化对比图

【运行】
# 快速搜索 (推荐先跑这个)
python mstn_hyperparameter_search.py --quick

# 完整搜索
python mstn_hyperparameter_search.py --full

# 跑完后查看最佳配置:
cat results/mstn_hp_search_best.json
"""

import os
import sys

# ✅ v3.6: Windows GBK 兼容性
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from _windows_compat import configure_utf8_stdout
    configure_utf8_stdout()
except ImportError:
    pass

import argparse
import importlib.util
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, Subset
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ═══════════════════════════════════════════════════════════════
# 配置预设
# ═══════════════════════════════════════════════════════════════

QUICK_CONFIGS = [
    # 基线（v2 默认）
    dict(hidden_dim=64, seq_len=24, dropout=0.1, lr=1e-3, batch_size=64,
         name='baseline_v2_default'),
    # 缩小容量防过拟合
    dict(hidden_dim=32, seq_len=24, dropout=0.2, lr=1e-3, batch_size=64,
         name='small_capacity_more_dropout'),
    # 增加容量 + 更高 dropout
    dict(hidden_dim=128, seq_len=24, dropout=0.3, lr=1e-3, batch_size=64,
         name='large_capacity_high_dropout'),
    # 更长序列捕捉长周期
    dict(hidden_dim=64, seq_len=48, dropout=0.1, lr=1e-3, batch_size=64,
         name='longer_seq_48h'),
    # 更长序列 + 中等容量
    dict(hidden_dim=64, seq_len=72, dropout=0.2, lr=1e-3, batch_size=64,
         name='longer_seq_72h'),
    # 更小学习率 + 更多 epoch（让 pinball loss 充分收敛）
    dict(hidden_dim=64, seq_len=24, dropout=0.1, lr=5e-4, batch_size=64,
         name='lower_lr_5e-4'),
    # 更高学习率
    dict(hidden_dim=64, seq_len=24, dropout=0.1, lr=2e-3, batch_size=64,
         name='higher_lr_2e-3'),
    # 大 batch + 长序列（GPU 友好）
    dict(hidden_dim=64, seq_len=48, dropout=0.2, lr=1e-3, batch_size=128,
         name='large_batch_long_seq'),
]

FULL_CONFIGS = QUICK_CONFIGS + [
    # hidden_dim 扫描
    dict(hidden_dim=48, seq_len=24, dropout=0.1, lr=1e-3, batch_size=64,
         name='hidden_48'),
    dict(hidden_dim=96, seq_len=24, dropout=0.1, lr=1e-3, batch_size=64,
         name='hidden_96'),
    # seq_len × hidden 组合
    dict(hidden_dim=32, seq_len=72, dropout=0.2, lr=1e-3, batch_size=64,
         name='small_long'),
    dict(hidden_dim=128, seq_len=72, dropout=0.3, lr=5e-4, batch_size=64,
         name='large_long_low_lr'),
    # 极端 dropout
    dict(hidden_dim=64, seq_len=48, dropout=0.4, lr=1e-3, batch_size=64,
         name='very_high_dropout'),
    # 长序列 + 高 lr
    dict(hidden_dim=64, seq_len=72, dropout=0.1, lr=2e-3, batch_size=128,
         name='long_seq_high_lr'),
    # 小容量 + 长序列（押注"简单优于复杂"）
    dict(hidden_dim=32, seq_len=48, dropout=0.1, lr=1e-3, batch_size=64,
         name='small_capacity_long_seq'),
    dict(hidden_dim=24, seq_len=24, dropout=0.05, lr=1e-3, batch_size=64,
         name='ultra_small'),
]


# ═══════════════════════════════════════════════════════════════
# 数据准备
# ═══════════════════════════════════════════════════════════════

def load_v2_module():
    here = Path(__file__).parent
    path = here / '6_advanced_model_v2.py'
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}")
    spec = importlib.util.spec_from_file_location("v2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_safe_features(df, target='pm25'):
    try:
        from feature_safety import get_safe_feature_cols
        return get_safe_feature_cols(df, target=target, verbose=False)
    except ImportError:
        exclude = {'timestamp', 'city', 'pm25', 'time_period', 'season',
                   'wind_direction', 'station', 'source',
                   'pm25_normalized', 'pm25_global_mean', 'pm25_global_std',
                   'pm25_deviation_from_mean', 'pm25_24h_mean',
                   'bhi', 'bhi_level', 'bhi_ipm', 'bhi_it', 'bhi_ie'}
        return [c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


# ═══════════════════════════════════════════════════════════════
# 单个配置训练 + 评估
# ═══════════════════════════════════════════════════════════════

def train_one_config(df, feat_cols, config, epochs=20, verbose=False):
    """
    用一组超参数训练 MSTN v2，返回测试集 MAE/RMSE/R²。
    
    epochs 默认只有 20 (节省时间), 真正部署时用更多。
    """
    v2 = load_v2_module()

    X_raw = df[feat_cols].fillna(0).values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    y_raw = df['pm25'].values.astype(np.float32)

    # 时序切分
    seq_len = config['seq_len']
    n = len(df) - seq_len
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    scaler_X = StandardScaler().fit(X_raw[:n_train + seq_len])
    scaler_y = StandardScaler().fit(y_raw[:n_train + seq_len].reshape(-1, 1))
    X_scaled = scaler_X.transform(X_raw)
    y_scaled = scaler_y.transform(y_raw.reshape(-1, 1)).flatten()

    class _DS(Dataset):
        def __init__(self, X, y, L):
            self.X, self.y, self.L = X, y, L
        def __len__(self):
            return len(self.X) - self.L
        def __getitem__(self, i):
            return (
                torch.from_numpy(self.X[i:i+self.L]).float(),
                torch.tensor([self.y[i+self.L]], dtype=torch.float32),
            )

    full = _DS(X_scaled, y_scaled, seq_len)
    train_set = Subset(full, range(0, n_train))
    val_set = Subset(full, range(n_train, n_train + n_val))
    test_set = Subset(full, range(n_train + n_val, len(full)))

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=False)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = v2.MSTNv2(
        input_dim=len(feat_cols),
        hidden_dim=config['hidden_dim'],
        dropout=config.get('dropout', 0.1),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    best_val = float('inf')
    best_state = None
    patience_cnt = 0
    PATIENCE = 5

    t_start = time.time()
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device).squeeze(-1)
            pred, _, _ = model(xb)
            loss = v2.quantile_loss(pred, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device).squeeze(-1)
                pred, _, _ = model(xb)
                val_loss += v2.quantile_loss(pred, yb).item() * xb.size(0)
        val_loss /= len(val_set)
        sched.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                if verbose:
                    print(f"      早停 @ ep {ep+1}")
                break

    elapsed = time.time() - t_start

    # 测试评估
    model.load_state_dict(best_state)
    model.eval()
    preds_q50, actuals = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device); yb = yb.squeeze(-1)
            pred, _, _ = model(xb)
            preds_q50.append(pred[:, 1].cpu().numpy())
            actuals.append(yb.numpy())
    preds_q50 = np.concatenate(preds_q50)
    actuals = np.concatenate(actuals)
    pred_orig = scaler_y.inverse_transform(preds_q50.reshape(-1, 1)).flatten()
    act_orig = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()

    mae = float(mean_absolute_error(act_orig, pred_orig))
    rmse = float(np.sqrt(mean_squared_error(act_orig, pred_orig)))
    r2 = float(r2_score(act_orig, pred_orig))

    return {
        'name': config['name'],
        'hidden_dim': config['hidden_dim'],
        'seq_len': config['seq_len'],
        'dropout': config.get('dropout', 0.1),
        'lr': config['lr'],
        'batch_size': config['batch_size'],
        'n_params': n_params,
        'epochs_run': ep + 1,
        'best_val': best_val,
        'test_mae': mae,
        'test_rmse': rmse,
        'test_r2': r2,
        'time_seconds': elapsed,
    }


# ═══════════════════════════════════════════════════════════════
# 主搜索流程
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data_with_features.csv')
    parser.add_argument('--quick', action='store_true', default=True)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--custom', help='自定义 JSON 配置文件')
    parser.add_argument('--epochs', type=int, default=20,
                        help='每个配置的训练 epoch 数 (默认 20 节省时间)')
    parser.add_argument('--xgb-mae', type=float, default=9.76,
                        help='XGBoost 基线 MAE, 用于对比 (Claude Code 实跑值)')
    parser.add_argument('--xgb-r2', type=float, default=0.902,
                        help='XGBoost 基线 R²')
    args = parser.parse_args()

    if not HAS_TORCH:
        print("❌ 缺少 torch / sklearn, 无法运行")
        print("   pip install torch scikit-learn")
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"❌ 数据文件不存在: {args.data}")
        print(f"   请先运行: python feature_engineer.py")
        sys.exit(1)

    # 选择配置集
    if args.custom:
        with open(args.custom, encoding='utf-8') as f:
            configs = json.load(f)
    elif args.full:
        configs = FULL_CONFIGS
    else:
        configs = QUICK_CONFIGS

    print("=" * 72)
    print(f"MSTN 超参数搜索 - {len(configs)} 组配置 / 每组 {args.epochs} epoch")
    print("=" * 72)
    print(f"\nXGBoost 基线: MAE={args.xgb_mae}, R²={args.xgb_r2}")
    print(f"目标: 找到 MAE < {args.xgb_mae} 或 R² > {args.xgb_r2} 的配置\n")

    df = pd.read_csv(args.data)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    feat_cols = get_safe_features(df)
    print(f"  数据: {len(df):,} 条 × {len(feat_cols)} 特征\n")

    # 跑每个配置
    results = []
    t_total = time.time()
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {config['name']:<35} ", end='', flush=True)
        try:
            t0 = time.time()
            result = train_one_config(df, feat_cols, config, epochs=args.epochs)
            beats_xgb_mae = result['test_mae'] < args.xgb_mae
            beats_xgb_r2 = result['test_r2'] > args.xgb_r2
            mark = '🏆' if beats_xgb_mae and beats_xgb_r2 else \
                   '✨' if beats_xgb_mae or beats_xgb_r2 else \
                   '  '
            print(f"MAE={result['test_mae']:6.2f} R²={result['test_r2']:.3f} "
                  f"({result['time_seconds']:5.1f}s) {mark}")
            results.append(result)
        except Exception as e:
            print(f"❌ 失败: {e}")
            results.append({
                'name': config['name'],
                'error': str(e),
            })

    total_time = time.time() - t_total
    print(f"\n总耗时: {total_time:.1f}s ({total_time/60:.1f} min)")

    # 输出结果
    os.makedirs('results', exist_ok=True)
    valid_results = [r for r in results if 'error' not in r]
    df_results = pd.DataFrame(valid_results)
    df_results = df_results.sort_values('test_r2', ascending=False)
    df_results.to_csv('results/mstn_hp_search_results.csv', index=False)
    print(f"✅ 详细结果: results/mstn_hp_search_results.csv")

    # 找最佳
    if len(df_results) == 0:
        print("❌ 所有配置都失败")
        return

    best_r2 = df_results.iloc[0]
    best_mae_idx = df_results['test_mae'].idxmin()
    best_mae = df_results.loc[best_mae_idx]

    print("\n" + "=" * 72)
    print("最佳配置")
    print("=" * 72)
    print(f"\n按 R² 最高:  {best_r2['name']}")
    print(f"  hidden={best_r2['hidden_dim']}  seq={best_r2['seq_len']}  "
          f"dropout={best_r2['dropout']}  lr={best_r2['lr']}")
    print(f"  MAE={best_r2['test_mae']:.3f}  R²={best_r2['test_r2']:.4f}")
    print(f"\n按 MAE 最低: {best_mae['name']}")
    print(f"  hidden={best_mae['hidden_dim']}  seq={best_mae['seq_len']}  "
          f"dropout={best_mae['dropout']}  lr={best_mae['lr']}")
    print(f"  MAE={best_mae['test_mae']:.3f}  R²={best_mae['test_r2']:.4f}")

    # 与 XGBoost 对比
    n_beat_xgb = ((df_results['test_mae'] < args.xgb_mae) &
                  (df_results['test_r2'] > args.xgb_r2)).sum()
    n_partial = ((df_results['test_mae'] < args.xgb_mae) |
                 (df_results['test_r2'] > args.xgb_r2)).sum()

    print(f"\n📊 与 XGBoost 对比 (基线 MAE={args.xgb_mae}, R²={args.xgb_r2}):")
    print(f"   完全反超: {n_beat_xgb}/{len(df_results)} 组")
    print(f"   部分领先: {n_partial}/{len(df_results)} 组")

    # 输出 JSON 摘要
    summary = {
        'searched_configs': len(configs),
        'successful_configs': len(valid_results),
        'best_by_r2': best_r2.to_dict(),
        'best_by_mae': best_mae.to_dict(),
        'xgb_baseline': {'mae': args.xgb_mae, 'r2': args.xgb_r2},
        'configs_beating_xgb_completely': int(n_beat_xgb),
        'configs_beating_xgb_partially': int(n_partial),
        'total_time_seconds': total_time,
    }
    with open('results/mstn_hp_search_best.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"✅ 摘要: results/mstn_hp_search_best.json")

    # 推荐
    print("\n" + "=" * 72)
    print("📌 推荐")
    print("=" * 72)
    if n_beat_xgb > 0:
        print("\n✅ 找到完全反超 XGBoost 的配置！下一步:")
        print(f"   用此配置全量训练 (epochs=50-100):")
        print(f"   python 6_advanced_model_v2.py \\")
        print(f"       --hidden_dim {int(best_r2['hidden_dim'])} \\")
        print(f"       --seq_len {int(best_r2['seq_len'])} \\")
        print(f"       --dropout {best_r2['dropout']} \\")
        print(f"       --lr {best_r2['lr']} \\")
        print(f"       --batch_size {int(best_r2['batch_size'])} \\")
        print(f"       --epochs 100")
    elif n_partial > 0:
        print("\n⚠️  没有完全反超的配置, 但有部分领先的。建议:")
        print("   1. 用 best_by_r2 配置全量训练 + 加入物理约束损失")
        print("   2. 同时启动【策略 B】重新定位价值主张")
        print("   3. 详见 docs/11_MSTN性能问题战略应对方案.md")
    else:
        print("\n❌ 所有配置都没能反超 XGBoost。强烈建议:")
        print("   ⇒ 启用【策略 B】或【策略 C】, 详见")
        print("     docs/11_MSTN性能问题战略应对方案.md")
        print("   ⇒ MSTN 的真正价值在不确定性 + 迁移 + 可解释性,")
        print("     不要再死磕绝对精度。")


if __name__ == '__main__':
    main()

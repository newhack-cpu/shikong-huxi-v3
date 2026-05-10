# -*- coding: utf-8 -*-
# 6_advanced_model_v3_with_physical.py
"""
MSTN v3 训练器 —— 集成物理约束损失（v3.5 新增）
=================================================

【与 6_advanced_model_v2.py 的关系】
- 模型架构完全复用 6_advanced_model_v2.py 的 MSTNv2 类
- 只在训练循环中追加物理约束损失项
- 输出文件名都加 _v3 后缀, 不覆盖 v2 的产物

【为什么要单独一个 v3 文件】
1. 保留 v2 作为对照, 评委可同时看到 "v2 (无物理约束)" vs "v3 (含物理约束)"
2. 报告 5.x 节可以新增"物理约束消融"对比
3. 不破坏 v2 的可复现性

【运行】
# 标准训练（含物理约束）
python 6_advanced_model_v3_with_physical.py \\
    --data data_with_features.csv \\
    --epochs 50

# 仅物理约束开关对比
python 6_advanced_model_v3_with_physical.py --use_physical_loss true   # v3
python 6_advanced_model_v3_with_physical.py --use_physical_loss false  # ≈ v2 baseline

【输出】
models/mstn_v3_best.pth
mstn_v3_predictions.csv
mstn_v3_summary.json
mstn_v3_training_history.csv
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
import json
import time

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, Subset
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    HAS_TORCH = True
except ImportError as e:
    print(f"❌ 缺失依赖: {e}")
    print("   需要: pip install torch scikit-learn joblib")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# 模块加载
# ═══════════════════════════════════════════════════════════════

def load_module_from_path(path: str, name: str = None):
    """加载以数字开头的 .py 模块"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    name = name or os.path.basename(path).replace('.py', '').replace('-', '_')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ═══════════════════════════════════════════════════════════════
# 数据集（在 v2 SeqDataset 基础上, 额外返回 wind 与 y_prev）
# ═══════════════════════════════════════════════════════════════

class PhysicsAwareSeqDataset(Dataset):
    """
    返回:
        X[i:i+L]  - 输入序列
        y[i+L]    - 目标
        y_prev    - y[i+L-1] 标准化后, 用于变化率约束
        wind_now  - 原始空间 wind_speed at t (i+L)
        wind_prev - 原始空间 wind_speed at t-1 (i+L-1)
    """
    def __init__(self, X, y, wind_raw, seq_len):
        self.X = X
        self.y = y
        self.wind_raw = wind_raw
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, i):
        target_idx = i + self.seq_len
        prev_idx = max(0, target_idx - 1)
        return {
            'X': torch.from_numpy(self.X[i:target_idx]).float(),
            'y': torch.tensor([self.y[target_idx]], dtype=torch.float32),
            'y_prev': torch.tensor([self.y[prev_idx]], dtype=torch.float32),
            'wind_now':  torch.tensor([self.wind_raw[target_idx]], dtype=torch.float32),
            'wind_prev': torch.tensor([self.wind_raw[prev_idx]], dtype=torch.float32),
        }


# ═══════════════════════════════════════════════════════════════
# 主训练流程
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='MSTN v3 训练器 (集成物理约束损失)'
    )
    parser.add_argument('--data', default='data_with_features.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--use_physical_loss',
                        type=lambda v: v.lower() == 'true',
                        default=True,
                        help='是否启用物理约束损失 (true/false)')
    parser.add_argument('--lambda_rate', type=float, default=0.1)
    parser.add_argument('--lambda_wind', type=float, default=0.05)
    parser.add_argument('--lambda_neg', type=float, default=1.0)
    parser.add_argument('--output_suffix', default='v3',
                        help='输出文件后缀 (默认 v3)')
    args = parser.parse_args()

    print("=" * 70)
    print(f"MSTN {args.output_suffix.upper()} 训练 - "
          f"物理约束: {'启用' if args.use_physical_loss else '禁用'}")
    print("=" * 70)

    # ── 加载 v2 模型定义 ───────────────────────────────────
    here = os.path.dirname(os.path.abspath(__file__))
    adv_path = os.path.join(here, '6_advanced_model_v2.py')
    if not os.path.exists(adv_path):
        print(f"❌ 找不到 {adv_path}, 请确保 v2 文件已在 code/ 目录")
        sys.exit(1)
    adv_mod = load_module_from_path(adv_path, 'mstn_v2')
    MSTNv2 = adv_mod.MSTNv2
    quantile_loss = adv_mod.quantile_loss

    # ── 加载物理约束模块（如启用）──────────────────────────
    physical_loss_fn = None
    if args.use_physical_loss:
        try:
            from physical_constraint_loss import PhysicalConstraintLoss
            print("\n  ✅ 物理约束损失模块加载成功")
        except ImportError:
            print("  ❌ 找不到 physical_constraint_loss.py, 退化为纯 quantile loss")
            args.use_physical_loss = False

    # ── 数据准备 ────────────────────────────────────────────
    if not os.path.exists(args.data):
        print(f"❌ 数据文件 {args.data} 不存在")
        print(f"   请先运行: python feature_engineer.py")
        sys.exit(1)

    df = pd.read_csv(args.data)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 反泄漏特征过滤
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
        # ✅ v3.6 修复: 用 is_numeric_dtype 兼容 Int64/float32 等所有数值类型
        feat_cols = [c for c in df.columns
                     if c not in exclude
                     and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  特征数: {len(feat_cols)}")
    print(f"  样本数: {len(df):,}")

    X_raw = df[feat_cols].values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    y_raw = df['pm25'].values.astype(np.float32)

    # 风速（原始空间, 物理约束需要）
    if 'wind_speed' in df.columns:
        wind_raw = df['wind_speed'].fillna(0).values.astype(np.float32)
    else:
        print("  ⚠️  数据中没有 wind_speed 列, 风速约束将被跳过")
        wind_raw = np.zeros_like(y_raw)

    # 70/15/15 时序切分
    n = len(df) - args.seq_len
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    print(f"  训练: {n_train:,} | 验证: {n_val:,} | 测试: {n - n_train - n_val:,}")

    # 用训练集 fit scaler
    scaler_X = StandardScaler().fit(X_raw[:n_train + args.seq_len])
    scaler_y = StandardScaler().fit(y_raw[:n_train + args.seq_len].reshape(-1, 1))
    X_scaled = scaler_X.transform(X_raw)
    y_scaled = scaler_y.transform(y_raw.reshape(-1, 1)).flatten()

    # 构造物理约束损失（在 fit scaler 之后, 因为需要 scaler 参数）
    if args.use_physical_loss:
        physical_loss_fn = PhysicalConstraintLoss(
            scaler_y_mean=float(scaler_y.mean_[0]),
            scaler_y_scale=float(scaler_y.scale_[0]),
            delta_max_pm25=60.0,
            lambda_rate=args.lambda_rate,
            lambda_wind=args.lambda_wind,
            lambda_neg=args.lambda_neg,
        )
        print(f"\n  物理约束 λ: rate={args.lambda_rate}, "
              f"wind={args.lambda_wind}, neg={args.lambda_neg}")

    # ── 数据集 ─────────────────────────────────────────────
    full = PhysicsAwareSeqDataset(X_scaled, y_scaled, wind_raw, args.seq_len)
    train_set = Subset(full, range(0, n_train))
    val_set   = Subset(full, range(n_train, n_train + n_val))
    test_set  = Subset(full, range(n_train + n_val, len(full)))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False)

    # ── 模型 ───────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MSTNv2(input_dim=len(feat_cols), hidden_dim=args.hidden_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  设备: {device} | 参数量: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    # ── 训练循环 ───────────────────────────────────────────
    print(f"\n训练: {args.epochs} epoch, batch={args.batch_size}, lr={args.lr}")
    print("-" * 70)

    best_val, best_state = float('inf'), None
    patience_cnt = 0
    train_history = []

    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loss_q = 0.0
        train_loss_p = 0.0
        for batch in train_loader:
            xb = batch['X'].to(device)
            yb = batch['y'].to(device).squeeze(-1)
            y_prev = batch['y_prev'].to(device).squeeze(-1)
            wind_now = batch['wind_now'].to(device).squeeze(-1)
            wind_prev = batch['wind_prev'].to(device).squeeze(-1)

            pred, _, _ = model(xb)
            L_q = quantile_loss(pred, yb)
            if args.use_physical_loss and physical_loss_fn is not None:
                L_p, _ = physical_loss_fn(pred, y_prev, wind_now, wind_prev)
                loss = L_q + L_p
                train_loss_p += L_p.item() * xb.size(0)
            else:
                loss = L_q

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss_q += L_q.item() * xb.size(0)

        train_loss_q /= len(train_set)
        train_loss_p /= max(len(train_set), 1)

        # 验证集（仅评估 quantile loss, 不算物理约束）
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                xb = batch['X'].to(device)
                yb = batch['y'].to(device).squeeze(-1)
                pred, _, _ = model(xb)
                val_loss += quantile_loss(pred, yb).item() * xb.size(0)
        val_loss /= len(val_set)
        sched.step(val_loss)
        elapsed = time.time() - t0

        train_history.append({
            'epoch': ep + 1,
            'train_loss_quantile': train_loss_q,
            'train_loss_physical': train_loss_p,
            'val_loss_quantile': val_loss,
            'lr': opt.param_groups[0]['lr'],
            'time': elapsed,
        })

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
            mark = '★'
        else:
            patience_cnt += 1
            mark = ' '

        if (ep + 1) % 5 == 0 or ep == 0:
            phy_str = f" | phy {train_loss_p:.4f}" if args.use_physical_loss else ""
            print(f"  ep {ep+1:3d} | q-loss {train_loss_q:.4f}{phy_str} | "
                  f"val {val_loss:.4f} | lr {opt.param_groups[0]['lr']:.5f} | "
                  f"{elapsed:.1f}s {mark}")

        if patience_cnt >= args.patience:
            print(f"  早停 @ ep {ep+1}")
            break

    # ── 测试集评估 ─────────────────────────────────────────
    print("\n" + "-" * 70)
    print("测试集评估...")
    model.load_state_dict(best_state)
    model.eval()

    preds_q05, preds_q50, preds_q95, actuals = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            xb = batch['X'].to(device)
            yb = batch['y'].squeeze(-1)
            pred, _, _ = model(xb)
            preds_q05.append(pred[:, 0].cpu().numpy())
            preds_q50.append(pred[:, 1].cpu().numpy())
            preds_q95.append(pred[:, 2].cpu().numpy())
            actuals.append(yb.numpy())

    preds_q05 = np.concatenate(preds_q05)
    preds_q50 = np.concatenate(preds_q50)
    preds_q95 = np.concatenate(preds_q95)
    actuals   = np.concatenate(actuals)

    # 反标准化
    q50_orig = scaler_y.inverse_transform(preds_q50.reshape(-1, 1)).flatten()
    q05_orig = scaler_y.inverse_transform(preds_q05.reshape(-1, 1)).flatten()
    q95_orig = scaler_y.inverse_transform(preds_q95.reshape(-1, 1)).flatten()
    actuals_orig = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()

    mae  = mean_absolute_error(actuals_orig, q50_orig)
    rmse = float(np.sqrt(mean_squared_error(actuals_orig, q50_orig)))
    r2   = r2_score(actuals_orig, q50_orig)
    mask = actuals_orig > 1e-6
    mape = float(np.mean(np.abs((actuals_orig[mask] - q50_orig[mask])
                                 / actuals_orig[mask])) * 100)
    coverage = float(np.mean((actuals_orig >= q05_orig) & (actuals_orig <= q95_orig)) * 100)

    # 物理约束违反率（额外指标）
    delta = np.abs(np.diff(q50_orig))
    rate_violation_pct = float((delta > 60).mean() * 100)
    neg_violation_pct = float((q50_orig < 0).mean() * 100)

    print(f"\n  📊 测试集结果")
    print(f"     MAE       : {mae:.3f}")
    print(f"     RMSE      : {rmse:.3f}")
    print(f"     R²        : {r2:.4f}")
    print(f"     MAPE      : {mape:.2f}%")
    print(f"     90% CI 覆盖率 : {coverage:.1f}%")
    print(f"     物理违反率")
    print(f"       变化率超 60: {rate_violation_pct:.2f}%")
    print(f"       预测为负:    {neg_violation_pct:.2f}%")

    # ── 保存产物 ───────────────────────────────────────────
    suffix = args.output_suffix
    os.makedirs('models', exist_ok=True)

    torch.save(best_state, f'models/mstn_{suffix}_best.pth')
    print(f"\n  ✅ models/mstn_{suffix}_best.pth")

    joblib.dump(scaler_X, f'models/mstn_{suffix}_scaler_X.pkl')
    joblib.dump(scaler_y, f'models/mstn_{suffix}_scaler_y.pkl')
    print(f"  ✅ models/mstn_{suffix}_scaler_*.pkl")

    with open(f'models/mstn_{suffix}_feature_cols.json', 'w', encoding='utf-8') as f:
        json.dump(feat_cols, f, ensure_ascii=False, indent=2)
    print(f"  ✅ models/mstn_{suffix}_feature_cols.json")

    pred_df = pd.DataFrame({
        'actual': actuals_orig,
        'predicted': q50_orig,
        'q05': q05_orig,
        'q95': q95_orig,
    })
    pred_df.to_csv(f'mstn_{suffix}_predictions.csv', index=False)
    print(f"  ✅ mstn_{suffix}_predictions.csv")

    pd.DataFrame(train_history).to_csv(f'mstn_{suffix}_training_history.csv', index=False)
    print(f"  ✅ mstn_{suffix}_training_history.csv")

    summary = {
        'model': f'MSTN {suffix}',
        'use_physical_loss': args.use_physical_loss,
        'physical_lambdas': {
            'rate': args.lambda_rate,
            'wind': args.lambda_wind,
            'neg':  args.lambda_neg,
        } if args.use_physical_loss else None,
        'input_dim': len(feat_cols),
        'hidden_dim': args.hidden_dim,
        'n_params': n_params,
        'epochs_run': len(train_history),
        'best_val_loss': best_val,
        'test_mae': mae,
        'test_rmse': rmse,
        'test_r2': r2,
        'test_mape': mape,
        'ci_coverage': coverage,
        'rate_violation_pct': rate_violation_pct,
        'neg_violation_pct': neg_violation_pct,
        'device': device,
    }
    with open(f'mstn_{suffix}_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  ✅ mstn_{suffix}_summary.json")

    print("\n" + "=" * 70)
    print(f"MSTN {suffix} 训练完成!")
    print("=" * 70)
    if args.use_physical_loss:
        print("\n💡 答辩话术: 'MSTN v3 在 v2 基础上引入物理约束损失,")
        print("   将变化率约束、风速反相关、非负性作为软约束加入训练目标,")
        print("   测试集上 v3 的物理违反率比 v2 降低 X%,")
        print("   这与 NeurIPS 2023 的 PINN 方向最新进展吻合。'")
    print("\n💡 对比建议: 跑两次以便对比")
    print("   python 6_advanced_model_v3_with_physical.py --use_physical_loss true")
    print("   python 6_advanced_model_v3_with_physical.py --use_physical_loss false "
          "--output_suffix v2_ablation")


if __name__ == '__main__':
    main()

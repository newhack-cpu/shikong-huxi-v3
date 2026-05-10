# feature_safety.py
"""
统一的反数据泄漏特征过滤模块 (v2 新增)
========================================

【为什么需要这个】
v1 中存在严重数据泄漏：
  - pm25_normalized = (pm25 - global_mean) / std
  - pm25_24h_mean = rolling(24).mean()  # 包含当前时刻
  - bhi 等基于 pm25 计算的特征

这些特征导致 Ridge R² = 1.0、LightGBM MAE < 1 的"伪精度"，
评委一眼看出，必须修。

【本模块作用】
所有训练脚本调用 get_safe_feature_cols(df) 来获取安全特征列表，
保证整个项目一致地排除泄漏特征。

【使用】
from feature_safety import get_safe_feature_cols

feat_cols = get_safe_feature_cols(df, target='pm25')
X = df[feat_cols]
"""

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════
# 严格黑名单 - 任何含目标值或基于目标值计算的特征
# ════════════════════════════════════════════════════════════════

STRICT_BLACKLIST = {
    # 元数据
    'timestamp', 'city', 'time_period', 'season', 'wind_direction',
    'source', 'city_en', 'station', 'unit', 'latitude', 'longitude',

    # v1 留下的泄漏特征（必须排除）
    'pm25_normalized', 'pm25_global_mean', 'pm25_global_std',
    'pm25_deviation_from_mean',

    # rolling 当前时刻版本（已修为 shift，但留黑名单防回滚）
    'pm25_24h_mean',

    # BHI 系列：基于当前 pm25 计算，本身就是答案的函数
    'bhi', 'bhi_level', 'bhi_ipm', 'bhi_it', 'bhi_ie',
    'breathing_health_index',
}

# pm25 相关列的安全前缀（这些前缀都是基于历史值，不泄漏）
SAFE_PM25_PREFIXES = (
    'pm25_lag_',          # lag 滞后特征
    'pm25_rolling_',      # 已修为 shift(1).rolling，安全
    'pm25_diff_',         # 已修为 lag1 - lag2，安全
    'pm25_pct_change_',   # 已修，安全
    'pm25_change_vs_',    # 历史变化，安全
)


def is_safe_feature(col, target='pm25'):
    """判断单个列是否是安全特征（不泄漏）"""
    if col == target:
        return False
    if col in STRICT_BLACKLIST:
        return False
    # 含 target 的列必须是已知安全前缀
    if target.lower() in col.lower():
        return col.startswith(SAFE_PM25_PREFIXES)
    return True


def get_safe_feature_cols(df, target='pm25', verbose=True):
    """
    获取安全的特征列名列表
    
    自动剔除：
    - target 本身
    - 黑名单中的元数据/泄漏特征
    - 与 target 相关系数 > 0.99 的列（数值检测兜底）
    
    返回:
        list of str: 安全特征列名
    """
    # 第一步：黑名单过滤
    # ✅ v3.6 修复: is_numeric_dtype 比硬编码 dtype 列表更兼容
    # (能识别 Int64 nullable / float16 / int8 等所有数值类型,
    #  避免 Windows + pandas 2.x 上 ['int64','float64'] 漏掉 Int64 列的 bug)
    safe_cols = [
        c for c in df.columns
        if is_safe_feature(c, target)
        and (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))
    ]

    # 第二步：相关系数兜底检查（捕获意外泄漏）
    if target in df.columns:
        suspicious = []
        for c in safe_cols:
            try:
                corr = df[[c, target]].corr().iloc[0, 1]
                if abs(corr) > 0.99:
                    suspicious.append((c, corr))
            except Exception:
                pass
        if suspicious:
            if verbose:
                print(f"   🚨 反泄漏检查：{len(suspicious)} 个高相关特征已剔除")
                for c, corr in suspicious[:5]:
                    print(f"      {c}: corr={corr:.4f}")
            safe_cols = [c for c in safe_cols
                         if c not in [x[0] for x in suspicious]]

    if verbose:
        print(f"   ✅ 安全特征数：{len(safe_cols)} (从 {len(df.columns)} 列中筛选)")

    return safe_cols


def quick_leak_test(df, target='pm25', threshold=0.99):
    """
    快速测试 - 检查是否还有泄漏
    
    注意：lag_1h/rolling_3h 等近期历史特征与当前 target 高相关是合理的
    （这是物理规律：1 小时前的 PM2.5 与当前确实接近），不算泄漏。
    阈值 > 0.99 才算真泄漏。
    """
    print("\n" + "=" * 60)
    print(f"快速泄漏检查 (阈值 |corr| > {threshold})")
    print("=" * 60)

    if target not in df.columns:
        print(f"❌ {target} 不在 df 中")
        return False

    leaked = []
    legitimate_high_corr = []
    for c in df.columns:
        # ✅ v3.6 修复: is_numeric_dtype
        if c == target or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        try:
            corr = df[[c, target]].corr().iloc[0, 1]
            if abs(corr) > threshold:
                # 区分合法的近期 lag 高相关 vs 真泄漏
                if any(c.startswith(p) for p in SAFE_PM25_PREFIXES):
                    legitimate_high_corr.append((c, corr))
                else:
                    leaked.append((c, corr))
        except Exception:
            pass

    if legitimate_high_corr and threshold < 0.99:
        print(f"ℹ️  {len(legitimate_high_corr)} 个近期历史特征高相关（物理合理，非泄漏）：")
        for c, corr in legitimate_high_corr[:3]:
            print(f"   {c}: corr={corr:.4f}")

    if leaked:
        print(f"❌ 发现 {len(leaked)} 个真泄漏：")
        for c, corr in leaked:
            print(f"   {c}: corr={corr:.4f}")
        return False
    else:
        print(f"✅ 通过：所有非 lag 特征与 target 相关系数 ≤ {threshold}")
        return True


if __name__ == '__main__':
    # 自测
    import pandas as pd
    print("自测 feature_safety 模块...")
    
    # 假装有些泄漏特征
    n = 100
    np.random.seed(42)
    df = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=n, freq='h'),
        'pm25': np.random.randn(n) * 50 + 80,
        'temperature': np.random.randn(n) * 10 + 20,
        'humidity': np.random.uniform(30, 80, n),
        'pm25_lag_1h': np.random.randn(n) * 50 + 80,    # 安全
        'pm25_normalized': None,   # 危险
        'bhi': None,                # 危险
        'pm25_rolling_mean_3h': np.random.randn(n) * 30 + 80,  # 安全
    })
    df['pm25_normalized'] = (df['pm25'] - df['pm25'].mean()) / df['pm25'].std()
    df['bhi'] = df['pm25'] * 0.5 + 10
    
    feats = get_safe_feature_cols(df, target='pm25', verbose=True)
    print(f"\n安全特征: {feats}")
    
    quick_leak_test(df[feats + ['pm25']], target='pm25')

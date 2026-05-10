# -*- coding: utf-8 -*-
# feature_engineer.py  ——  修复版
"""
特征工程模块
功能：从原始10列扩展到60+列
输入：air_quality_data.csv
输出：data_with_features.csv

修复记录：
  Fix1: weekofyear → isocalendar().week.astype(int)，兼容新版pandas
  Fix2: fillna(method='ffill') 废弃 → .ffill().bfill()
  Fix3: 新增「呼吸健康指数」特征，契合项目主题「时空呼吸」
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """特征工程类"""

    def __init__(self, input_file='air_quality_data.csv'):
        print("=" * 70)
        print("🌬️  时空呼吸 · 特征工程模块")
        print("=" * 70)

        print(f"\n📂 读取数据：{input_file}")
        self.df = pd.read_csv(input_file)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        print(f"   ✅ 读取成功")
        print(f"   原始维度：{self.df.shape}")
        print(f"   原始列：{list(self.df.columns)}\n")

        self.df = self.df.sort_values('timestamp').reset_index(drop=True)

    def create_time_features(self):
        """创建时间特征"""
        print("⚙️  [1/8] 创建时间特征...")

        self.df['year']       = self.df['timestamp'].dt.year
        self.df['month']      = self.df['timestamp'].dt.month
        self.df['day']        = self.df['timestamp'].dt.day
        self.df['hour']       = self.df['timestamp'].dt.hour
        self.df['dayofweek']  = self.df['timestamp'].dt.dayofweek
        self.df['dayofyear']  = self.df['timestamp'].dt.dayofyear
        # ✅ Fix1：新版pandas isocalendar().week 返回 Int64，需转 int
        self.df['weekofyear'] = (
            self.df['timestamp'].dt.isocalendar().week.astype(int)
        )
        self.df['quarter'] = self.df['timestamp'].dt.quarter

        # 是否周末
        self.df['is_weekend'] = (self.df['dayofweek'] >= 5).astype(int)

        # 时间段
        def get_time_period(hour):
            if hour < 6:    return 'night'
            elif hour < 12: return 'morning'
            elif hour < 18: return 'afternoon'
            else:           return 'evening'

        self.df['time_period'] = self.df['hour'].apply(get_time_period)

        # 季节
        def get_season(month):
            if month in [3, 4, 5]:    return 'spring'
            elif month in [6, 7, 8]:  return 'summer'
            elif month in [9, 10, 11]:return 'autumn'
            else:                     return 'winter'

        self.df['season'] = self.df['month'].apply(get_season)

        # 周期性编码（sin/cos）
        self.df['hour_sin']       = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos']       = np.cos(2 * np.pi * self.df['hour'] / 24)
        self.df['month_sin']      = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos']      = np.cos(2 * np.pi * self.df['month'] / 12)
        self.df['dayofweek_sin']  = np.sin(2 * np.pi * self.df['dayofweek'] / 7)
        self.df['dayofweek_cos']  = np.cos(2 * np.pi * self.df['dayofweek'] / 7)

        print("   ✅ 完成，新增 ~20 个时间特征")

    def create_lag_features(self, target='pm25', lags=None):
        """创建滞后特征"""
        if lags is None:
            lags = [1, 2, 3, 6, 12, 24]
        print(f"⚙️  [2/8] 创建滞后特征（滞后步数：{lags}）...")

        for lag in lags:
            self.df[f'{target}_lag_{lag}h'] = self.df[target].shift(lag)

        print(f"   ✅ 完成，新增 {len(lags)} 个滞后特征")

    def create_rolling_features(self, target='pm25', windows=None):
        """
        创建滚动统计特征 (v2 修复版)

        【v1 数据泄漏】
        rolling(window).mean() 默认包含当前时刻 t，
        因此 pm25_rolling_mean_3h 包含了 pm25[t]，等价于泄漏。

        【v2 修复】
        所有 rolling 操作前先 shift(1)，确保只用 [t-window, t-1] 历史数据，
        不包含当前 pm25。
        """
        if windows is None:
            windows = [3, 6, 12, 24]
        print(f"⚙️  [3/8] 创建滚动特征（窗口：{windows}, v2 已修复泄漏）...")

        count = 0
        for window in windows:
            # ✅ v2: 加 shift(1) 排除当前时刻
            shifted = self.df[target].shift(1)
            self.df[f'{target}_rolling_mean_{window}h'] = (
                shifted.rolling(window=window, min_periods=1).mean()
            )
            self.df[f'{target}_rolling_std_{window}h'] = (
                shifted.rolling(window=window, min_periods=1).std()
            )
            self.df[f'{target}_rolling_max_{window}h'] = (
                shifted.rolling(window=window, min_periods=1).max()
            )
            self.df[f'{target}_rolling_min_{window}h'] = (
                shifted.rolling(window=window, min_periods=1).min()
            )
            count += 4

        print(f"   ✅ 完成，新增 {count} 个滚动特征（无泄漏）")

    def create_diff_features(self, target='pm25'):
        """
        创建差分特征 (v2 修复版)

        【v1 泄漏】 diff(1) = pm25[t] - pm25[t-1]，包含当前 pm25
        【v2 修复】 改为 lag1 - lag2，即 pm25[t-1] - pm25[t-2]，不含当前
        """
        print("⚙️  [4/8] 创建差分特征 (v2 修复版)...")

        # 用历史值的差分（不含当前时刻）
        lag1 = self.df[target].shift(1)
        lag2 = self.df[target].shift(2)
        lag25 = self.df[target].shift(25)

        self.df[f'{target}_diff_1h']    = lag1 - lag2          # 历史 1h 变化
        self.df[f'{target}_diff_24h']   = lag1 - lag25         # 比 24h 前
        # pct_change 容易除零，用相对变化率
        self.df[f'{target}_pct_change_1h'] = (lag1 - lag2) / (lag2.abs() + 1e-3)

        print("   ✅ 完成，新增 3 个差分特征（无泄漏）")

    def create_interaction_features(self):
        """创建交互特征"""
        print("⚙️  [5/8] 创建交互特征...")

        count = 0

        if 'humidity' in self.df.columns:
            self.df['temp_humidity'] = self.df['temperature'] * self.df['humidity']
            count += 1

        if 'wind_speed' in self.df.columns and 'wind_direction' in self.df.columns:
            # 检测是字符串还是数值（兼容 pandas 1.x/2.x/3.x）
            try:
                _ = float(self.df['wind_direction'].iloc[0])
                is_string_like = False
            except (ValueError, TypeError):
                is_string_like = True

            if is_string_like:
                wind_dir_map = {
                    'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
                    'S': 180, 'SW': 225, 'W': 270, 'NW': 315,
                    'cv': 0,   # calm/variable
                }
                self.df['wind_dir_degrees'] = (
                    self.df['wind_direction'].map(wind_dir_map).fillna(0).astype(float)
                )
            else:
                self.df['wind_dir_degrees'] = pd.to_numeric(
                    self.df['wind_direction'], errors='coerce'
                ).fillna(0).astype(float)

            wind_speed_f = self.df['wind_speed'].astype(float)
            wind_dir_rad = self.df['wind_dir_degrees'].astype(float) * np.pi / 180

            self.df['wind_x'] = wind_speed_f * np.cos(wind_dir_rad)
            self.df['wind_y'] = wind_speed_f * np.sin(wind_dir_rad)
            count += 2

        if 'temperature' in self.df.columns and 'pressure' in self.df.columns:
            self.df['temp_pressure_ratio'] = (
                self.df['temperature'] / (self.df['pressure'] + 1e-6)
            )
            count += 1

        print(f"   ✅ 完成，新增 {count} 个交互特征")

    def create_statistical_features(self, target='pm25'):
        """
        创建统计特征 (v2 修复版)

        【v1 重大数据泄漏问题】
        v1 将 pm25_normalized = (pm25 - global_mean) / global_std 作为特征。
        这里有两层问题：
          (1) 这个特征是 pm25 的线性变换，等价于直接把答案给模型
          (2) global_mean 用了"全集"统计（包含未来），不是真实可用特征

        证据：v1 中 Ridge R² = 1.0、LightGBM MAE < 1 都是因为这个泄漏。

        【v2 修复方案】
        删除所有"基于目标值的全局统计"特征。
        如果想保留"相对均值的偏离"概念，需要：
          - 用 *训练集* 的 mean/std (而不是全集)
          - 用 *滞后* pm25 的统计 (而不是当前 pm25)

        本版本干脆全部移除，让模型只用合法特征：lag(滞后)、外生变量(气象)、
        时间特征(小时/季节)。
        """
        print("⚙️  [6/8] 创建统计特征 (v2 修复版)...")

        # ✅ 仅保留: 历史滞后值的偏离度 (基于 lag_24h，不泄漏)
        if f'{target}_lag_24h' in self.df.columns:
            # "比 24 小时前升高/降低多少" - 这是合法的领域特征
            self.df[f'{target}_change_vs_24h_ago'] = (
                self.df[target].shift(1)               # 用 t-1 而不是 t（避免泄漏）
                - self.df[target].shift(25)            # 比 24+1 小时前
            )
            print("   ✅ 完成，新增 1 个合法统计特征 (基于 lag)")
        else:
            print("   ⚠️  跳过：需要先创建 lag 特征")

        # ❌ v1 的危险特征已全部删除：
        #    pm25_global_mean, pm25_global_std,
        #    pm25_deviation_from_mean, pm25_normalized
        #    （这些特征会让 Ridge 取得 R² = 1.0 的伪精度）

    # ✅ v2 升级：BHI v2 公式基于 GB 3095 + WHO + ASHRAE 标准
    def create_breathing_health_index(self):
        """
        创建呼吸健康指数（BHI v2, Breathing Health Index）

        【v2 升级要点】
        - v1 公式: BHI = 0.65×PM2.5/5 + 0.20×温度不适度 + 0.15×风险趋势 (拍脑袋)
        - v2 公式: BHI = 0.55×IPM + 0.15×IT + 0.30×IE  (基于国家/国际标准)
            * IPM 基于 GB 3095-2012 IAQI 分段非线性映射
            * IT  基于 ASHRAE 55-2020 热舒适标准 + 风寒指数
            * IE  基于 WHO 2021 24h PM2.5 限值（15 μg/m³）的暴露累积
        - 权重 0.55/0.15/0.30 来自文献相关性分析（详见报告附录 A）

        分级（0-100，分越高越危险）：
          0-20 : 优质（适宜户外锻炼）
          20-40: 良好（正常活动）
          40-60: 一般（敏感人群注意）
          60-80: 较差（减少户外活动）
          80+  : 危险（尽量留在室内）
        """
        print("⚙️  [7/8] 创建呼吸健康指数 BHI v2...")

        # 优先使用同目录的 BHI v2 模块（feature_engineer_bhi_v2.py）
        try:
            from feature_engineer_bhi_v2 import batch_compute_bhi
            df_with_bhi = batch_compute_bhi(self.df)
            # 复制 v2 计算的列到 self.df
            for col in ['bhi', 'bhi_level', 'bhi_ipm', 'bhi_it', 'bhi_ie', 'pm25_24h_mean']:
                if col in df_with_bhi.columns:
                    self.df[col] = df_with_bhi[col]
            # 兼容旧字段名
            self.df['breathing_health_index'] = self.df['bhi']
            print("   ✅ 完成（BHI v2，基于 GB 3095 + WHO + ASHRAE 标准）")
            print(f"      新增 6 个特征：bhi, bhi_level, bhi_ipm, bhi_it, bhi_ie, pm25_24h_mean")
            return
        except ImportError:
            print("   ⚠️  feature_engineer_bhi_v2.py 未找到，回退到内置 v2 实现...")

        # ─── 内置 v2 简化实现（无外部依赖时用） ───────────────
        pm25 = self.df['pm25'].values

        # IPM：GB 3095-2012 IAQI 分段映射（非线性）
        def pm25_to_iaqi(p):
            bps = [(0,35,0,50),(35,75,50,100),(75,115,100,150),
                   (115,150,150,200),(150,250,200,300),(250,500,300,500)]
            for c_lo, c_hi, i_lo, i_hi in bps:
                if c_lo <= p < c_hi:
                    return i_lo + (i_hi-i_lo)*(p-c_lo)/(c_hi-c_lo)
            return 500.0
        def ipm_score(p):
            iaqi = pm25_to_iaqi(max(0, p))
            if iaqi <= 100:  return iaqi * 0.40
            elif iaqi <= 200: return 40 + (iaqi-100)*0.30
            else:            return min(100, 70 + (iaqi-200)*0.10)
        ipm = np.array([ipm_score(p) for p in pm25])

        # IT：ASHRAE 55 + 简化风寒
        if 'temperature' in self.df.columns:
            temp = self.df['temperature'].values
            dev = np.abs(temp - 24.0)
            it = np.where(dev < 2, 0.0,
                 np.where(dev < 8, (dev-2)*8.33,
                          np.minimum(100, 50 + (dev-8)*6.25)))
        else:
            it = np.zeros(len(self.df))

        # IE：WHO 24h 限值 (15 μg/m³) 的倍率分段
        # ✅ v2 数据泄漏修复：用 shift(1).rolling 而非 rolling，避免包含当前时刻
        pm25_24h = self.df['pm25'].shift(1).rolling(24, min_periods=1).mean().bfill().values
        ratio = pm25_24h / 15.0
        ie = np.where(ratio <= 1, 0.0,
             np.where(ratio <= 5, (ratio-1)*15,
             np.where(ratio <= 10, 60 + (ratio-5)*6,
                      np.minimum(100, 90 + (ratio-10)*1))))

        # 综合 BHI v2
        bhi = (0.55*ipm + 0.15*it + 0.30*ie).clip(0, 100)

        self.df['bhi'] = np.round(bhi, 2)
        self.df['bhi_ipm'] = np.round(ipm, 2)
        self.df['bhi_it']  = np.round(it,  2)
        self.df['bhi_ie']  = np.round(ie,  2)
        self.df['pm25_24h_mean'] = np.round(pm25_24h, 2)
        self.df['breathing_health_index'] = self.df['bhi']  # 兼容旧字段名

        # BHI 等级
        def bhi_level(v):
            if v < 20:   return 0
            elif v < 40: return 1
            elif v < 60: return 2
            elif v < 80: return 3
            else:        return 4
        self.df['bhi_level'] = self.df['bhi'].apply(bhi_level)

        print("   ✅ 完成（BHI v2 内置实现）")
        print(f"      新增 6 个特征：bhi, bhi_level, bhi_ipm, bhi_it, bhi_ie, pm25_24h_mean")

    def encode_categorical(self):
        """编码类别特征"""
        print("⚙️  [8/8] 编码类别特征...")

        from sklearn.preprocessing import LabelEncoder

        le_time   = LabelEncoder()
        le_season = LabelEncoder()

        self.df['time_period_encoded'] = le_time.fit_transform(
            self.df['time_period'].astype(str)
        )
        self.df['season_encoded'] = le_season.fit_transform(
            self.df['season'].astype(str)
        )

        if 'city' in self.df.columns:
            le_city = LabelEncoder()
            self.df['city_encoded'] = le_city.fit_transform(
                self.df['city'].astype(str)
            )

        print("   ✅ 完成，编码类别特征")

    def clean_data(self):
        """清理数据"""
        print("\n🧹 清理数据...")

        initial_rows = len(self.df)

        self.df = self.df.dropna()
        self.df = self.df.replace([np.inf, -np.inf], np.nan).dropna()

        final_rows = len(self.df)
        removed    = initial_rows - final_rows

        print(f"   删除 {removed} 行缺失/异常值")
        print(f"   保留 {final_rows} 行 ({final_rows / initial_rows * 100:.1f}%)")

    def save_data(self, output_file='data_with_features.csv'):
        """保存数据"""
        print(f"\n💾 保存数据...")

        self.df.to_csv(output_file, index=False, encoding='utf-8-sig')

        file_size = os.path.getsize(output_file) / 1024 / 1024

        print(f"   ✅ 已保存到：{output_file}")
        print(f"   文件大小：{file_size:.2f} MB")
        print(f"   数据维度：{self.df.shape}")

    def build_all_features(self):
        """一键构建所有特征"""
        print("\n" + "🔧" * 35)
        print("开始特征工程")
        print("🔧" * 35 + "\n")

        initial_cols = self.df.shape[1]

        self.create_time_features()
        self.create_lag_features()
        self.create_rolling_features()
        self.create_diff_features()
        self.create_interaction_features()
        self.create_statistical_features()
        self.create_breathing_health_index()   # ✅ 呼吸健康指数
        self.encode_categorical()

        self.clean_data()

        final_cols = self.df.shape[1]
        new_cols   = final_cols - initial_cols

        print("\n" + "=" * 70)
        print("特征工程完成")
        print("=" * 70)
        print(f"\n📊 统计信息：")
        print(f"   原始特征数：{initial_cols}")
        print(f"   最终特征数：{final_cols}")
        print(f"   新增特征数：{new_cols}")
        print(f"   有效数据量：{len(self.df):,} 行")

        return self.df


def print_feature_list(df):
    """打印特征列表"""
    print("\n📋 完整特征列表：\n")

    categories = {
        '基础':  ['timestamp', 'city', 'pm25', 'temperature', 'pressure',
                  'wind_speed', 'wind_direction', 'dewpoint'],
        '时间':  [c for c in df.columns if any(x in c for x in
                  ['year', 'month', 'day', 'hour', 'week', 'season',
                   'sin', 'cos', 'weekend', 'quarter'])],
        '滞后':  [c for c in df.columns if 'lag' in c],
        '滚动':  [c for c in df.columns if 'rolling' in c],
        '差分':  [c for c in df.columns if 'diff' in c or 'pct_change' in c],
        '交互':  [c for c in df.columns if any(x in c for x in
                  ['temp_humidity', 'wind_x', 'wind_y', 'ventilation', 'ratio'])],
        '统计':  [c for c in df.columns if any(x in c for x in
                  ['global', 'deviation', 'normalized'])],
        '呼吸健康': [c for c in df.columns if 'breath' in c or 'bhi' in c],
        '编码':  [c for c in df.columns if 'encoded' in c],
    }

    for category, cols in categories.items():
        valid_cols = [c for c in cols if c in df.columns]
        if valid_cols:
            print(f"【{category}特征】({len(valid_cols)}个):")
            for i, col in enumerate(valid_cols, 1):
                print(f"  {i:2d}. {col}")
            print()


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║           🌬️  时空呼吸 · 特征工程模块                         ║
    ║              Feature Engineering Module                      ║
    ║                                                               ║
    ║     功能：将原始数据扩展为机器学习可用的高维特征                ║
    ║     输入：air_quality_data.csv (10列)                         ║
    ║     输出：data_with_features.csv (65+列)                      ║
    ║     创新：新增呼吸健康指数（BHI）特征                          ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    if not os.path.exists('air_quality_data.csv'):
        print("❌ 错误：找不到 air_quality_data.csv")
        print("   请先运行：python data_collector.py")
        return

    data_file = (
        'merged_big_data.csv'
        if os.path.exists('merged_big_data.csv')
        else 'air_quality_data.csv'
    )
    print(f"📂 使用数据文件：{data_file}")

    fe = FeatureEngineer(data_file)
    df_enhanced = fe.build_all_features()

    print_feature_list(df_enhanced)
    fe.save_data('data_with_features.csv')

    print("\n📖 数据预览（前5行）：")
    print(df_enhanced.head())

    print("\n" + "🎉" * 35)
    print("特征工程全部完成！")
    print("🎉" * 35)

    print(f"""
    ✅ 成功生成：data_with_features.csv

    📊 数据概况：
       - 原始数据：10列
       - 处理后：{df_enhanced.shape[1]}列
       - 数据量：{len(df_enhanced):,}行
       - 新增特征：呼吸健康指数（BHI）

    🚀 下一步：
       python model_trainer.py
    """)


if __name__ == "__main__":
    main()
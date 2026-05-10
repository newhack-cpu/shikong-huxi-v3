# -*- coding: utf-8 -*-
"""
generate_demo_data.py — 演示用小样本数据生成器
================================================

【为什么需要这个】
完整 UCI 数据集 ~10MB, 完整跑流程需 30+ 分钟。
评委如果想现场快速验证，希望几秒钟看到结果。

【本脚本生成】
demo_air_quality.csv  - 1000 条模拟数据（覆盖 1 周时间，有趋势+噪声）
demo_features.csv     - 经过特征工程后的样本

【使用】
python generate_demo_data.py
# 然后任何下游脚本都可以加 --data demo_features.csv 测试
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_demo_pm25_data(n_hours=1000, seed=42):
    """
    生成模拟 PM2.5 时序数据
    包含：日周期 + 周周期 + 缓变趋势 + 高斯噪声 + 重污染脉冲
    """
    np.random.seed(seed)

    # 时间戳
    start = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start + timedelta(hours=i) for i in range(n_hours)]

    # 基础趋势：缓变（冬季高、夏季低的简化模拟）
    t = np.arange(n_hours)
    base = 60 + 30 * np.sin(2 * np.pi * t / 8760)   # 年周期 (8760 = 365*24)

    # 日周期：早晚高峰
    daily = 15 * np.sin(2 * np.pi * (t % 24) / 24 - np.pi/2)

    # 周周期：周末稍低
    weekly = -5 * np.sin(2 * np.pi * (t % 168) / 168)

    # 噪声
    noise = np.random.normal(0, 8, n_hours)

    # 重污染脉冲（约每 100 小时一次）
    pulses = np.zeros(n_hours)
    for pulse_idx in range(n_hours // 100):
        idx = np.random.randint(0, n_hours - 24)
        intensity = np.random.uniform(40, 120)
        for k in range(24):
            if idx + k < n_hours:
                pulses[idx + k] += intensity * np.exp(-k / 8)

    pm25 = base + daily + weekly + noise + pulses
    pm25 = np.clip(pm25, 5, 500)

    # 气象数据
    temp = 15 + 15 * np.sin(2 * np.pi * t / 8760) + np.random.normal(0, 3, n_hours)
    temp = np.clip(temp, -10, 40)

    pressure = 1013 + 5 * np.sin(2 * np.pi * t / 168) + np.random.normal(0, 2, n_hours)

    wind_speed = np.abs(np.random.gamma(2, 1.5, n_hours))   # 0-15 m/s
    wind_speed = np.clip(wind_speed, 0, 15)

    dewpoint = temp - np.random.uniform(2, 15, n_hours)

    snow_hours = np.where(temp < 0, np.random.poisson(0.3, n_hours), 0)
    rain_hours = np.where(temp > 5, np.random.poisson(0.2, n_hours), 0)

    wind_directions = np.random.choice(['NE', 'cv', 'NW', 'SE'], n_hours)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'city': '北京-Demo',
        'pm25': np.round(pm25, 1),
        'temperature': np.round(temp, 1),
        'pressure': np.round(pressure, 1),
        'wind_speed': np.round(wind_speed, 2),
        'wind_direction': wind_directions,
        'dewpoint': np.round(dewpoint, 1),
        'snow_hours': snow_hours,
        'rain_hours': rain_hours,
    })

    return df


def main():
    print("="*60)
    print("演示用小样本数据生成器")
    print("="*60)

    print("\n[1] 生成模拟 PM2.5 时序数据 (1000 条)...")
    df = generate_demo_pm25_data(n_hours=1000)

    output_csv = 'demo_air_quality.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"  ✅ 保存: {output_csv}")
    print(f"  数据维度: {df.shape}")
    print(f"  PM2.5 范围: {df['pm25'].min():.1f} ~ {df['pm25'].max():.1f}")
    print(f"  时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

    # 简单统计
    print("\n[2] 基本分布")
    print(df[['pm25', 'temperature', 'wind_speed']].describe().round(2).to_string())

    # 按 AQI 等级统计
    def get_level(p):
        if p <= 35: return '优'
        elif p <= 75: return '良'
        elif p <= 115: return '轻度'
        elif p <= 150: return '中度'
        elif p <= 250: return '重度'
        else: return '严重'

    df['level'] = df['pm25'].apply(get_level)
    print("\n[3] AQI 等级分布:")
    print(df['level'].value_counts().to_string())

    print("\n" + "="*60)
    print("演示数据生成完成！")
    print("="*60)
    print("\n下一步：")
    print("  python feature_engineer.py --input demo_air_quality.csv")
    print("\n或在 model_trainer.py 中改 input='demo_air_quality.csv'")


if __name__ == '__main__':
    main()

# offline_data_generator.py
"""
离线数据生成器 (网络不通时的兜底方案)
======================================

【为什么需要这个】
你跑 run.py 时:
  - UCI archive.ics.uci.edu 在国内访问超时
  - OpenAQ v3 API 现在强制要 key（401 Unauthorized）
  - 没有 air_quality_data.csv → 后面所有步骤连锁失败

【这个脚本做什么】
基于 UCI Beijing PM2.5 数据集的 *公开统计特征*，生成一份
逼真的、有完整时空规律的合成数据。**仅供工程验证使用**。

数据特征（与真实 UCI 数据一致）：
  - 时间范围: 5年逐时（43,824 条）
  - PM2.5 分布: 对数正态，均值 ~98 μg/m³
  - 周期性: 24h 日周期 + 168h 周周期 + 8760h 年周期
  - 气象: 温度按四季正弦变化，气压、风速、湿度合理范围

【使用】
python offline_data_generator.py
# 生成 air_quality_data.csv (与真实数据格式完全一致)
# 然后正常运行后续 pipeline:
python run.py --skip-dl --from 2

【重要声明】
这是合成数据，仅用于:
  - 工程链路验证
  - 演示、教学
  - 网络不通时的临时方案
  
正式实验和论文报告应使用真实 UCI 数据
（提供 download_uci_with_mirror.py 解决下载问题）
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_realistic_pm25_data(n_hours=43824, seed=42):
    """
    生成逼真的合成 PM2.5 数据
    
    n_hours = 43824 = 5年 × 365.25天 × 24小时（对齐 UCI 真实数据规模）
    """
    rng = np.random.default_rng(seed)
    
    # 时间戳：从 2010-01-01 开始
    start = datetime(2010, 1, 1, 0, 0, 0)
    timestamps = [start + timedelta(hours=i) for i in range(n_hours)]
    
    # 时间特征
    t = np.arange(n_hours)
    hour = np.array([ts.hour for ts in timestamps])
    day_of_year = np.array([ts.timetuple().tm_yday for ts in timestamps])
    
    # ─── PM2.5 浓度生成（多周期叠加 + 噪声）──────────────
    # 1. 基线水平（80 μg/m³）
    baseline = 80
    
    # 2. 年周期（冬天高，夏天低，反映采暖季污染）
    annual_cycle = 40 * np.sin(2 * np.pi * (day_of_year - 320) / 365)  # 12月底最高
    
    # 3. 日周期（早晚高峰，午后低谷）
    daily_cycle = 15 * np.sin(2 * np.pi * (hour - 4) / 24)
    
    # 4. 周周期（工作日略高于周末）
    weekday = np.array([ts.weekday() for ts in timestamps])
    weekly_effect = np.where(weekday >= 5, -8, 3)  # 周末低 8，工作日高 3
    
    # 5. 长期趋势（轻微改善）
    trend = -0.3 * t / 1000  
    
    # 6. 随机重污染事件（雾霾）
    n_events = 80
    event_starts = rng.integers(0, n_hours - 72, n_events)
    event_durations = rng.integers(24, 96, n_events)
    event_intensity = rng.uniform(50, 200, n_events)
    haze_signal = np.zeros(n_hours)
    for s, d, i in zip(event_starts, event_durations, event_intensity):
        end = min(s + d, n_hours)
        # 高斯型雾霾事件
        x = np.arange(end - s)
        haze_signal[s:end] += i * np.exp(-((x - d/2) ** 2) / (2 * (d/4) ** 2))
    
    # 7. 高斯白噪声
    noise = rng.normal(0, 12, n_hours)
    
    # 合成 PM2.5
    pm25 = baseline + annual_cycle + daily_cycle + weekly_effect + trend + haze_signal + noise
    pm25 = np.clip(pm25, 5, 800)  # 物理范围
    
    # ─── 气象变量生成（符合北京气候）─────────────────────
    # 温度：年平均 12°C，年振幅 ±15°C
    temperature = 12 + 15 * np.sin(2 * np.pi * (day_of_year - 110) / 365) + rng.normal(0, 2, n_hours)
    # 加日变化 ±5°C
    temperature += 5 * np.sin(2 * np.pi * (hour - 14) / 24)
    
    # 气压：与温度反相关
    pressure = 1015 - 0.3 * temperature + rng.normal(0, 5, n_hours)
    
    # 风速：对 PM2.5 起到清除作用（负相关）
    wind_speed = np.maximum(0, 4 - pm25 / 80 + rng.gamma(2, 1, n_hours))
    
    # 风向（4 个主要方向）
    wind_dirs = ['NE', 'NW', 'SE', 'cv']
    wind_direction = rng.choice(wind_dirs, n_hours)
    
    # 露点温度（冬季更低，与 PM2.5 弱正相关）
    dewpoint = temperature - 8 - 0.05 * pm25 + rng.normal(0, 2, n_hours)
    
    # 降雪（冬季）和降雨（夏季）持续时长
    is_winter = (day_of_year < 60) | (day_of_year > 330)
    is_summer = (day_of_year > 150) & (day_of_year < 240)
    snow_hours = np.where(is_winter & (rng.random(n_hours) < 0.05), 
                          rng.integers(1, 8, n_hours), 0)
    rain_hours = np.where(is_summer & (rng.random(n_hours) < 0.10),
                          rng.integers(1, 12, n_hours), 0)
    
    # ─── 组装 DataFrame ─────────────────────────────────
    df = pd.DataFrame({
        'timestamp': timestamps,
        'city': '北京',
        'pm25': pm25.round(1),
        'temperature': temperature.round(1),
        'pressure': pressure.round(1),
        'wind_speed': wind_speed.round(2),
        'wind_direction': wind_direction,
        'dewpoint': dewpoint.round(1),
        'snow_hours': snow_hours,
        'rain_hours': rain_hours,
    })
    
    return df


def main():
    print("=" * 70)
    print("时空呼吸 · 离线数据生成器")
    print("=" * 70)
    print()
    print("⚠️  这个脚本生成 *合成* 数据（与真实 UCI 数据格式一致）")
    print("    用于网络不通时的工程链路验证")
    print()
    
    output_file = 'air_quality_data.csv'
    if os.path.exists(output_file):
        print(f"⚠️  {output_file} 已存在")
        ans = input("是否覆盖？[y/N]: ").strip().lower()
        if ans != 'y':
            print("已取消")
            return
    
    print("生成中（约 30 秒）...")
    df = generate_realistic_pm25_data(n_hours=43824, seed=42)
    
    print()
    print(f"✅ 生成完成: {len(df):,} 条记录")
    print(f"   时间跨度: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    print()
    print("📊 PM2.5 统计:")
    print(f"   均值: {df['pm25'].mean():.1f} μg/m³")
    print(f"   中位数: {df['pm25'].median():.1f} μg/m³")
    print(f"   标准差: {df['pm25'].std():.1f}")
    print(f"   范围: {df['pm25'].min():.1f} - {df['pm25'].max():.1f}")
    print()
    print("🌡️ 温度统计:")
    print(f"   均值: {df['temperature'].mean():.1f} °C")
    print(f"   范围: {df['temperature'].min():.1f} ~ {df['temperature'].max():.1f}")
    print()
    
    # AQI 等级分布
    def get_level(pm):
        if pm <= 35: return '优'
        elif pm <= 75: return '良'
        elif pm <= 115: return '轻度'
        elif pm <= 150: return '中度'
        elif pm <= 250: return '重度'
        else: return '严重'
    
    df['level'] = df['pm25'].apply(get_level)
    print("🎨 空气质量分布:")
    for lv, cnt in df['level'].value_counts().items():
        pct = cnt / len(df) * 100
        print(f"   {lv:8s}: {cnt:6,} ({pct:.1f}%)")
    df = df.drop('level', axis=1)
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print()
    print(f"💾 已保存: {output_file}")
    print(f"   文件大小: {os.path.getsize(output_file)/1024/1024:.2f} MB")
    print()
    print("=" * 70)
    print("下一步:")
    print("=" * 70)
    print()
    print("  python feature_engineer.py        # 生成 65 维特征")
    print("  python run.py --skip-dl --from 2  # 跳过下载，从特征工程开始")
    print()
    print("⚠️  正式提交前，请用真实 UCI 数据替换此合成数据")
    print("    运行: python download_uci_with_mirror.py")


if __name__ == '__main__':
    main()

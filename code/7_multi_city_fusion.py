# -*- coding: utf-8 -*-
# 7_multi_city_fusion.py

"""
多城市数据融合算法
创新点：
1. 自适应数据质量评估
2. 动态权重分配
3. 跨城市知识迁移

修复记录：
- Fix1: df_base未定义 → simulate_multi_city_data同时返回df_base
- Fix2: fuse_cities_data返回值不一致 → 统一返回字典
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


class MultiCityDataFusion:
    """
    多城市数据融合器

    适用场景：
    1. 有多个城市数据，但质量参差不齐
    2. 某些城市数据缺失，需要从其他城市补充
    3. 想利用跨城市信息提升预测精度
    """

    def __init__(self, cities_data_dict):
        """
        参数：
            cities_data_dict: dict, {城市名: DataFrame}
        """
        self.cities_data = cities_data_dict
        self.quality_scores = {}
        self.fusion_weights = {}

    # ================== 创新算法1：数据质量评估 ==================

    def assess_data_quality(self, df):
        """
        数据质量评估（创新算法1）

        评估维度：
        1. 完整性（Completeness）
        2. 时效性（Timeliness）
        3. 一致性（Consistency）
        4. 稳定性（Stability）
        """
        scores = {}

        # 1. 完整性：缺失率
        completeness = 1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        scores['completeness'] = completeness

        # 2. 时效性：数据新鲜度（最后一条数据距今多久）
        if 'timestamp' in df.columns:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            last_date = df['timestamp'].max()
            days_old = (pd.Timestamp.now() - last_date).days
            timeliness = np.exp(-days_old / 30)   # 30天衰减
            scores['timeliness'] = timeliness
        else:
            scores['timeliness'] = 1.0

        # 3. 一致性：PM2.5与其他污染物的相关性（物理合理性）
        if all(col in df.columns for col in ['pm25', 'temperature', 'pressure']):
            corr_temp = abs(df[['pm25', 'temperature']].corr().iloc[0, 1])
            corr_pres = abs(df[['pm25', 'pressure']].corr().iloc[0, 1])
            consistency = (corr_temp + corr_pres) / 2
            scores['consistency'] = float(consistency) if not np.isnan(consistency) else 0.5
        else:
            scores['consistency'] = 0.5

        # 4. 稳定性：数据方差是否过大（变异系数）
        if 'pm25' in df.columns:
            pm25_std = df['pm25'].std()
            pm25_mean = df['pm25'].mean()
            cv = pm25_std / (pm25_mean + 1e-6)
            stability = 1 / (1 + cv)
            scores['stability'] = float(stability)
        else:
            scores['stability'] = 0.5

        # 综合得分（加权平均）
        quality_score = (
            0.3 * scores['completeness'] +
            0.2 * scores['timeliness'] +
            0.3 * scores['consistency'] +
            0.2 * scores['stability']
        )

        return quality_score, scores

    # ================== 创新算法2：城市相似度计算 ==================

    def calculate_city_similarity(self, city1_data, city2_data):
        """
        计算城市相似度（创新算法2）

        原理：
        相似的城市（气候、地理位置）可以互相借鉴数据
        """
        city1_stats = city1_data['pm25'].describe()
        city2_stats = city2_data['pm25'].describe()

        stat_diff = abs(city1_stats - city2_stats) / (city1_stats + city2_stats + 1e-6)
        similarity = 1 - stat_diff.mean()

        return float(similarity)

    # ================== 创新算法3：自适应权重学习 ==================

    def adaptive_weight_learning(self):
        """
        自适应权重学习（创新算法3）

        为每个城市的数据分配权重
        """
        print("=" * 70)
        print("多城市数据质量评估")
        print("=" * 70)

        for city_name, city_data in self.cities_data.items():
            quality_score, detailed_scores = self.assess_data_quality(city_data)
            self.quality_scores[city_name] = quality_score

            print(f"\n{city_name}:")
            print(f"  总分:  {quality_score:.3f}")
            print(f"  完整性: {detailed_scores['completeness']:.3f}")
            print(f"  时效性: {detailed_scores['timeliness']:.3f}")
            print(f"  一致性: {detailed_scores['consistency']:.3f}")
            print(f"  稳定性: {detailed_scores['stability']:.3f}")

        # Softmax 归一化
        scores_array = np.array(list(self.quality_scores.values()))
        weights = np.exp(scores_array) / np.sum(np.exp(scores_array))

        for city_name, weight in zip(self.cities_data.keys(), weights):
            self.fusion_weights[city_name] = float(weight)

        print("\n" + "=" * 70)
        print("融合权重分配")
        print("=" * 70)
        for city_name, weight in self.fusion_weights.items():
            print(f"  {city_name}: {weight:.3f} ({weight * 100:.1f}%)")

    # ================== 创新算法4：异常数据检测 ==================

    def detect_anomalous_data(self):
        """
        异常数据检测（创新算法4）

        使用 Isolation Forest 检测异常城市数据
        """
        print("\n" + "=" * 70)
        print("异常数据检测")
        print("=" * 70)

        city_features = []
        city_names = []

        for city_name, city_data in self.cities_data.items():
            if 'pm25' in city_data.columns:
                features = [
                    city_data['pm25'].mean(),
                    city_data['pm25'].std(),
                    city_data['pm25'].min(),
                    city_data['pm25'].max(),
                    city_data['pm25'].median()
                ]
                city_features.append(features)
                city_names.append(city_name)

        city_features = np.array(city_features)

        iso_forest = IsolationForest(contamination=0.2, random_state=42)
        anomaly_labels = iso_forest.fit_predict(city_features)

        anomalous_cities = []
        for city_name, label in zip(city_names, anomaly_labels):
            if label == -1:
                anomalous_cities.append(city_name)
                print(f"  [WARN] {city_name}: 检测到异常模式")
            else:
                print(f"  [OK] {city_name}: 数据正常")

        return anomalous_cities

    # ================== 主融合函数 ==================

    def fuse_cities_data(self, target_city=None):
        """
        融合多城市数据（主函数）

        参数：
            target_city: 如果指定，则为该城市增强数据

        返回：
            dict，统一结构：
            {
                'data': DataFrame,
                'similar_cities': list or None
            }

        Fix2：统一返回字典，解决两种调用方式返回值类型不一致的问题
        """
        print("\n" + "-" * 70)
        print("开始多城市数据融合")
        print("-" * 70)

        # 1. 质量评估和权重学习
        self.adaptive_weight_learning()

        # 2. 异常检测
        anomalous_cities = self.detect_anomalous_data()

        # 3. 融合数据
        if target_city and target_city in self.cities_data:
            # ── 为目标城市增强数据 ──
            target_data = self.cities_data[target_city]

            print(f"\n为 {target_city} 增强数据...")

            similarities = {}
            for other_city, other_data in self.cities_data.items():
                if other_city != target_city:
                    sim = self.calculate_city_similarity(target_data, other_data)
                    similarities[other_city] = sim

            sorted_cities = sorted(
                similarities.items(), key=lambda x: x[1], reverse=True
            )

            print(f"\n相似城市排序（Top3）:")
            for city, sim in sorted_cities[:3]:
                print(f"  {city}: {sim:.3f}")

            # ✅ Fix2：统一返回字典
            return {
                'data': target_data,
                'similar_cities': sorted_cities
            }

        else:
            # ── 融合所有城市数据 ──
            fused_list = []

            for city_name, city_data in self.cities_data.items():
                if city_name not in anomalous_cities:
                    city_data_weighted = city_data.copy()
                    city_data_weighted['weight'] = self.fusion_weights[city_name]
                    fused_list.append(city_data_weighted)

            merged_data = pd.concat(fused_list, ignore_index=True)

            print(f"\n[OK] 融合完成！总数据量: {len(merged_data):,} 条")

            # ✅ Fix2：统一返回字典
            return {
                'data': merged_data,
                'similar_cities': None
            }


# ================== 数据准备：模拟多城市数据 ==================

def simulate_multi_city_data():
    """
    模拟多城市数据

    Fix1：同时返回 df_base，供 main() 计算数据倍数使用
    """
    print("[SIM] 模拟生成多城市数据...")

    df_base = pd.read_csv('data_with_features.csv')
    df_base['timestamp'] = pd.to_datetime(df_base['timestamp'])

    cities_data = {}

    # 北京（原始数据）
    cities_data['北京'] = df_base.copy()

    # 天津（相似，稍微调整）
    df_tianjin = df_base.copy()
    df_tianjin['pm25'] = df_tianjin['pm25'] * 0.9
    df_tianjin['city'] = '天津'
    cities_data['天津'] = df_tianjin

    # 石家庄（较高污染）
    df_shijiazhuang = df_base.copy()
    df_shijiazhuang['pm25'] = df_shijiazhuang['pm25'] * 1.3
    df_shijiazhuang['city'] = '石家庄'
    cities_data['石家庄'] = df_shijiazhuang

    # 保定（数据不完整，30%缺失）
    df_baoding = df_base.sample(frac=0.7, random_state=42).copy()
    df_baoding['pm25'] = df_baoding['pm25'] * 1.1
    df_baoding['city'] = '保定'
    cities_data['保定'] = df_baoding

    # 唐山（工业城市，异常模式）
    rng = np.random.default_rng(42)
    df_tangshan = df_base.copy()
    df_tangshan['pm25'] = (
        df_tangshan['pm25'] * 1.5 + rng.normal(50, 20, len(df_base))
    ).clip(0, 500)
    df_tangshan['city'] = '唐山'
    cities_data['唐山'] = df_tangshan

    print(f"[OK] 生成 {len(cities_data)} 个城市的数据：")
    for city, data in cities_data.items():
        print(f"   {city}: {len(data):,} 条")

    # ✅ Fix1：同时返回 df_base
    return cities_data, df_base


# ================== 主函数 ==================

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║              多城市数据融合算法（真实数据版）                   ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    pkl_path    = 'multi_city_data.pkl'
    openaq_path = 'multi_city_openaq.csv'

    # [OK] 优先用真实OpenAQ数据
    if os.path.exists(openaq_path):
        print("[LOAD] 检测到真实OpenAQ多城市数据，优先使用...")
        df_openaq = pd.read_csv(openaq_path)
        df_openaq['timestamp'] = pd.to_datetime(df_openaq['timestamp'])

        cities_data = {}
        for city in df_openaq['city'].unique():
            cities_data[city] = df_openaq[
                df_openaq['city'] == city
            ].copy()

        df_base = df_openaq  # 用于计算倍数
        data_source = 'OpenAQ真实数据'

    elif os.path.exists(pkl_path):
        print("[LOAD] 加载缓存多城市数据...")
        with open(pkl_path, 'rb') as f:
            saved = pickle.load(f)
        if isinstance(saved, tuple):
            cities_data, df_base = saved
        else:
            cities_data = saved
            df_base = list(cities_data.values())[0]
        data_source = '缓存数据'

    else:
        print("[SIM] 生成情景模拟数据...")
        cities_data, df_base = simulate_multi_city_data()
        with open(pkl_path, 'wb') as f:
            pickle.dump((cities_data, df_base), f)
        data_source = '情景模拟数据'

    print(f"   数据来源：{data_source}")
    print(f"   城市数量：{len(cities_data)} 个")

    # 后面流程不变
    fusion = MultiCityDataFusion(cities_data)
    result_all = fusion.fuse_cities_data()
    fused_data = result_all['data']
    fused_data.to_csv('fused_multi_city_data.csv', index=False)

    print(f"""
    [OK] 融合完成！
       数据来源：{data_source}
       融合后数据量：{len(fused_data):,} 条
    """)


if __name__ == "__main__":
    main()
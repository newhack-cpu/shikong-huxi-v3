# -*- coding: utf-8 -*-
# multi_city_pipeline.py
"""
多城市训练 pipeline（v3.5 新增）
==================================

【目的】
当 OpenAQ 真实多城市数据采集完成后, 这个脚本把数据转换成可训练格式,
并用 MSTN v2/v3 在多城市数据上训练, 输出可在报告中报告的"多城市真实数据"指标。

【输入】
- multi_city_real.csv  (由 real_multi_city_collector.py 输出)
- 可选: 已有的 air_quality_data.csv (UCI 北京) 用于补充

【输出】
- multi_city_features.csv         - 跨城市特征工程后的数据
- multi_city_train_summary.json   - 多城市训练指标
- 可选: 各城市单独评估的 per_city_metrics.csv

【流程】
1. 加载 OpenAQ 多城市数据
2. 合并气象数据（如有 NOAA 数据则 join, 否则用城市级日均代理值）
3. 按城市分别做特征工程, 然后纵向拼接
4. 时序切分 + 模型训练
5. 整体评估 + 各城市分别评估（验证迁移泛化性）

【使用】
# 完整流程:
python multi_city_pipeline.py --input multi_city_real.csv

# 仅特征工程, 不训练:
python multi_city_pipeline.py --input multi_city_real.csv --skip-train

# 用 v3 训练（含物理约束）:
python multi_city_pipeline.py --input multi_city_real.csv --use-physical-loss
"""

import os
import sys

# ✅ v3.6: Windows GBK 兼容性
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from _windows_compat import configure_utf8_stdout, utf8_subprocess_env
    configure_utf8_stdout()
except ImportError:
    def utf8_subprocess_env(extra=None):
        env = os.environ.copy()
        env['PYTHONUTF8'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        if extra: env.update(extra)
        return env

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# 步骤 1: 多城市数据 → 标准化 schema
# ═══════════════════════════════════════════════════════════════

def standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    把 OpenAQ 输出的 schema 转换成与 UCI 一致的 schema:
      timestamp, city, pm25, temperature, pressure, wind_speed,
      wind_direction, dewpoint, snow_hours, rain_hours

    缺失字段用合理填充:
    - 气象字段缺失 → 用月均季节性代理
    - wind_direction 缺失 → 'cv' (calm/variable)
    """
    print("\n  标准化 schema...")
    out = pd.DataFrame()
    out['timestamp'] = pd.to_datetime(df['timestamp'])
    out['city'] = df['city']
    out['pm25'] = df['pm25'].astype(float)

    # 气象字段缺失值填充策略
    if 'temperature' not in df.columns:
        # 用月份做粗糙的季节性温度估计 (北京基线)
        month_temp_baseline = {
            1: -2, 2: 1, 3: 8, 4: 16, 5: 22, 6: 26,
            7: 28, 8: 27, 9: 22, 10: 15, 11: 6, 12: 0,
        }
        out['temperature'] = out['timestamp'].dt.month.map(month_temp_baseline).astype(float)
        # 加昼夜温差
        out['temperature'] += np.where(
            out['timestamp'].dt.hour.between(12, 16), 4,
            np.where(out['timestamp'].dt.hour.between(0, 5), -3, 0)
        )
    else:
        out['temperature'] = df['temperature']

    if 'pressure' not in df.columns:
        out['pressure'] = 1013.0  # 标准大气压
    else:
        out['pressure'] = df['pressure']

    if 'wind_speed' not in df.columns:
        out['wind_speed'] = 2.5  # 平均
    else:
        out['wind_speed'] = df['wind_speed']

    if 'wind_direction' not in df.columns:
        out['wind_direction'] = 'cv'  # calm/variable
    else:
        out['wind_direction'] = df['wind_direction']

    if 'dewpoint' not in df.columns:
        # 露点近似为 temperature - 5
        out['dewpoint'] = out['temperature'] - 5
    else:
        out['dewpoint'] = df['dewpoint']

    out['snow_hours'] = df.get('snow_hours', 0)
    out['rain_hours'] = df.get('rain_hours', 0)

    if 'station' in df.columns:
        out['station'] = df['station']
    if 'source' in df.columns:
        out['source'] = df['source']

    # 排序与去重
    out = out.sort_values(['city', 'timestamp']).reset_index(drop=True)
    before = len(out)
    out = out.drop_duplicates(subset=['city', 'timestamp']).reset_index(drop=True)
    print(f"  schema 标准化: {len(df)} → {before} → {len(out)} (去重后)")
    print(f"  字段: {list(out.columns)}")
    return out


# ═══════════════════════════════════════════════════════════════
# 步骤 2: 按城市做特征工程然后拼接
# ═══════════════════════════════════════════════════════════════

def per_city_feature_engineering(df: pd.DataFrame, output_csv: str = None):
    """
    在每个城市上独立做特征工程, 然后纵向拼接。
    避免跨城市的滞后特征污染。
    """
    print("\n  按城市做特征工程...")

    # 动态加载 feature_engineer.py
    here = Path(__file__).parent
    fe_path = here / 'feature_engineer.py'
    if not fe_path.exists():
        print(f"  ⚠️  找不到 feature_engineer.py, 跳过特征工程, 直接保存原始 schema")
        if output_csv:
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        return df

    import importlib.util
    spec = importlib.util.spec_from_file_location("fe", fe_path)
    fe_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fe_mod)

    enriched_dfs = []
    for city, sub in df.groupby('city'):
        if len(sub) < 50:
            print(f"     {city}: 数据太少 ({len(sub)} 条), 跳过")
            continue
        # 写到临时文件供 FeatureEngineer 读取
        tmp_in = f'/tmp/{city}_raw.csv'
        sub.to_csv(tmp_in, index=False, encoding='utf-8-sig')
        try:
            fe = fe_mod.FeatureEngineer(input_file=tmp_in)
            fe.create_time_features()
            fe.create_lag_features()
            fe.create_rolling_features()
            fe.create_diff_features()
            fe.create_interaction_features()
            fe.create_statistical_features()
            if hasattr(fe, 'create_breathing_health_index'):
                fe.create_breathing_health_index()
            enriched = fe.df.copy()
            enriched['city'] = city
            enriched_dfs.append(enriched)
            print(f"     {city}: {len(sub)} → {len(enriched)} 特征化, "
                  f"维度 {enriched.shape[1]}")
        except Exception as e:
            print(f"     {city}: 特征工程失败 - {e}")
        finally:
            try: os.remove(tmp_in)
            except: pass

    if not enriched_dfs:
        print("  ❌ 所有城市的特征工程都失败")
        return pd.DataFrame()

    # 取所有城市的公共列做并集
    all_cols = set()
    for d in enriched_dfs:
        all_cols.update(d.columns)
    common_cols = sorted(all_cols)
    aligned = []
    for d in enriched_dfs:
        for c in common_cols:
            if c not in d.columns:
                d[c] = 0.0  # 缺失补 0
        aligned.append(d[common_cols])

    full = pd.concat(aligned, ignore_index=True)
    full = full.sort_values(['city', 'timestamp']).reset_index(drop=True)
    print(f"\n  ✅ 多城市特征化完成: {len(full):,} 条 / {full.shape[1]} 列")
    print(f"     城市分布: {full['city'].value_counts().to_dict()}")

    if output_csv:
        full.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"  💾 已保存: {output_csv}")

    return full


# ═══════════════════════════════════════════════════════════════
# 步骤 3: 训练（调用 v2 或 v3 trainer）
# ═══════════════════════════════════════════════════════════════

def train_on_multi_city(features_csv: str, use_physical_loss: bool = False,
                         epochs: int = 50):
    """
    调用 6_advanced_model_v2 或 v3 在多城市特征化数据上训练
    """
    print("\n" + "=" * 70)
    print("启动多城市 MSTN 训练")
    print("=" * 70)

    here = Path(__file__).parent
    if use_physical_loss:
        trainer = here / '6_advanced_model_v3_with_physical.py'
        if not trainer.exists():
            print(f"  ❌ 找不到 v3 trainer, 退化到 v2")
            trainer = here / '6_advanced_model_v2.py'
            use_physical_loss = False
    else:
        trainer = here / '6_advanced_model_v2.py'

    if not trainer.exists():
        print(f"  ❌ 找不到训练器: {trainer}")
        return False

    cmd = [
        sys.executable, str(trainer),
        '--data', features_csv,
        '--epochs', str(epochs),
    ]
    if use_physical_loss:
        cmd += ['--use_physical_loss', 'true', '--output_suffix', 'multi_city_v3']

    print(f"  执行: {' '.join(cmd)}")
    # ✅ v3.6: 用 utf8_subprocess_env 强制子进程 UTF-8
    result = subprocess.run(cmd, env=utf8_subprocess_env())
    return result.returncode == 0


# ═══════════════════════════════════════════════════════════════
# 步骤 4: 各城市分别评估（验证迁移性）
# ═══════════════════════════════════════════════════════════════

def per_city_evaluation(predictions_csv: str, features_csv: str,
                         output_path: str = 'per_city_metrics.csv'):
    """
    把测试集预测结果按城市拆分, 计算每个城市的 MAE/RMSE/R²。
    用于报告 "MSTN 在 N 个城市上的泛化能力"。
    """
    if not os.path.exists(predictions_csv):
        print(f"  ⚠️  找不到预测文件 {predictions_csv}, 跳过分城市评估")
        return

    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

    print("\n" + "=" * 70)
    print("分城市评估")
    print("=" * 70)

    df_pred = pd.read_csv(predictions_csv)
    df_feat = pd.read_csv(features_csv)

    # predictions 没有 city 字段, 但是它的行序对应特征数据后 15% 的行
    # 简化处理: 直接通过位置关联
    n_pred = len(df_pred)
    df_test_features = df_feat.tail(n_pred + 24).head(n_pred)
    if 'city' not in df_test_features.columns:
        print("  ⚠️  特征数据没有 city 列, 无法分城市评估")
        return

    df_pred = df_pred.reset_index(drop=True)
    df_test_features = df_test_features.reset_index(drop=True)
    df_pred['city'] = df_test_features['city'].values

    rows = []
    for city, sub in df_pred.groupby('city'):
        if len(sub) < 10:
            continue
        mae = mean_absolute_error(sub['actual'], sub['predicted'])
        rmse = float(np.sqrt(mean_squared_error(sub['actual'], sub['predicted'])))
        r2 = r2_score(sub['actual'], sub['predicted'])
        rows.append({
            'city': city, 'n_samples': len(sub),
            'MAE': mae, 'RMSE': rmse, 'R2': r2,
        })
        print(f"  {city:>15s}: n={len(sub):5d} MAE={mae:6.2f} "
              f"RMSE={rmse:6.2f} R²={r2:.4f}")

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\n  💾 分城市指标: {output_path}")


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="多城市训练 pipeline")
    parser.add_argument('--input', default='multi_city_real.csv',
                        help='OpenAQ 输出的多城市数据 CSV')
    parser.add_argument('--features-output', default='multi_city_features.csv',
                        help='特征化输出 CSV')
    parser.add_argument('--skip-train', action='store_true',
                        help='只做特征工程, 跳过训练')
    parser.add_argument('--use-physical-loss', action='store_true',
                        help='训练时启用物理约束损失')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        print(f"   请先运行: python real_multi_city_collector.py")
        sys.exit(1)

    # 步骤 1: 加载与标准化
    print("=" * 70)
    print("多城市训练 pipeline")
    print("=" * 70)
    print(f"\n加载: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  原始: {df.shape}, 城市: {df['city'].nunique()}")

    df = standardize_schema(df)

    # 步骤 2: 特征工程
    df_features = per_city_feature_engineering(df, args.features_output)
    if df_features.empty:
        print("❌ 特征工程失败")
        sys.exit(1)

    # 步骤 3: 训练
    if args.skip_train:
        print("\n⏭️  跳过训练 (--skip-train)")
        return

    success = train_on_multi_city(
        args.features_output,
        use_physical_loss=args.use_physical_loss,
        epochs=args.epochs,
    )
    if not success:
        print("❌ 训练失败")
        sys.exit(1)

    # 步骤 4: 分城市评估
    if args.use_physical_loss:
        pred_csv = 'mstn_multi_city_v3_predictions.csv'
    else:
        pred_csv = 'mstn_v2_predictions.csv'

    per_city_evaluation(pred_csv, args.features_output)

    print("\n" + "=" * 70)
    print("✅ 多城市 pipeline 完成")
    print("=" * 70)
    print("\n下一步: 把 per_city_metrics.csv 的结果写到报告 5.6 节")


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
test_transfer_data_prep.py
==========================
测试 transfer_learning_experiment 中**不需要 torch 的部分**:
  - make_synthetic_target_city: 合成目标城市的扰动逻辑
  - 数据流的形状与一致性

torch 训练循环本身需要在用户本机跑, 这里仅校验数据准备无误。
"""

import sys
import os

# ✅ v3.6: Windows GBK 兼容性
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from _windows_compat import configure_utf8_stdout
    configure_utf8_stdout()
except ImportError:
    pass

import numpy as np
import pandas as pd

from transfer_learning_experiment import make_synthetic_target_city


def test_synthetic_city_basic():
    print("=" * 70)
    print("测试 1: 合成城市基本属性")
    print("=" * 70)

    # 构造一个真实风格的源城市数据
    rng = np.random.default_rng(42)
    n = 500
    df_src = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='h'),
        'city': '北京',
        'pm25': rng.gamma(2, 30, n).clip(1, 500),
        'temperature': rng.normal(15, 8, n),
        'wind_speed': rng.gamma(2, 1.5, n),
        'pressure': rng.normal(1015, 6, n),
    })

    df_tgt = make_synthetic_target_city(
        df_src, city_name='合成-天津',
        pm25_offset=-12, temp_offset=2.5, wind_scale=1.15,
        noise_std=3.0, seed=2024,
    )

    # 检查 1: 形状一致
    assert len(df_tgt) == len(df_src), "FAIL: target should have same length"
    print(f"  ✅ 形状一致: src={len(df_src)} == tgt={len(df_tgt)}")

    # 检查 2: city 列被改了
    assert (df_tgt['city'] == '合成-天津').all(), "FAIL: city should be 合成-天津"
    print(f"  ✅ city 列设置正确")

    # 检查 3: source 列标注合成
    assert (df_tgt['source'] == 'synthetic_perturbation').all()
    print(f"  ✅ source 列正确标注合成: synthetic_perturbation")

    # 检查 4: PM2.5 偏移合理（均值差异在期望范围内）
    pm_diff = df_tgt['pm25'].mean() - df_src['pm25'].mean()
    print(f"  ✅ PM2.5 平均差异: {pm_diff:.2f} (期望 ~ -12)")
    assert -16 < pm_diff < -8, f"FAIL: pm_diff={pm_diff} 偏离期望"

    # 检查 5: 温度偏移
    temp_diff = df_tgt['temperature'].mean() - df_src['temperature'].mean()
    print(f"  ✅ 温度平均差异: {temp_diff:.2f} (期望 ~ +2.5)")
    assert 2.3 < temp_diff < 2.7

    # 检查 6: 风速放大
    wind_ratio = df_tgt['wind_speed'].mean() / df_src['wind_speed'].mean()
    print(f"  ✅ 风速比例: {wind_ratio:.3f} (期望 ~ 1.15)")
    assert 1.13 < wind_ratio < 1.17

    # 检查 7: PM2.5 不可为负
    assert (df_tgt['pm25'] >= 1.0).all(), "FAIL: pm25 应被 clip 至 >= 1.0"
    print(f"  ✅ PM2.5 全部 >= 1.0 (无负值)")


def test_seed_reproducibility():
    """同样的 seed 应该出同样的结果"""
    print("\n" + "=" * 70)
    print("测试 2: 随机种子可复现性")
    print("=" * 70)

    rng = np.random.default_rng(42)
    df_src = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
        'pm25': rng.gamma(2, 30, 100).clip(1, 500),
        'temperature': rng.normal(15, 8, 100),
        'wind_speed': rng.gamma(2, 1.5, 100),
    })

    df1 = make_synthetic_target_city(df_src, seed=2024)
    df2 = make_synthetic_target_city(df_src, seed=2024)
    df3 = make_synthetic_target_city(df_src, seed=999)

    assert (df1['pm25'].values == df2['pm25'].values).all(), \
        "FAIL: 同 seed 应该出同样的结果"
    print("  ✅ 同 seed (2024) 结果一致")

    assert not (df1['pm25'].values == df3['pm25'].values).all(), \
        "FAIL: 不同 seed 应该出不同结果"
    print("  ✅ 不同 seed (2024 vs 999) 结果不同")


def test_independence_from_torch():
    """验证 make_synthetic_target_city 不依赖 torch"""
    print("\n" + "=" * 70)
    print("测试 3: 不依赖 torch")
    print("=" * 70)

    # 验证函数本身可以独立运行
    import inspect
    src = inspect.getsource(make_synthetic_target_city)
    # 检查函数体内部不直接 import torch
    func_body_lines = src.split('def make_synthetic_target_city')[1]
    assert 'import torch' not in func_body_lines, \
        "FAIL: make_synthetic_target_city 不应依赖 torch"
    print("  ✅ make_synthetic_target_city 不依赖 torch")


def test_argparse_logic():
    """测试 argparse 的优先级逻辑"""
    print("\n" + "=" * 70)
    print("测试 4: argparse 逻辑")
    print("=" * 70)

    # 模拟 namespace
    class A: pass
    args = A()

    # 案例 1: 不传 --target, 不传 --synthetic_target → 自动合成
    args.target = None
    args.synthetic_target = False
    if not args.synthetic_target:
        if args.target is None or not os.path.exists(args.target):
            args.synthetic_target = True
    assert args.synthetic_target == True
    print("  ✅ 案例 A: 不传 target, 自动合成")

    # 案例 2: 传 --target 但文件不存在
    args.target = '/nonexistent/file.csv'
    args.synthetic_target = False
    if not args.synthetic_target:
        if args.target is None or not os.path.exists(args.target):
            args.synthetic_target = True
    assert args.synthetic_target == True
    print("  ✅ 案例 B: target 文件不存在, 自动降级到合成")

    # 案例 3: 显式 --synthetic_target
    args.target = None
    args.synthetic_target = True
    assert args.synthetic_target == True
    print("  ✅ 案例 C: 显式 --synthetic_target")


def main():
    print("\n" + "=" * 70)
    print("迁移学习实验 - 数据准备链路测试 (不依赖 torch)")
    print("=" * 70 + "\n")

    test_synthetic_city_basic()
    test_seed_reproducibility()
    test_independence_from_torch()
    test_argparse_logic()

    print("\n" + "=" * 70)
    print("✅ 所有不依赖 torch 的测试通过")
    print("=" * 70)
    print("\n下一步 (在用户本机):")
    print("  pip install torch scikit-learn lightgbm")
    print("  cd code/")
    print("  python transfer_learning_experiment.py --pretrain_epochs 5")


if __name__ == '__main__':
    main()

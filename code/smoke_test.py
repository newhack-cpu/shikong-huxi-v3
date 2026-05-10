# -*- coding: utf-8 -*-
# smoke_test.py
"""
冒烟测试（Smoke Test）
======================

在不跑完整训练的情况下，10 秒内验证整个 pipeline 的关键模块都能 import + 实例化。

【运行】
python smoke_test.py

【输出】
✅ 全部通过 / ❌ 哪些模块失败 + 具体错误
"""

import sys
import importlib
import traceback


def test_imports():
    """测试关键依赖能 import"""
    print("\n[1/4] 检查 Python 依赖...")
    required = [
        'numpy', 'pandas', 'sklearn', 'matplotlib',
        'xgboost', 'lightgbm', 'statsmodels', 'requests',
        'joblib', 'plotly',
    ]
    optional = [
        'torch',          # 深度学习
        'streamlit',      # Web 应用
        'seaborn',
    ]

    missing_required = []
    missing_optional = []

    for pkg in required:
        try:
            mod = importlib.import_module(pkg)
            v = getattr(mod, '__version__', '?')
            print(f"   ✅ {pkg:<15} {v}")
        except ImportError:
            print(f"   ❌ {pkg:<15} 未安装")
            missing_required.append(pkg)

    for pkg in optional:
        try:
            mod = importlib.import_module(pkg)
            v = getattr(mod, '__version__', '?')
            print(f"   ✅ {pkg:<15} {v} (可选)")
        except ImportError:
            print(f"   ⚠️  {pkg:<15} 未安装 (可选)")
            missing_optional.append(pkg)

    return missing_required, missing_optional


def test_bhi_v2():
    """测试 BHI v2 公式正确性"""
    print("\n[2/4] BHI v2 单元测试...")
    try:
        sys.path.insert(0, '.')
        from feature_engineer_bhi_v2 import compute_bhi
        import numpy as np

        # 测试用例 1: 春日清晨
        history = np.array([10.0] * 24)
        r = compute_bhi(10, 22, 50, 1.0, history)
        assert r['level'] == '优质', f"期望优质，得到 {r['level']}"
        assert r['bhi'] < 20, f"BHI 应 < 20，得到 {r['bhi']}"

        # 测试用例 2: 严重污染
        history = np.array([200.0] * 24)
        r = compute_bhi(300, 32, 80, 0.5, history)
        assert r['level'] == '危险', f"期望危险，得到 {r['level']}"

        # 测试敏感人群
        r1 = compute_bhi(75, 5, 40, 4.0, np.array([60.0]*24), sensitive_group=False)
        r2 = compute_bhi(75, 5, 40, 4.0, np.array([60.0]*24), sensitive_group=True)
        assert r2['bhi'] > r1['bhi'], "敏感人群 BHI 应高于一般人群"

        print(f"   ✅ BHI v2 通过 3 个测试用例")
        return True
    except Exception as e:
        print(f"   ❌ BHI v2 失败: {e}")
        traceback.print_exc()
        return False


def test_mstn_v2():
    """测试 MSTN v2 模型能实例化 + 前向"""
    print("\n[3/4] MSTN v2 模型测试...")
    try:
        import torch
    except ImportError:
        print("   ⚠️  PyTorch 未安装，跳过 MSTN v2 测试（不影响判定）")
        print("       安装命令: pip install torch")
        return True   # 不算失败，仅警告

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("adv", "6_advanced_model_v2.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        model = mod.MSTNv2(input_dim=65, hidden_dim=64)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 1_000_000, f"参数量 {n_params} 超过 100 万限制"
        print(f"   ✅ MSTN v2 实例化成功（{n_params:,} 参数 ≈ {n_params/1e3:.1f}K）")

        x = torch.randn(4, 24, 65)
        pred, scale_w, feat_attn = model(x)
        assert pred.shape == (4, 3), f"预期 [4,3]，得到 {pred.shape}"
        assert (pred[:, 0] <= pred[:, 1]).all(), "q05 应 ≤ q50"
        assert (pred[:, 1] <= pred[:, 2]).all(), "q50 应 ≤ q95"
        print(f"   ✅ 前向传播正常，输出形状 {tuple(pred.shape)}")
        print(f"   ✅ 分位数单调性：q05 ≤ q50 ≤ q95")

        loss = mod.quantile_loss(pred, torch.randn(4))
        print(f"   ✅ Quantile Loss 计算正常: {loss.item():.4f}")
        return True

    except Exception as e:
        print(f"   ❌ MSTN v2 失败: {e}")
        traceback.print_exc()
        return False


def test_data_files():
    """检查关键数据文件"""
    print("\n[4/4] 关键文件检查...")
    import os

    required_code = [
        'feature_engineer.py',
        'feature_engineer_bhi_v2.py',
        '6_advanced_model_v2.py',
        '8_ablation_study_v2.py',
        '9_comparison_baselines.py',
        'real_multi_city_collector.py',
        'paper_figures_generator.py',
        'data_collector.py',
        'model_trainer.py',
        'app.py',
        'run.py',
    ]

    missing_code = []
    for f in required_code:
        if os.path.exists(f):
            print(f"   ✅ {f}")
        else:
            print(f"   ❌ {f}  缺失")
            missing_code.append(f)

    optional_data = [
        'air_quality_data.csv',
        'data_with_features.csv',
        'multi_city_real.csv',
        'models/LightGBM_model.pkl',
        'models/mstn_v2_best.pth',
    ]

    print("\n   可选数据/模型文件（首次运行会自动生成）：")
    for f in optional_data:
        if os.path.exists(f):
            sz = os.path.getsize(f) / 1024
            print(f"   ✅ {f}  ({sz:.1f} KB)")
        else:
            print(f"   ○  {f}  尚未生成")

    return missing_code


def main():
    print("═" * 60)
    print("时空呼吸 · Smoke Test (10 秒快速验证)")
    print("═" * 60)

    missing_req, missing_opt = test_imports()
    bhi_ok  = test_bhi_v2()
    mstn_ok = test_mstn_v2()
    missing_code = test_data_files()

    print("\n" + "═" * 60)
    print("总结")
    print("═" * 60)

    issues = []
    if missing_req:
        issues.append(f"❌ 缺少必需依赖: {', '.join(missing_req)}")
        issues.append(f"   → 运行: pip install {' '.join(missing_req)}")
    if missing_code:
        issues.append(f"❌ 缺少代码文件: {', '.join(missing_code)}")
    if not bhi_ok:
        issues.append("❌ BHI v2 测试失败")
    if not mstn_ok:
        issues.append("❌ MSTN v2 测试失败")

    if not issues:
        print("✅ 全部通过！可以直接运行 python run.py 启动完整流程")
    else:
        print("⚠️  发现以下问题：")
        for i in issues:
            print(f"   {i}")
        print("\n请修复后重试。")

    if missing_opt:
        print(f"\n💡 可选依赖未安装（不影响核心流程）: {', '.join(missing_opt)}")

    return 0 if not issues else 1


if __name__ == '__main__':
    sys.exit(main())

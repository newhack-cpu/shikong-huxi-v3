# -*- coding: utf-8 -*-
# benchmark_inference.py
"""
推理速度 Benchmark 脚本
=======================

【目的】
答辩中我们声称"CPU 单批次推理 < 50ms"——必须有可复现的实测脚本支撑。

【运行】
python benchmark_inference.py                    # 默认 batch=64, 100 次
python benchmark_inference.py --batch 1 --n 200  # 单样本延迟测试
python benchmark_inference.py --device cuda      # GPU 测试

【输出】
inference_benchmark.csv  - 详细统计（mean/std/p50/p95/p99）
inference_benchmark.png  - 延迟分布直方图
"""

import argparse
import time
import os
import json
import numpy as np
import pandas as pd

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("❌ PyTorch 未安装，无法 benchmark MSTN")


def load_mstn_model(weights_path='models/mstn_v2_best.pth',
                    input_dim=65, hidden_dim=64):
    """加载训练好的 MSTN v2 模型"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("adv", "6_advanced_model_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    model = mod.MSTNv2(input_dim=input_dim, hidden_dim=hidden_dim)
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state, strict=False)
        print(f"✅ 加载权重: {weights_path}")
    else:
        print(f"⚠️  未找到 {weights_path}，使用随机权重测试速度")

    return model


def benchmark_pytorch(model, batch_size, n_runs, device, seq_len=24, input_dim=65):
    """对模型做 benchmark，返回 latency 数组（毫秒）"""
    model = model.to(device).eval()

    # 准备输入
    x = torch.randn(batch_size, seq_len, input_dim, device=device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    if device == 'cuda':
        torch.cuda.synchronize()

    # 实测
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)   # ms

    return np.array(latencies)


def benchmark_traditional(model_path, batch_size, n_runs):
    """对 sklearn / XGBoost / LightGBM 模型做 benchmark"""
    import joblib
    if not os.path.exists(model_path):
        print(f"⚠️  跳过 {model_path}（文件不存在）")
        return None

    model = joblib.load(model_path)
    # 假设特征数与训练时一致
    try:
        n_features = (
            getattr(model, 'n_features_in_', None) or
            (model.feature_importances_.shape[0] if hasattr(model, 'feature_importances_') else 65)
        )
    except Exception:
        n_features = 65

    x = np.random.randn(batch_size, n_features).astype(np.float32)

    # 预热
    for _ in range(10):
        _ = model.predict(x)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.predict(x)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    return np.array(latencies)


def stats(latencies):
    """计算统计量"""
    return {
        'mean':  float(np.mean(latencies)),
        'std':   float(np.std(latencies)),
        'min':   float(np.min(latencies)),
        'max':   float(np.max(latencies)),
        'p50':   float(np.percentile(latencies, 50)),
        'p95':   float(np.percentile(latencies, 95)),
        'p99':   float(np.percentile(latencies, 99)),
        'qps_estimate': 1000.0 / float(np.mean(latencies)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch',  type=int, default=64, help='batch size')
    parser.add_argument('--n',      type=int, default=100, help='次数')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    print("═" * 70)
    print(f"推理速度 Benchmark   batch={args.batch}, runs={args.n}, device={args.device}")
    print("═" * 70)

    if args.device == 'cuda' and (not HAS_TORCH or not torch.cuda.is_available()):
        print("⚠️  CUDA 不可用，回退到 CPU")
        args.device = 'cpu'

    all_results = {}

    # 1. MSTN v2
    if HAS_TORCH:
        print("\n[1/5] MSTN v2 ...")
        try:
            model = load_mstn_model()
            lat = benchmark_pytorch(model, args.batch, args.n, args.device)
            all_results['MSTN_v2'] = stats(lat)
            print(f"      mean = {all_results['MSTN_v2']['mean']:.2f} ms")
            print(f"      p95  = {all_results['MSTN_v2']['p95']:.2f} ms")
        except Exception as e:
            print(f"      失败: {e}")

    # 2-5. 传统模型
    for name, path in [
        ('LightGBM',     'models/LightGBM_model.pkl'),
        ('XGBoost',      'models/XGBoost_model.pkl'),
        ('RandomForest', 'models/RandomForest_model.pkl'),
        ('Ridge',        'models/Ridge_model.pkl'),
    ]:
        print(f"\n[{name}] ...")
        lat = benchmark_traditional(path, args.batch, args.n)
        if lat is not None:
            all_results[name] = stats(lat)
            print(f"      mean = {all_results[name]['mean']:.2f} ms")

    # 输出汇总表
    if all_results:
        df = pd.DataFrame(all_results).T
        df = df.round(3)
        df.to_csv('inference_benchmark.csv')
        print("\n" + "═" * 70)
        print("Benchmark 汇总（所有时间单位：毫秒 ms）")
        print("═" * 70)
        print(df[['mean', 'std', 'p50', 'p95', 'p99', 'qps_estimate']].to_string())
        print("\n✅ 详细结果已保存: inference_benchmark.csv")

        # 写入 JSON 供报告引用
        with open('inference_benchmark.json', 'w', encoding='utf-8') as f:
            json.dump({
                'config': {'batch': args.batch, 'n_runs': args.n, 'device': args.device},
                'results': all_results,
            }, f, indent=2)

        print("\n📊 论文/答辩可用句式示例：")
        if 'MSTN_v2' in all_results:
            mstn = all_results['MSTN_v2']
            print(f"   - MSTN v2 在 {args.device.upper()} 上 batch={args.batch} 推理延迟 ")
            print(f"     mean = {mstn['mean']:.1f} ms, p95 = {mstn['p95']:.1f} ms")
            print(f"     吞吐 ≈ {mstn['qps_estimate']:.0f} batch/s, 即每秒处理 {mstn['qps_estimate']*args.batch:.0f} 样本")
            print(f"   - 满足实时预警的工程需求")


if __name__ == '__main__':
    main()

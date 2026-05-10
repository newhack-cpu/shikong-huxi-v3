# -*- coding: utf-8 -*-
# verify_optimization_modules.py
"""
一键自检所有 v3.5 新增模块
============================

运行此脚本会依次:
  1. 检查 Python 依赖是否就绪
  2. 跑 OpenAQ 解析逻辑 mock 测试 (不需网络/torch)
  3. 跑迁移学习数据准备测试 (不需 torch)
  4. 跑物理约束损失逻辑测试 (numpy 验证, 不需 torch)
  5. 跑 BHI 权重溯源完整流程
  6. 跑 OpenAQ 离线模式 (验证完整数据采集链路)
  7. 如果安装了 torch, 额外跑 torch 单元测试

使用:
  python verify_optimization_modules.py
  python verify_optimization_modules.py --skip-torch   # 跳过 torch 测试

成功标志: 所有项打 ✅, 末尾输出"全部模块自检通过"
"""

import os
import sys

# ✅ v3.6: 使用统一的 Windows 兼容工具
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from _windows_compat import configure_utf8_stdout, utf8_subprocess_env
    configure_utf8_stdout()
except ImportError:
    # _windows_compat.py 缺失时的简易兜底
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    def utf8_subprocess_env(extra=None):
        env = os.environ.copy()
        env['PYTHONUTF8'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        if extra: env.update(extra)
        return env

import subprocess
import importlib
from pathlib import Path

CODE_DIR = Path(__file__).parent.absolute()
os.chdir(CODE_DIR)


def header(title):
    print("\n" + "═" * 72)
    print(f"  {title}")
    print("═" * 72)


def run_subprocess(cmd: list, label: str) -> bool:
    """
    运行子进程, 返回是否成功。
    
    ✅ v3.6: 使用 utf8_subprocess_env() 强制子进程 UTF-8 模式,
       避免 Windows GBK 默认环境下的 UnicodeDecodeError。
    """
    print(f"\n▶ 执行: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            encoding='utf-8', errors='replace',
            env=utf8_subprocess_env(),
        )
        if result.returncode == 0:
            print(f"  ✅ {label} - 通过")
            return True
        else:
            print(f"  ❌ {label} - 失败 (returncode={result.returncode})")
            print(f"     stderr: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ⏱️  {label} - 超时 (>120s)")
        return False
    except Exception as e:
        print(f"  ❌ {label} - 异常: {e}")
        return False


def check_python_deps():
    """检查 Python 核心依赖"""
    header("步骤 1: Python 依赖检查")
    required = {
        'numpy': '✅ 必需 (基础)',
        'pandas': '✅ 必需 (数据处理)',
        'sklearn': '✅ 必需 (基线模型/标准化)',
        'matplotlib': '✅ 必需 (可视化)',
    }
    optional = {
        'torch': '🟡 训练 MSTN/迁移学习时需要',
        'lightgbm': '🟡 基线对比时需要',
        'requests': '🟡 OpenAQ 真实采集时需要',
        'streamlit': '🟡 Web 应用时需要',
    }
    print("\n核心依赖:")
    all_required_ok = True
    for pkg, desc in required.items():
        try:
            importlib.import_module(pkg)
            print(f"  ✅ {pkg:12s} {desc}")
        except ImportError:
            print(f"  ❌ {pkg:12s} 未安装! 必须 pip install {pkg}")
            all_required_ok = False

    print("\n可选依赖:")
    for pkg, desc in optional.items():
        try:
            importlib.import_module(pkg)
            print(f"  ✅ {pkg:12s} {desc}")
        except ImportError:
            print(f"  ⚠️  {pkg:12s} {desc} - 未安装")

    return all_required_ok


def check_files_exist():
    """检查所有新增模块文件存在"""
    header("步骤 2: 文件存在性检查")
    required_files = [
        'real_multi_city_collector.py',     # OpenAQ 采集器
        'physical_constraint_loss.py',       # 物理约束损失
        'bhi_weight_regression.py',          # BHI 权重溯源
        'transfer_learning_experiment.py',   # 跨城市迁移
        '6_advanced_model_v3_with_physical.py',  # MSTN v3 (含物理约束)
        'multi_city_pipeline.py',            # 多城市训练 pipeline
        'mstn_hyperparameter_search.py',     # 超参数搜索 (策略 A 工具)
        'test_openaq_parsing.py',            # OpenAQ mock 测试
        'test_transfer_data_prep.py',        # 迁移数据测试
        'test_physical_loss_logic.py',       # 物理损失逻辑测试
        '_windows_compat.py',                # Windows GBK 兼容性工具
    ]
    all_ok = True
    for f in required_files:
        if (CODE_DIR / f).exists():
            print(f"  ✅ {f}")
        else:
            print(f"  ❌ {f} - 缺失!")
            all_ok = False
    return all_ok


def main():
    print("=" * 72)
    print("  时空呼吸 v3.5 - 优化模块自检脚本")
    print("=" * 72)
    print(f"\n工作目录: {CODE_DIR}")

    skip_torch = '--skip-torch' in sys.argv

    results = {}
    results['依赖检查'] = check_python_deps()
    results['文件检查'] = check_files_exist()

    if not all([results['依赖检查'], results['文件检查']]):
        print("\n❌ 前置检查失败, 终止")
        sys.exit(1)

    # ── 跑各个 mock/逻辑测试 ──────────────────────────────
    header("步骤 3: OpenAQ 解析逻辑 (mock)")
    results['OpenAQ 解析'] = run_subprocess(
        [sys.executable, 'test_openaq_parsing.py'],
        'OpenAQ 解析逻辑测试',
    )

    header("步骤 4: 迁移学习数据准备")
    results['迁移数据准备'] = run_subprocess(
        [sys.executable, 'test_transfer_data_prep.py'],
        '迁移数据准备测试',
    )

    header("步骤 5: 物理约束损失逻辑")
    results['物理约束逻辑'] = run_subprocess(
        [sys.executable, 'test_physical_loss_logic.py'],
        '物理约束损失逻辑测试',
    )

    header("步骤 6: BHI 权重溯源完整流程")
    results['BHI 溯源'] = run_subprocess(
        [sys.executable, 'bhi_weight_regression.py'],
        'BHI 权重溯源',
    )

    header("步骤 7: OpenAQ 采集器 (离线模式)")
    results['OpenAQ 离线'] = run_subprocess(
        [sys.executable, 'real_multi_city_collector.py',
         '--offline', '--output', 'test_offline_master.csv',
         '--log', 'test_offline_master.json'],
        'OpenAQ 离线采集',
    )

    # ── torch 相关测试（可选）─────────────────────────────
    if not skip_torch:
        try:
            import torch  # noqa
            header("步骤 8: 物理约束损失 (torch 单元测试)")
            results['物理约束 torch'] = run_subprocess(
                [sys.executable, 'physical_constraint_loss.py'],
                'physical_constraint_loss torch 测试',
            )
        except ImportError:
            print("\n  ⚠️  torch 未安装, 跳过 torch 单元测试")
            print("     生产环境需要: pip install torch")

    # ── 汇总 ──────────────────────────────────────────────
    header("自检结果汇总")
    all_passed = True
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {test_name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("=" * 72)
        print("  ✅ 全部模块自检通过 - 代码可以提交")
        print("=" * 72)
        print("\n后续动作:")
        print("  1. 申请 OpenAQ API key: https://explore.openaq.org/register")
        print("  2. export OPENAQ_API_KEY='your_key'")
        print("  3. python real_multi_city_collector.py --dry-run --api-key $OPENAQ_API_KEY")
        print("  4. python real_multi_city_collector.py --days 90")
        sys.exit(0)
    else:
        print("=" * 72)
        print("  ❌ 部分模块自检失败 - 请查看上方日志定位问题")
        print("=" * 72)
        sys.exit(1)


if __name__ == '__main__':
    main()

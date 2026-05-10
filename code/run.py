# run.py — 时空呼吸 · 一键运行管道（v2 升级版）
"""
🌬️ 时空呼吸 · 一键运行管道（v2）

执行顺序（v2 升级，集成全部新模块）：
  Step 0: 多源大数据采集
  Step 1: UCI 数据下载与清洗
  Step 2: 特征工程（含 BHI v2，65+ 维特征）
  Step 3: 基础模型训练（Ridge / RF / XGBoost / LightGBM 等）
  Step 4: 可视化生成（8 张交互图表）
  Step 5: MSTN v2 高级模型训练（推荐）★
  Step 6: 多城市数据融合 / 真实数据采集
  Step 7: 真·消融实验（v2，3 次重复，mean ± std）★
  Step 8: 基线对比实验（v2 自动加载 MSTN v2 结果）
  Step 9: 论文级图表生成（7 张 PNG + PDF）★

★ = v2 升级新增/重构的步骤

运行方式：
  python run.py              # 完整流程
  python run.py --skip-dl    # 跳过网络下载（离线模式）
  python run.py --from 3     # 从 Step 3 开始
  python run.py --only 7     # 只跑消融实验
  python run.py --quick      # 快速模式（消融用 --quick）
"""

import subprocess
import sys
import os
import time
import argparse
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

# ── 颜色辅助 ──────────────────────────────────────────────
GREEN  = '\033[92m'
YELLOW = '\033[93m'
RED    = '\033[91m'
CYAN   = '\033[96m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

def cprint(msg, color=CYAN):
    print(f"{color}{msg}{RESET}")


# ── 步骤定义（v2 升级） ──────────────────────────────────
STEPS = [
    {
        'id':    0,
        'name':  '多源大数据采集',
        'file':  '0_collect_multi_source_data.py',
        'desc':  'OpenAQ + NOAA + UCI 三源融合，构建大数据集',
        'online': True,
        'optional': True,
    },
    {
        'id':    1,
        'name':  'UCI 数据下载与清洗',
        'file':  'data_collector.py',
        'desc':  'UCI 北京 PM2.5 数据下载、清洗、缺失值处理',
        'online': True,
        'optional': False,
    },
    {
        'id':    2,
        'name':  '特征工程 (含 BHI v2)',
        'file':  'feature_engineer.py',
        'desc':  '65 维特征工程 + BHI v2（GB 3095 + WHO + ASHRAE 标准）',
        'online': False,
        'optional': False,
    },
    {
        'id':    3,
        'name':  '基础模型训练',
        'file':  'model_trainer.py',
        'desc':  'Ridge / RF / GBoost / XGBoost / LightGBM 五模型对比',
        'online': False,
        'optional': False,
    },
    {
        'id':    4,
        'name':  '可视化生成',
        'file':  'visualizer.py',
        'desc':  '8 张交互式 HTML 图表',
        'online': False,
        'optional': True,
    },
    {
        'id':    5,
        'name':  'MSTN v2 高级模型 ★',
        'file':  '6_advanced_model_v2.py',
        'desc':  '三尺度因果膨胀 TCN + 跨尺度注意力 + 分位数预测',
        'online': False,
        'optional': True,
    },
    {
        'id':    6,
        'name':  '多城市数据采集',
        'file':  'real_multi_city_collector.py',
        'desc':  'OpenAQ 真实多城市数据（失败回退到合成扩展）',
        'online': True,
        'optional': True,
    },
    {
        'id':    7,
        'name':  '真·消融实验 ★',
        'file':  '8_ablation_study_v2.py',
        'desc':  '7 个变体 × 3 次重复，输出 LaTeX 三线表',
        'online': False,
        'optional': True,
        'extra_args_quick': ['--quick'],         # 在 --quick 模式时附加
    },
    {
        'id':    8,
        'name':  '基线对比实验',
        'file':  '9_comparison_baselines.py',
        'desc':  'ARIMA / SVR / XGBoost / LightGBM / MSTN v2 横向对比',
        'online': False,
        'optional': True,
    },
    {
        'id':    9,
        'name':  '多步预测对比 ★',
        'file':  'multistep_forecasting.py',
        'desc':  '验证 MSTN v2 在 1h/6h/12h/24h/48h/72h 上的优势',
        'online': False,
        'optional': True,
        'extra_args_quick': ['--quick'],
    },
    {
        'id':    10,
        'name':  '论文级图表生成 ★',
        'file':  'paper_figures_generator.py',
        'desc':  '7 张论文级 PNG + PDF（IEEE 双栏可发表风格）',
        'online': False,
        'optional': True,
    },
]


def run_step(step: dict, python_exe: str = 'python', quick_mode: bool = False) -> bool:
    """运行单个步骤"""
    file = step['file']
    name = step['name']

    if not os.path.exists(file):
        cprint(f"   ⚠️  找不到 {file}，跳过", YELLOW)
        return False

    cprint(f"\n{'═' * 60}", CYAN)
    cprint(f"  ▶  Step {step['id']:d} | {name}", BOLD)
    cprint(f"     {step['desc']}", CYAN)
    cprint(f"{'═' * 60}", CYAN)

    # 构造命令
    cmd = [python_exe, file]
    if quick_mode and 'extra_args_quick' in step:
        cmd += step['extra_args_quick']
        cprint(f"     [quick] 附加参数: {' '.join(step['extra_args_quick'])}", YELLOW)

    start = time.time()
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUNBUFFERED'] = '1'
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)
    elapsed = time.time() - start

    if result.returncode == 0:
        cprint(f"\n  ✅  {name} 完成（耗时 {elapsed:.1f}s）", GREEN)
        return True
    else:
        cprint(f"\n  ❌  {name} 失败（耗时 {elapsed:.1f}s）", RED)
        if step.get('optional'):
            cprint(f"     （可选步骤，继续执行）", YELLOW)
        return False


def print_banner():
    cprint("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║          🌬️   时空呼吸  · Temporal-Spatial Breathing        ║
║                                                              ║
║     大数据驱动的城市空气质量智能预测与呼吸健康评估系统       ║
║                                                              ║
║     赛道：大数据实践赛 · 环境与人类发展大数据                ║
║     版本：v2.0 (国奖冲刺版)                                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """, CYAN)


def print_summary(results: dict, total_elapsed: float):
    cprint("\n" + "═" * 60, CYAN)
    cprint("  📊  运行汇总", BOLD)
    cprint("═" * 60, CYAN)

    ok    = sum(1 for v in results.values() if v)
    total = len(results)

    for step_name, success in results.items():
        icon = '✅' if success else '❌'
        print(f"  {icon}  {step_name}")

    cprint(f"\n  完成 {ok}/{total} 步  |  总耗时 {total_elapsed:.1f}s", GREEN if ok == total else YELLOW)

    if results.get('基础模型训练', False):
        cprint("""
  🚀  下一步操作：
       streamlit run app.py        # 启动 Web 应用
       open paper_figures/         # 查看论文级图表
        """, GREEN)


def main():
    parser = argparse.ArgumentParser(description='时空呼吸 · 一键运行管道 v2')
    parser.add_argument('--skip-dl',  action='store_true', help='跳过需要网络的步骤')
    parser.add_argument('--from',     dest='from_step', type=int, default=0,
                        help='从指定步骤开始（0-9）')
    parser.add_argument('--only',     dest='only_step', type=int, default=None,
                        help='只运行指定步骤（0-9）')
    parser.add_argument('--quick',    action='store_true',
                        help='快速模式（消融实验只跑 1 次种子，10 epoch）')
    parser.add_argument('--python',   dest='python_exe', default='python',
                        help='Python 解释器路径')
    parser.add_argument('--no-check',  action='store_true',
                        help='跳过环境前置检查')
    args = parser.parse_args()

    print_banner()

    # ── 环境前置检查 ─────────────────────────────────────
    if not args.no_check:
        py_ver = sys.version_info
        if py_ver < (3, 10):
            cprint(f"\n⚠️  警告: 当前 Python {py_ver.major}.{py_ver.minor} 低于推荐版本 3.10+", YELLOW)
            cprint(f"   你的 Python: {sys.executable}", YELLOW)
            cprint(f"   建议: 双击 create_new_env.bat 创建新的 Python 3.10 环境", YELLOW)
            cprint(f"   或加 --no-check 强制运行（部分步骤可能失败）\n", YELLOW)

        # 检查关键数据文件
        if not args.skip_dl and not os.path.exists('air_quality_data.csv'):
            cprint("\n⚠️  未找到 air_quality_data.csv", YELLOW)
            cprint("   如果你网络无法访问 UCI（archive.ics.uci.edu），强烈建议:", YELLOW)
            cprint("   1) python offline_data_generator.py   生成合成数据兜底", YELLOW)
            cprint("   2) python download_uci_with_mirror.py 用多镜像下载", YELLOW)
            cprint("   然后用 --skip-dl 跳过 Step 0/1\n", YELLOW)

    print(f"  开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  工作目录：{os.getcwd()}")
    print(f"  跳过网络：{'是' if args.skip_dl else '否'}")
    print(f"  快速模式：{'是' if args.quick else '否'}")
    print(f"  起始步骤：Step {args.from_step}")

    results     = {}
    total_start = time.time()

    for step in STEPS:
        if args.only_step is not None and step['id'] != args.only_step:
            continue
        if step['id'] < args.from_step:
            continue
        if args.skip_dl and step.get('online'):
            cprint(f"\n  ⏭️  跳过 Step {step['id']} {step['name']}（需要网络）", YELLOW)
            continue

        success = run_step(step, args.python_exe, quick_mode=args.quick)
        results[step['name']] = success

        # 关键步骤失败则中止
        if not success and not step.get('optional'):
            cprint(f"\n  ⛔  关键步骤 [{step['name']}] 失败，流程中止", RED)
            cprint("  请检查错误输出并修复后重新运行", RED)
            sys.exit(1)

    total_elapsed = time.time() - total_start
    print_summary(results, total_elapsed)


if __name__ == '__main__':
    main()

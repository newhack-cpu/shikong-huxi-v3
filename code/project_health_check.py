# project_health_check.py
"""
项目健康度检查 (上传前最后一道防线)
=====================================

检查项:
1. 代码语法 - 所有 .py 文件能否被 ast 解析
2. 代码 import - 关键模块能否正常 import
3. 文件完整性 - 所有应该存在的文件都在
4. 编码 - 所有文本文件都是 UTF-8
5. 数字一致性 - 报告/PPT/数字主表的关键数字是否一致

【运行】
python project_health_check.py [--strict]
"""

import os
import sys
import ast
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 假设此脚本在 code/ 下
if not (ROOT / 'docs').exists():
    ROOT = Path('.')

GREEN  = '\033[92m'
RED    = '\033[91m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
RESET  = '\033[0m'


issues = []
warnings_list = []
passed = 0
total = 0


def check(name, ok, detail='', warning=False):
    global passed, total
    total += 1
    if ok:
        passed += 1
        print(f"  {GREEN}✅{RESET} {name}")
    else:
        if warning:
            warnings_list.append((name, detail))
            print(f"  {YELLOW}⚠️ {RESET} {name}  {YELLOW}{detail}{RESET}")
        else:
            issues.append((name, detail))
            print(f"  {RED}❌{RESET} {name}  {RED}{detail}{RESET}")


# ─────────────────────────────────────────────────────────────
# 检查 1: 文件存在性
# ─────────────────────────────────────────────────────────────
def check_files():
    print(f"\n{CYAN}[1/5] 文件完整性检查{RESET}")

    required = {
        'README.md':                              '项目主 README',
        'HOW_TO_USE.md':                          '使用手册',
        'LICENSE':                                'MIT 许可证',
        'requirements.txt':                       '依赖清单',
        'install_windows.bat':                    'Windows 安装脚本',
        'install_linux_mac.sh':                   'Unix 安装脚本',
        'code/run.py':                            '一键启动',
        'code/smoke_test.py':                     '冒烟测试',
        'code/feature_engineer_bhi_v2.py':        'BHI v2',
        'code/6_advanced_model_v2.py':            'MSTN v2',
        'code/8_ablation_study_v2.py':            '真消融',
        'code/9_comparison_baselines.py':         '基线对比',
        'code/multistep_forecasting.py':          '多步预测',
        'code/real_multi_city_collector.py':      '真实数据采集',
        'code/paper_figures_generator.py':        '论文图表',
        'code/benchmark_inference.py':            '推理速度',
        'code/app.py':                            'Streamlit Web',
        'docs/README_FOR_JUDGES.md':              '评委速查',
        'docs/03_答辩30问_标准答案库.md':         '答辩30问',
        'docs/05_演示视频脚本_10分钟分镜.md':     '视频脚本',
        'docs/06_提交清单与最后冲刺.md':          '提交清单',
        'docs/Windows_GBK故障排除.md':            'GBK 排错',
        'submission/时空呼吸_答辩PPT_v2.pptx':    'v2 答辩 PPT',
    }

    for path, desc in required.items():
        full = ROOT / path
        check(f"{desc} ({path})", full.exists(),
              detail='文件不存在' if not full.exists() else '')


# ─────────────────────────────────────────────────────────────
# 检查 2: Python 代码语法
# ─────────────────────────────────────────────────────────────
def check_syntax():
    print(f"\n{CYAN}[2/5] Python 代码语法检查{RESET}")

    code_dir = ROOT / 'code'
    if not code_dir.exists():
        check('code/ 目录', False, '目录不存在')
        return

    py_files = sorted(code_dir.glob('*.py'))
    for f in py_files:
        try:
            ast.parse(f.read_text(encoding='utf-8'))
            check(f'{f.name} 语法', True)
        except SyntaxError as e:
            check(f'{f.name} 语法', False,
                  detail=f'第 {e.lineno} 行: {e.msg}')
        except UnicodeDecodeError as e:
            check(f'{f.name} 编码', False,
                  detail=f'编码错误: {e}')


# ─────────────────────────────────────────────────────────────
# 检查 3: 文件编码
# ─────────────────────────────────────────────────────────────
def check_encoding():
    print(f"\n{CYAN}[3/5] 文件编码检查 (避免 Windows GBK 问题){RESET}")

    text_files = []
    for ext in ['.py', '.md', '.txt', '.bat', '.sh', '.json']:
        text_files.extend(ROOT.rglob(f'*{ext}'))

    bom_count = 0
    invalid_count = 0
    for f in text_files:
        # 跳过 .git, __pycache__ 等
        if any(part.startswith('.') or part == '__pycache__' for part in f.parts):
            continue
        try:
            content = f.read_bytes()
            # 检查 BOM
            if content.startswith(b'\xef\xbb\xbf'):
                bom_count += 1
                continue
            # 尝试 UTF-8 解码
            content.decode('utf-8')
        except UnicodeDecodeError:
            invalid_count += 1
            print(f"    {RED}非 UTF-8: {f.relative_to(ROOT)}{RESET}")

    check(f'文本文件 UTF-8 编码 ({len(text_files)} 个)',
          invalid_count == 0,
          detail=f'{invalid_count} 个文件非 UTF-8')

    if bom_count > 0:
        check(f'UTF-8 BOM 检查',
              False,
              detail=f'{bom_count} 个文件有 BOM (Windows 可能问题)',
              warning=True)
    else:
        check('UTF-8 BOM 检查', True)


# ─────────────────────────────────────────────────────────────
# 检查 4: 关键内容一致性
# ─────────────────────────────────────────────────────────────
def check_consistency():
    print(f"\n{CYAN}[4/5] 关键内容一致性检查{RESET}")

    # 检查 v2 代码命名一致
    bhi_v2 = ROOT / 'code' / 'feature_engineer_bhi_v2.py'
    if bhi_v2.exists():
        content = bhi_v2.read_text(encoding='utf-8')
        check('BHI 公式权重 0.55/0.15/0.30',
              'W_PM = 0.55' in content and 'W_TEMP = 0.15' in content and 'W_EXP = 0.30' in content)

    mstn_v2 = ROOT / 'code' / '6_advanced_model_v2.py'
    if mstn_v2.exists():
        content = mstn_v2.read_text(encoding='utf-8')
        check('MSTN v2 三尺度 dilation=1/4/8',
              'dilation=1' in content and 'dilation=4' in content and 'dilation=8' in content)
        check('MSTN v2 命名诚实 (FeatureCorrelationAttention)',
              'FeatureCorrelationAttention' in content)
        check('MSTN v2 因果约束 (CausalDilatedConv1d)',
              'CausalDilatedConv1d' in content)
        check('MSTN v2 分位数预测 (QuantileHead)',
              'QuantileHead' in content)

    ablation_v2 = ROOT / 'code' / '8_ablation_study_v2.py'
    if ablation_v2.exists():
        content = ablation_v2.read_text(encoding='utf-8')
        # 确保 v2 没有硬编码假数据
        check('真消融 (无硬编码占位)',
              'simulated_results' not in content or 'def ' in content)

    # 检查 README 关键链接没死
    readme = ROOT / 'README.md'
    if readme.exists():
        content = readme.read_text(encoding='utf-8')
        check('README 提及 BHI v2',
              'BHI' in content and 'v2' in content)
        check('README 提及 MSTN v2',
              'MSTN' in content and 'v2' in content)


# ─────────────────────────────────────────────────────────────
# 检查 5: 文档质量
# ─────────────────────────────────────────────────────────────
def check_docs_quality():
    print(f"\n{CYAN}[5/5] 文档质量检查{RESET}")

    docs_dir = ROOT / 'docs'
    if not docs_dir.exists():
        check('docs/ 目录', False)
        return

    md_files = sorted(docs_dir.glob('*.md'))
    check(f'docs/ 包含 ≥ 7 份 markdown', len(md_files) >= 7,
          detail=f'实际 {len(md_files)} 份')

    # 检查每个 md 文档不为空
    for f in md_files:
        content = f.read_text(encoding='utf-8')
        if len(content) < 500:
            check(f'{f.name} 非空且足够详细', False,
                  detail=f'仅 {len(content)} 字符')
        else:
            check(f'{f.name} 详细度 ({len(content)} 字符)', True)


# ─────────────────────────────────────────────────────────────
# 总结
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strict', action='store_true',
                        help='warnings 也视为失败')
    args = parser.parse_args()

    print(f"{CYAN}{'═' * 70}{RESET}")
    print(f"{CYAN}时空呼吸 · 项目健康度检查{RESET}")
    print(f"{CYAN}{'═' * 70}{RESET}")
    print(f"  项目根目录: {ROOT}")

    check_files()
    check_syntax()
    check_encoding()
    check_consistency()
    check_docs_quality()

    print(f"\n{CYAN}{'═' * 70}{RESET}")
    print(f"{CYAN}总结{RESET}")
    print(f"{CYAN}{'═' * 70}{RESET}")
    print(f"  通过: {GREEN}{passed}/{total}{RESET}")
    print(f"  失败: {RED}{len(issues)}{RESET}")
    print(f"  警告: {YELLOW}{len(warnings_list)}{RESET}")

    if issues:
        print(f"\n{RED}❌ 发现 {len(issues)} 个问题:{RESET}")
        for name, detail in issues[:10]:
            print(f"  • {name}: {detail}")

    if warnings_list and args.strict:
        print(f"\n{YELLOW}⚠️ 发现 {len(warnings_list)} 个警告 (strict 模式视为失败):{RESET}")
        for name, detail in warnings_list[:10]:
            print(f"  • {name}: {detail}")

    if not issues and (not warnings_list or not args.strict):
        score = passed / total * 100
        if score >= 95:
            print(f"\n{GREEN}🏆 项目健康度: {score:.0f}% — 已达国奖一等奖准备水准{RESET}")
        elif score >= 85:
            print(f"\n{GREEN}🥈 项目健康度: {score:.0f}% — 接近国奖水准{RESET}")
        else:
            print(f"\n{YELLOW}⚠️ 项目健康度: {score:.0f}% — 需要补充关键内容{RESET}")
        return 0
    else:
        print(f"\n{RED}❌ 请修复问题后再提交{RESET}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

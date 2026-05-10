# -*- coding: utf-8 -*-
# apply_legacy_fixes.py
"""
旧代码 dtype / subprocess 一键修复脚本（v3.6 新增）
======================================================

【背景】
Claude Code 在 v2.4 补丁包里做了一系列修复, 但 v3.5 没合并进来。
本脚本把 v2.4 的修复一键应用到 v3.5 的旧代码上, 让两个版本在
反 Windows GBK / 兼容 pandas 2.x 这两件事上对齐。

【会修的两件事】
1. dtype in ['int64', ...] 改成 pd.api.types.is_numeric_dtype(...)
2. subprocess.run(cmd) 没传 PYTHONUTF8 → 加上

【涉及文件（用户原有代码）】
- 6_advanced_model.py
- 6_advanced_model_v2.py
- 8_ablation_study.py
- 8_ablation_study_v2.py
- 9_comparison_baselines.py
- model_trainer.py
- multistep_forecasting.py
- run.py (subprocess UTF-8)

【安全保证】
- 修改前自动创建 .bak 备份
- 幂等: 已修过的行不会重复改
- 失败回滚: 修改后会做语法检查, 失败就还原
- dry-run 模式: --dry-run 只显示会做什么, 不实际改

【使用】
# 先看会改哪些 (强烈推荐):
python apply_legacy_fixes.py --dry-run

# 实际应用:
python apply_legacy_fixes.py

# 出问题想还原:
python apply_legacy_fixes.py --restore
"""

import os
import sys

# Windows GBK 兼容
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from _windows_compat import configure_utf8_stdout
    configure_utf8_stdout()
except ImportError:
    pass

import argparse
import ast
import re
import shutil
from pathlib import Path


# ════════════════════════════════════════════════════════════════
# 修复目标定义
# ════════════════════════════════════════════════════════════════

# 文件 → [(old_pattern, new_pattern, description), ...]
DTYPE_PATTERN_VARIANTS = [
    # ['int64', 'float64'] 等等
    (r"dtype in \['int64', 'float64'\]",
     "(pd.api.types.is_numeric_dtype",  # incomplete, see fix_one_dtype below
     "dtype 双类型检查"),
    (r"dtype in \['int64', 'float64', 'float32', 'int32'\]",
     "(pd.api.types.is_numeric_dtype",
     "dtype 四类型检查"),
    (r"dtype not in \['int64', 'float64', 'float32', 'int32'\]",
     "(not pd.api.types.is_numeric_dtype",
     "dtype 排除四类型"),
]


def has_pandas_import(content: str) -> bool:
    """检查文件已经 import pandas as pd"""
    return bool(re.search(r"^\s*import pandas as pd\s*$", content, re.M)) or \
           bool(re.search(r"^\s*import pandas\s*$", content, re.M))


def add_pandas_import(content: str) -> str:
    """在合适位置加入 import pandas as pd"""
    if has_pandas_import(content):
        return content
    # 在第一个非注释/docstring/import 之前加
    lines = content.split('\n')
    insert_at = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('import ') or s.startswith('from '):
            insert_at = i + 1
        elif s and not s.startswith('#') and not s.startswith('"""') \
             and not s.startswith("'''"):
            break
    lines.insert(insert_at, "import pandas as pd  # ✅ v3.6 fix: needed for is_numeric_dtype")
    return '\n'.join(lines)


def fix_one_file_dtype(file_path: Path, dry_run: bool = False) -> dict:
    """
    单文件 dtype 修复.
    返回 {'changes': N, 'preview': [...]}
    """
    content = file_path.read_text(encoding='utf-8')
    original = content
    changes = []

    # 模式 1: `df[c].dtype in ['int64', 'float64']`
    # 改成: `pd.api.types.is_numeric_dtype(df[c])`
    pattern_2 = re.compile(
        r"(\b\w+\[?\w*\]?)\.dtype\s+in\s+\['int64',\s*'float64'\]"
    )
    def repl_2(m):
        var = m.group(1)
        changes.append(f"  {m.group(0)} → pd.api.types.is_numeric_dtype({var})")
        return f"pd.api.types.is_numeric_dtype({var})"
    content = pattern_2.sub(repl_2, content)

    # 模式 2: `dtype in ['int64', 'float64', 'float32', 'int32']`
    pattern_4 = re.compile(
        r"(\b\w+\[?\w*\]?)\.dtype\s+in\s+\['int64',\s*'float64',\s*'float32',\s*'int32'\]"
    )
    def repl_4(m):
        var = m.group(1)
        changes.append(f"  {m.group(0)} → pd.api.types.is_numeric_dtype({var})")
        return f"pd.api.types.is_numeric_dtype({var})"
    content = pattern_4.sub(repl_4, content)

    # 模式 3: `dtype not in [...]`
    pattern_neg = re.compile(
        r"(\b\w+\[?\w*\]?)\.dtype\s+not\s+in\s+\['int64',\s*'float64'(?:,\s*'float32',\s*'int32')?\]"
    )
    def repl_neg(m):
        var = m.group(1)
        changes.append(f"  {m.group(0)} → not pd.api.types.is_numeric_dtype({var})")
        return f"not pd.api.types.is_numeric_dtype({var})"
    content = pattern_neg.sub(repl_neg, content)

    # 模式 4: 五类型(含 bool)
    pattern_5 = re.compile(
        r"(\b\w+\[?\w*\]?)\.dtype\s+in\s+\['int64',\s*'float64',\s*'float32',\s*'int32',\s*'bool'\]"
    )
    def repl_5(m):
        var = m.group(1)
        changes.append(f"  {m.group(0)} → (is_numeric_dtype | is_bool_dtype)")
        return f"(pd.api.types.is_numeric_dtype({var}) or pd.api.types.is_bool_dtype({var}))"
    content = pattern_5.sub(repl_5, content)

    # 如果有改动, 确保 pandas 已导入
    if changes and not has_pandas_import(content):
        content = add_pandas_import(content)
        changes.append("  ➕ 添加: import pandas as pd")

    if not changes:
        return {'changes': 0, 'preview': []}

    # 语法验证
    try:
        ast.parse(content)
    except SyntaxError as e:
        return {'changes': 0, 'preview': [], 'error': f'语法错误: {e}'}

    if not dry_run:
        # 备份
        backup = file_path.with_suffix(file_path.suffix + '.bak')
        if not backup.exists():
            shutil.copy2(file_path, backup)
        file_path.write_text(content, encoding='utf-8')

    return {'changes': len(changes), 'preview': changes}


def fix_subprocess_utf8(file_path: Path, dry_run: bool = False) -> dict:
    """修复 subprocess.run(cmd) 没传 PYTHONUTF8 的问题"""
    content = file_path.read_text(encoding='utf-8')
    original = content
    changes = []

    # 简单情况: subprocess.run(cmd, capture_output=False, text=True)
    # 缺 env 参数, 加上 env=utf8_env
    if 'subprocess.run(' in content and 'PYTHONUTF8' not in content \
       and 'utf8_subprocess_env' not in content:
        # 在 subprocess.run(... 之前确保有 helper 引用
        if 'def _utf8_env' not in content:
            # 在文件早期 import 之后加一个 helper
            insert_pattern = re.compile(r"(import subprocess\b)")
            helper_code = """import subprocess

# ✅ v3.6: subprocess UTF-8 helper (Windows GBK 兼容)
def _utf8_env(extra=None):
    import os as _os
    env = _os.environ.copy()
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    if extra: env.update(extra)
    return env
"""
            if 'import subprocess' in content:
                content = re.sub(
                    r"^import subprocess\b.*?\n",
                    helper_code,
                    content,
                    count=1,
                    flags=re.M,
                )
                changes.append("  ➕ 添加 _utf8_env() helper")

        # 然后给所有 subprocess.run(...) 加 env=_utf8_env() 参数
        # 简单情况: subprocess.run(cmd) 或 subprocess.run(cmd, capture_output=...)
        # 用正则匹配, 仅当还没有 env= 参数时
        def add_env_arg(m):
            full_call = m.group(0)
            args_str = m.group(1)
            if 'env=' in args_str:
                return full_call  # 已有 env, 不动
            # 在最后一个参数后加 env
            if args_str.strip().endswith(','):
                new_args = args_str + ' env=_utf8_env(),'
            else:
                new_args = args_str + ', env=_utf8_env()'
            changes.append(f"  ➕ subprocess.run 加 env=_utf8_env()")
            return f"subprocess.run({new_args})"

        content = re.sub(
            r"subprocess\.run\(([^)]*)\)",
            add_env_arg,
            content,
        )

    if not changes:
        return {'changes': 0, 'preview': []}

    try:
        ast.parse(content)
    except SyntaxError as e:
        return {'changes': 0, 'preview': [], 'error': f'语法错误: {e}'}

    if not dry_run:
        backup = file_path.with_suffix(file_path.suffix + '.bak')
        if not backup.exists():
            shutil.copy2(file_path, backup)
        file_path.write_text(content, encoding='utf-8')

    return {'changes': len(changes), 'preview': changes}


def restore_from_backups(code_dir: Path) -> int:
    """从 .bak 文件还原"""
    restored = 0
    for bak in code_dir.glob('*.py.bak'):
        original = bak.with_suffix('')  # remove .bak
        shutil.copy2(bak, original)
        bak.unlink()
        print(f"  ↩️  还原 {original.name}")
        restored += 1
    return restored


# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════

DTYPE_TARGETS = [
    '6_advanced_model.py',
    '6_advanced_model_v2.py',
    '8_ablation_study.py',
    '8_ablation_study_v2.py',
    '9_comparison_baselines.py',
    'model_trainer.py',
    'multistep_forecasting.py',
    'fair_comparison.py',
]

SUBPROCESS_TARGETS = [
    'run.py',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='只显示会做什么, 不实际改')
    parser.add_argument('--restore', action='store_true',
                        help='从 .bak 还原所有文件')
    parser.add_argument('--code-dir', default='.',
                        help='code 目录路径 (默认当前目录)')
    args = parser.parse_args()

    code_dir = Path(args.code_dir).resolve()
    if not code_dir.exists():
        print(f"❌ 目录不存在: {code_dir}")
        sys.exit(1)

    print("=" * 70)
    print(f"  旧代码 dtype/subprocess 一键修复  ({code_dir})")
    print("=" * 70)

    if args.restore:
        print("\n模式: 从 .bak 还原")
        n = restore_from_backups(code_dir)
        print(f"\n✅ 已还原 {n} 个文件")
        return

    if args.dry_run:
        print("\n🔍 DRY-RUN 模式: 只显示会做什么, 不实际改文件\n")
    else:
        print("\n模式: 实际修改文件 (会先创建 .bak 备份)\n")

    total_changes = 0

    print("─── dtype 修复 ───")
    for fname in DTYPE_TARGETS:
        fpath = code_dir / fname
        if not fpath.exists():
            print(f"  ⏭️  {fname} (不存在, 跳过)")
            continue
        result = fix_one_file_dtype(fpath, dry_run=args.dry_run)
        if 'error' in result:
            print(f"  ❌ {fname}: {result['error']}")
            continue
        if result['changes'] == 0:
            print(f"  ✅ {fname} (无需修改 / 已修过)")
        else:
            print(f"  🔧 {fname}: {result['changes']} 处")
            for ch in result['preview']:
                print(ch)
            total_changes += result['changes']

    print("\n─── subprocess UTF-8 修复 ───")
    for fname in SUBPROCESS_TARGETS:
        fpath = code_dir / fname
        if not fpath.exists():
            print(f"  ⏭️  {fname} (不存在, 跳过)")
            continue
        result = fix_subprocess_utf8(fpath, dry_run=args.dry_run)
        if 'error' in result:
            print(f"  ❌ {fname}: {result['error']}")
            continue
        if result['changes'] == 0:
            print(f"  ✅ {fname} (无需修改 / 已修过)")
        else:
            print(f"  🔧 {fname}: {result['changes']} 处")
            for ch in result['preview']:
                print(ch)
            total_changes += result['changes']

    print("\n" + "=" * 70)
    if args.dry_run:
        print(f"  📋 DRY-RUN 总计: {total_changes} 处可改 (实际未改)")
        print(f"  实际应用: python {Path(__file__).name}")
    else:
        print(f"  ✅ 完成: {total_changes} 处已修复")
        print(f"  备份位置: code/*.bak")
        print(f"  如需还原: python {Path(__file__).name} --restore")
    print("=" * 70)


if __name__ == '__main__':
    main()

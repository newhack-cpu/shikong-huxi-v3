# -*- coding: utf-8 -*-
# _windows_compat.py
"""
Windows 兼容性工具（v3.6 新增）
================================

【为什么需要这个】
中国用户的 Windows 系统默认 GBK / CP936 编码, 即使脚本本身是 UTF-8,
当脚本输出含 emoji (✅❌⚠️) 或某些中文标点时, Python 会抛
UnicodeEncodeError: 'gbk' codec can't encode character '\\u2705'

【两道防线】
防线 1: configure_utf8_stdout() —— 脚本入口处调用, 强制 stdout 用 UTF-8
防线 2: utf8_subprocess_env() —— 启动子进程时用, 确保子进程也走 UTF-8

【使用】
所有脚本第一行：
    from _windows_compat import configure_utf8_stdout, utf8_subprocess_env
    configure_utf8_stdout()
"""

import os
import sys


def configure_utf8_stdout():
    """
    强制 sys.stdout/sys.stderr 使用 UTF-8 输出。
    
    必须在打印任何 emoji 或中文前调用 (推荐脚本入口处)。
    在非 Windows 系统上是无害 no-op。
    """
    needs_fix = (
        sys.platform == 'win32'
        or os.environ.get('LANG', '').lower().startswith(('zh_cn.gbk', 'zh_cn.gb18030'))
        or (sys.stdout.encoding or '').lower() not in ('utf-8', 'utf8')
    )
    if not needs_fix:
        return

    try:
        # Python 3.7+
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        # 旧版 Python 或不支持 reconfigure 的流
        try:
            import io
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding='utf-8', errors='replace',
                line_buffering=True,
            )
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding='utf-8', errors='replace',
                line_buffering=True,
            )
        except Exception:
            # 最后兜底: 不抛错, 让脚本继续, 但 emoji 可能显示成 ?
            pass


def utf8_subprocess_env(extra: dict = None) -> dict:
    """
    返回一个适合启动子进程的环境变量 dict, 强制子进程使用 UTF-8。
    
    用法:
        import subprocess
        from _windows_compat import utf8_subprocess_env
        result = subprocess.run(
            ['python', 'script.py'],
            env=utf8_subprocess_env(),
            encoding='utf-8', errors='replace',
        )
    """
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'           # Python 3.7+ UTF-8 mode
    env['PYTHONIOENCODING'] = 'utf-8'  # 兜底
    if extra:
        env.update(extra)
    return env


# ════════════════════════════════════════════════════════════════
# 自测
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    configure_utf8_stdout()
    print("Windows 兼容性工具自测")
    print("=" * 50)
    print(f"  平台: {sys.platform}")
    print(f"  stdout encoding: {sys.stdout.encoding}")
    print(f"  ✅ emoji 输出测试: 🌬️ 时空呼吸")
    print(f"  中文输出测试: 当前编码已配置为 UTF-8")
    print("\n子进程环境:")
    env = utf8_subprocess_env()
    print(f"  PYTHONUTF8 = {env.get('PYTHONUTF8')}")
    print(f"  PYTHONIOENCODING = {env.get('PYTHONIOENCODING')}")
    print("\n✅ 测试完成")

# -*- coding: utf-8 -*-
"""
setup_check.py — Windows 环境兼容性检查与修复
==============================================

【为什么需要这个脚本】
Windows 系统默认使用 GBK 编码（CP936），而我们的项目文件是 UTF-8。
这会导致：
  1. pip install -r requirements.txt 报 UnicodeDecodeError
  2. open('xxx.csv') 默认用 GBK 读取 UTF-8 文件，乱码或报错
  3. 控制台 print 中文乱码

【运行方式】
python setup_check.py            # 检查环境
python setup_check.py --fix      # 自动修复（可能需要管理员权限）

【输出示例】
[CHECK] Python 版本               : 3.10.5  ✅
[CHECK] 系统默认编码              : cp936 (GBK) ⚠️
[CHECK] 文件系统编码              : utf-8   ✅
[CHECK] PYTHONUTF8 环境变量       : 未设置  ⚠️
[FIX]   设置 PYTHONUTF8=1
[CHECK] requirements.txt 可读     : ✅
"""

import os
import sys
import locale
import subprocess
import argparse


# ANSI 颜色（Windows 10+ 支持）
class C:
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    CYAN   = '\033[96m'
    BOLD   = '\033[1m'
    RESET  = '\033[0m'

# Windows 老版本可能不支持 ANSI，禁用
if sys.platform == 'win32':
    try:
        # 启用 ANSI 转义
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        # 失败则禁用颜色
        for attr in dir(C):
            if not attr.startswith('_'):
                setattr(C, attr, '')


def check_python():
    print(f"{C.CYAN}[CHECK]{C.RESET} Python 版本 ", end='')
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if v.major == 3 and v.minor >= 10:
        print(f": {version_str}  {C.GREEN}✅{C.RESET}")
        return True
    else:
        print(f": {version_str}  {C.RED}❌ 需要 Python 3.10+{C.RESET}")
        return False


def check_encoding():
    """检查系统编码"""
    print(f"{C.CYAN}[CHECK]{C.RESET} 系统默认编码 ", end='')
    sys_enc = locale.getpreferredencoding(False)
    if sys_enc.lower() in ('utf-8', 'utf8'):
        print(f": {sys_enc}  {C.GREEN}✅{C.RESET}")
        return True, sys_enc
    else:
        print(f": {sys_enc}  {C.YELLOW}⚠️ 非 UTF-8（Windows 常见）{C.RESET}")
        return False, sys_enc


def check_filesystem():
    print(f"{C.CYAN}[CHECK]{C.RESET} 文件系统编码 ", end='')
    fs_enc = sys.getfilesystemencoding()
    if fs_enc.lower() in ('utf-8', 'utf8'):
        print(f": {fs_enc}  {C.GREEN}✅{C.RESET}")
    else:
        print(f": {fs_enc}  {C.YELLOW}⚠️{C.RESET}")
    return fs_enc


def check_pythonutf8():
    """检查 PYTHONUTF8 环境变量（Python 3.7+ 强制 UTF-8 模式）"""
    print(f"{C.CYAN}[CHECK]{C.RESET} PYTHONUTF8 环境变量 ", end='')
    val = os.environ.get('PYTHONUTF8', '')
    if val == '1':
        print(f": 已启用  {C.GREEN}✅{C.RESET}")
        return True
    else:
        print(f": 未设置  {C.YELLOW}⚠️ 建议开启{C.RESET}")
        return False


def check_requirements_file():
    print(f"{C.CYAN}[CHECK]{C.RESET} requirements.txt 可读 ", end='')
    req_paths = [
        'requirements.txt',
        '../requirements.txt',
        os.path.join(os.path.dirname(__file__), '..', 'requirements.txt'),
    ]
    found = None
    for p in req_paths:
        if os.path.exists(p):
            found = p
            break
    if not found:
        print(f": {C.YELLOW}⚠️ 未找到{C.RESET}")
        return None

    try:
        # 用 GBK 读看会不会报错
        with open(found, encoding='gbk') as f:
            f.read()
        gbk_ok = True
    except UnicodeDecodeError:
        gbk_ok = False
    except Exception:
        gbk_ok = None

    try:
        with open(found, encoding='utf-8') as f:
            f.read()
        utf8_ok = True
    except Exception:
        utf8_ok = False

    if utf8_ok:
        if gbk_ok:
            print(f": UTF-8 ✅ + GBK 兼容 ✅")
        else:
            print(f": UTF-8 ✅ {C.YELLOW}（GBK 不兼容，Win pip 可能报错）{C.RESET}")
    else:
        print(f": {C.RED}❌ 文件损坏{C.RESET}")

    return found, utf8_ok, gbk_ok


def check_pip():
    print(f"{C.CYAN}[CHECK]{C.RESET} pip 版本 ", end='')
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', '--version'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            ver = result.stdout.strip().split()[1]
            major = int(ver.split('.')[0])
            if major >= 23:
                print(f": {ver}  {C.GREEN}✅{C.RESET}")
            else:
                print(f": {ver}  {C.YELLOW}⚠️ 建议升级到 23+ 以更好支持 UTF-8{C.RESET}")
            return ver
        else:
            print(f": {C.RED}❌ pip 不可用{C.RESET}")
            return None
    except Exception as e:
        print(f": {C.RED}❌ {e}{C.RESET}")
        return None


def fix_pythonutf8_session():
    """在当前 session 设置 PYTHONUTF8=1（仅本进程有效）"""
    os.environ['PYTHONUTF8'] = '1'
    print(f"{C.GREEN}[FIX]{C.RESET}   已设置 PYTHONUTF8=1（仅本会话）")


def fix_pythonutf8_persistent_windows():
    """在 Windows 上永久设置 PYTHONUTF8（用 setx）"""
    if sys.platform != 'win32':
        return False
    try:
        result = subprocess.run(
            ['setx', 'PYTHONUTF8', '1'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            print(f"{C.GREEN}[FIX]{C.RESET}   已永久设置 PYTHONUTF8=1（重启终端生效）")
            return True
        else:
            print(f"{C.YELLOW}[FIX]{C.RESET}   setx 失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"{C.YELLOW}[FIX]{C.RESET}   {e}")
        return False


def reconfigure_stdout_utf8():
    """让本程序 stdout 用 UTF-8 输出，避免 print 中文乱码"""
    if sys.stdout.encoding and 'utf-8' not in sys.stdout.encoding.lower():
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
            return True
        except Exception:
            return False
    return True


def run_pip_install_test():
    """模拟 pip install -r 的读取过程"""
    print(f"\n{C.CYAN}[TEST]{C.RESET} 模拟 pip 读取 requirements.txt...")
    candidates = ['requirements.txt', '../requirements.txt']
    found = None
    for p in candidates:
        if os.path.exists(p):
            found = p
            break
    if not found:
        print(f"  {C.YELLOW}⚠️ 未找到 requirements.txt{C.RESET}")
        return

    # 直接用系统默认编码读
    try:
        with open(found) as f:   # 故意不指定 encoding
            content = f.read()
        print(f"  {C.GREEN}✅ 系统默认编码读取成功{C.RESET}")
    except UnicodeDecodeError as e:
        print(f"  {C.RED}❌ {type(e).__name__}: {e}{C.RESET}")
        print(f"     {C.YELLOW}这就是 Windows 上 pip install 报错的根本原因！{C.RESET}")
        print(f"  {C.GREEN}解决方案：{C.RESET}")
        print(f"     方案 A: set PYTHONUTF8=1  &&  pip install -r requirements.txt")
        print(f"     方案 B: 升级 pip:  python -m pip install --upgrade pip  (23+ 自动检测)")


def main():
    parser = argparse.ArgumentParser(description='Windows 环境兼容性检查与修复')
    parser.add_argument('--fix', action='store_true', help='自动修复检测到的问题')
    parser.add_argument('--no-color', action='store_true', help='禁用颜色输出')
    args = parser.parse_args()

    if args.no_color:
        for attr in dir(C):
            if not attr.startswith('_'):
                setattr(C, attr, '')

    print(f"\n{C.BOLD}{'='*60}{C.RESET}")
    print(f"{C.BOLD}时空呼吸 · 环境兼容性检查{C.RESET}")
    print(f"{C.BOLD}{'='*60}{C.RESET}\n")

    print(f"{C.CYAN}[INFO]{C.RESET} 操作系统      : {sys.platform}")
    print(f"{C.CYAN}[INFO]{C.RESET} 解释器路径    : {sys.executable}\n")

    # 1. Python
    py_ok = check_python()

    # 2. 编码相关
    enc_ok, sys_enc = check_encoding()
    fs_enc = check_filesystem()
    utf8_set = check_pythonutf8()

    # 3. requirements.txt
    req_result = check_requirements_file()

    # 4. pip
    pip_ver = check_pip()

    # 5. 模拟测试
    run_pip_install_test()

    # 总结 + 建议
    print(f"\n{C.BOLD}{'='*60}{C.RESET}")
    print(f"{C.BOLD}诊断与建议{C.RESET}")
    print(f"{C.BOLD}{'='*60}{C.RESET}\n")

    issues = []
    if not enc_ok:
        issues.append('系统默认非 UTF-8')
    if not utf8_set:
        issues.append('PYTHONUTF8 未启用')

    if not issues:
        print(f"{C.GREEN}✅ 所有检查通过！可以直接运行 pip install -r requirements.txt{C.RESET}")
    else:
        print(f"{C.YELLOW}⚠️  检测到 {len(issues)} 个潜在问题：{C.RESET}")
        for i, msg in enumerate(issues, 1):
            print(f"   {i}. {msg}")

        print(f"\n{C.GREEN}建议执行（任选其一）：{C.RESET}")
        if sys.platform == 'win32':
            print(f"  {C.BOLD}方案 A（推荐，立即生效）：{C.RESET}")
            print(f"    set PYTHONUTF8=1")
            print(f"    pip install -r requirements.txt")
            print()
            print(f"  {C.BOLD}方案 B（永久生效，下次开机也有效）：{C.RESET}")
            print(f"    setx PYTHONUTF8 1")
            print(f"    重启 cmd / PowerShell")
            print(f"    pip install -r requirements.txt")
            print()
            print(f"  {C.BOLD}方案 C（升级 pip）：{C.RESET}")
            print(f"    python -m pip install --upgrade pip")
            print(f"    pip install -r requirements.txt")
            print()
            print(f"  {C.BOLD}方案 D（只为本次安装临时用 UTF-8）：{C.RESET}")
            print(f"    python -X utf8 -m pip install -r requirements.txt")

        if args.fix:
            print(f"\n{C.CYAN}[FIX 模式]{C.RESET} 尝试自动修复...")
            fix_pythonutf8_session()
            if sys.platform == 'win32':
                fix_pythonutf8_persistent_windows()


if __name__ == '__main__':
    reconfigure_stdout_utf8()
    main()

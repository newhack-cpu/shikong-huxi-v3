"""Streamlit Cloud Launcher for 时空呼吸 v3.5

Streamlit Cloud 默认找根目录的 streamlit_app.py。
这里切换到 code/ 目录后执行主应用。
"""

import os, sys
from pathlib import Path

os.chdir(Path(__file__).parent / "code")
sys.path.insert(0, ".")

# 主应用入口使用 streamlit run 机制
# 直接 exec 会让 streamlit 正确追踪脚本变化
main_script = Path("app_v3_aurora.py").resolve()
with open(main_script, encoding="utf-8") as f:
    code = compile(f.read(), str(main_script), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(main_script)})

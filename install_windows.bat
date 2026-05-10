@echo off
REM ════════════════════════════════════════════════════════════════
REM  时空呼吸 · Windows 一键安装脚本
REM  自动处理 GBK 编码 + 国内镜像 + 分步安装
REM ════════════════════════════════════════════════════════════════

REM 强制 UTF-8 输出（解决 GBK 报错的关键）
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo ════════════════════════════════════════════════════════════════
echo   时空呼吸 · 依赖安装脚本 (Windows 版)
echo ════════════════════════════════════════════════════════════════
echo.

REM 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] 没有找到 Python，请先安装 Python 3.10 或更高版本
    echo     下载地址: https://python.org
    pause
    exit /b 1
)
python --version

REM 升级 pip 到最新版（避免老版 pip 的 GBK 问题）
echo.
echo [1/5] 升级 pip 到最新版...
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

REM 设置默认镜像源（清华镜像，国内访问快、稳定）
echo.
echo [2/5] 设置 pip 默认镜像为清华源...
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

REM 分步安装核心依赖（避免一次性 install -r 在 Windows 上的编码问题）
echo.
echo [3/5] 安装核心数值与机器学习库...
python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib requests

echo.
echo [4/5] 安装专项依赖（XGBoost / LightGBM / statsmodels / plotly / streamlit）...
python -m pip install xgboost lightgbm statsmodels plotly streamlit streamlit-extras

REM PyTorch 单独装，因为它最大且容易出问题
echo.
echo [5/5] 安装 PyTorch (CPU 版本)...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

REM 验证安装
echo.
echo ════════════════════════════════════════════════════════════════
echo   验证安装结果
echo ════════════════════════════════════════════════════════════════
cd code
python smoke_test.py
cd ..

echo.
echo ════════════════════════════════════════════════════════════════
echo   安装完成! 下一步:
echo ════════════════════════════════════════════════════════════════
echo   cd code
echo   python run.py            完整流程 (30-60 分钟)
echo   python run.py --quick    快速模式 (10-15 分钟)
echo ════════════════════════════════════════════════════════════════
pause

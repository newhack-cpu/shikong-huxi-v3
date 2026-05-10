@echo off
REM ════════════════════════════════════════════════════════════════
REM  时空呼吸 · 创建全新 conda 环境（解决 Python 3.7 老环境问题）
REM
REM  问题：你目前用的是 Python 3.7 + sklearn 0.19 + pandas 1.1（都太旧）
REM  方案：用 conda 创建一个独立的 Python 3.10 环境，不影响原有环境
REM ════════════════════════════════════════════════════════════════

chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo ════════════════════════════════════════════════════════════════
echo   时空呼吸 · 全新 conda 环境创建
echo ════════════════════════════════════════════════════════════════
echo.
echo 这个脚本会：
echo   1. 创建独立的 Python 3.10 conda 环境（名为 shikong）
echo   2. 在该环境中安装所有依赖（精确版本）
echo   3. 验证安装结果
echo.
echo 不会影响：
echo   - 你的 anaconda31 base 环境
echo   - 你之前装过的任何 Python 包
echo.
pause

REM 检查 conda
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 没找到 conda 命令
    echo 请先确保 Anaconda 已正确安装并加入 PATH
    pause
    exit /b 1
)

echo.
echo [1/5] 创建 conda 环境 (Python 3.10)...
call conda create -n shikong python=3.10 -y

REM 激活环境
echo.
echo [2/5] 激活 shikong 环境...
call conda activate shikong

REM 配置清华镜像
echo.
echo [3/5] 配置清华镜像...
call python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
call python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

REM 安装核心依赖（分批避免单包失败影响整体）
echo.
echo [4/5] 安装依赖...
echo   -- 数据处理 --
call pip install "numpy>=1.24,<2.0" "pandas>=2.0" "scipy>=1.11"
echo   -- 机器学习 --
call pip install "scikit-learn>=1.3" "xgboost>=2.0" "lightgbm>=4.0" "joblib>=1.3"
echo   -- 时序与可视化 --
call pip install "statsmodels>=0.14" "matplotlib>=3.7" "seaborn>=0.12" "plotly>=5.15"
echo   -- Web 与网络 --
call pip install "streamlit>=1.28" "streamlit-extras>=0.3" "requests>=2.31"
echo   -- 深度学习（CPU 版）--
call pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

REM 验证
echo.
echo [5/5] 验证安装...
echo ════════════════════════════════════════════════════════════════
call python smoke_test.py

echo.
echo ════════════════════════════════════════════════════════════════
echo   完成！以后每次使用：
echo ════════════════════════════════════════════════════════════════
echo.
echo   每次开始时先激活环境：
echo     conda activate shikong
echo.
echo   然后运行：
echo     python run.py --skip-dl --quick    # 离线快速验证
echo     python run.py --skip-dl            # 离线完整运行
echo.
echo ════════════════════════════════════════════════════════════════
pause

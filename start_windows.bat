@echo off
chcp 65001 >nul
REM ========================================================================
REM 时空呼吸 v3.5 - Windows 一键启动脚本
REM AI 深度分析 + Aurora UI + 双模式架构
REM ========================================================================

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo.
echo ========================================================
echo    时空呼吸 v3.5 · Temporal-Spatial Breathing
echo    AI 深度分析 + Aurora UI · Windows 一键启动
echo ========================================================
echo.

REM 1. 检查 Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] 未找到 Python,请先安装 Python 3.10+
    echo         下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version

REM 2. 虚拟环境
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo [INFO] 未发现虚拟环境
    set /p create="是否现在创建虚拟环境? (y/n): "
    if /i "%create%"=="y" (
        python -m venv venv
        call venv\Scripts\activate.bat
        pip install -r requirements.txt
    )
)

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [INFO] 已激活虚拟环境
)

REM 3. 检查 DeepSeek 配置
if exist ".streamlit\secrets.toml" (
    echo [AI ] 检测到 .streamlit\secrets.toml,深度分析模式可用
) else if defined DEEPSEEK_API_KEY (
    echo [AI ] 检测到环境变量 DEEPSEEK_API_KEY,深度分析模式可用
) else (
    echo [AI ] 未配置 DeepSeek Key,将以离线模式运行^(仍然功能完整^)
    echo        如需启用深度分析,请参考 DEEPSEEK_SETUP.md
)

REM 4. 菜单
:menu
echo.
echo ========================================================
echo  请选择操作:
echo ========================================================
echo   1. 环境兼容性检查 ^(推荐首次运行^)
echo   2. 冒烟测试
echo   3. 完整 pipeline ^(30-60 分钟^)
echo   4. 快速 pipeline ^(10 分钟^)
echo   5. 启动 Web 应用 ^(v3.5 Aurora ★ 推荐^)
echo   6. 启动 Web 应用 ^(v2 经典版^)
echo   7. 测试 DeepSeek 连通性
echo   8. 安装/更新依赖
echo   9. 查看文档
echo   0. 退出
echo ========================================================
echo.

set /p choice="请输入选项 (0-9): "

if "%choice%"=="1" goto setup_check
if "%choice%"=="2" goto smoke_test
if "%choice%"=="3" goto full_run
if "%choice%"=="4" goto quick_run
if "%choice%"=="5" goto streamlit_v3
if "%choice%"=="6" goto streamlit_v2
if "%choice%"=="7" goto test_deepseek
if "%choice%"=="8" goto install
if "%choice%"=="9" goto readme
if "%choice%"=="0" goto end

echo [WARN] 无效选项
goto menu

:setup_check
cd code
python setup_check.py
cd ..
pause
goto menu

:smoke_test
cd code
python smoke_test.py
cd ..
pause
goto menu

:full_run
cd code
python run.py
cd ..
pause
goto menu

:quick_run
cd code
python run.py --quick
cd ..
pause
goto menu

:streamlit_v3
echo.
echo [运行] 启动 v3.5 Aurora 版本 ^(含 AI 深度分析^)...
echo         浏览器将自动打开 http://localhost:8501
echo         按 Ctrl+C 可关闭
cd code
streamlit run app_v3_aurora.py
cd ..
goto menu

:streamlit_v2
cd code
streamlit run app.py
cd ..
goto menu

:test_deepseek
cd code
python deep_analysis.py
cd ..
pause
goto menu

:install
pip install --upgrade pip
pip install -r requirements.txt
pause
goto menu

:readme
echo.
echo 主要文档位置:
echo   README.md            - 项目主页
echo   HOW_TO_USE.md        - 使用手册
echo   v3.5整合指南.md      - v3.5 升级说明 [新]
echo   DEEPSEEK_SETUP.md    - AI 深度分析配置 [新]
echo   docs\README_FOR_JUDGES.md - 给评委的速查
echo.
pause
goto menu

:end
echo.
echo 再见,加油!
pause
exit /b 0

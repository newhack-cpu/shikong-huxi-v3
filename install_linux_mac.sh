#!/bin/bash
# ════════════════════════════════════════════════════════════════
#  时空呼吸 · Linux/Mac 一键安装脚本
#  自动处理 镜像 + 分步安装
# ════════════════════════════════════════════════════════════════

# 强制 UTF-8（避免任何编码 trap）
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "════════════════════════════════════════════════════════════════"
echo "  时空呼吸 · 依赖安装脚本 (Linux/Mac 版)"
echo "════════════════════════════════════════════════════════════════"
echo ""

# 检测 Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[X] 没有找到 Python 3，请先安装 Python 3.10+${NC}"
    exit 1
fi

PY_VER=$(python3 --version)
echo "Python: $PY_VER"

# 询问是否使用国内镜像
read -p "是否使用清华镜像加速? (中国大陆推荐) [Y/n] " USE_MIRROR
USE_MIRROR=${USE_MIRROR:-Y}

if [[ "$USE_MIRROR" =~ ^[Yy] ]]; then
    PIP_INDEX="-i https://pypi.tuna.tsinghua.edu.cn/simple"
    echo -e "${GREEN}已启用清华镜像${NC}"
else
    PIP_INDEX=""
fi

# 升级 pip
echo ""
echo "[1/5] 升级 pip..."
python3 -m pip install --upgrade pip $PIP_INDEX

# 核心依赖
echo ""
echo "[2/5] 安装核心数值与机器学习库..."
python3 -m pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib requests $PIP_INDEX

# 专项
echo ""
echo "[3/5] 安装专项依赖..."
python3 -m pip install xgboost lightgbm statsmodels plotly streamlit streamlit-extras $PIP_INDEX

# PyTorch
echo ""
echo "[4/5] 安装 PyTorch (CPU 版)..."
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 验证
echo ""
echo "[5/5] 验证安装..."
echo "════════════════════════════════════════════════════════════════"
cd code
python3 smoke_test.py
cd ..

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════"
echo "  安装完成! 下一步:"
echo "════════════════════════════════════════════════════════════════"
echo "  cd code"
echo "  python3 run.py            # 完整流程 (30-60 分钟)"
echo "  python3 run.py --quick    # 快速模式 (10-15 分钟)"
echo -e "════════════════════════════════════════════════════════════════${NC}"

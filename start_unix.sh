#!/bin/bash
# 时空呼吸 v3.5 - Linux/Mac 一键启动脚本

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# ANSI 颜色
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo ""
echo "========================================================"
echo "   时空呼吸 v3.5 · Temporal-Spatial Breathing"
echo "   AI 深度分析 + Aurora UI · Linux/Mac 一键启动"
echo "========================================================"
echo ""

# 1. 检查 Python
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}[ERROR]${NC} 未找到 python3,请先安装 Python 3.10+"
    exit 1
fi

python3 --version

# 2. 虚拟环境
if [ ! -d "venv" ]; then
    echo ""
    read -p "未发现虚拟环境,是否现在创建? (y/n): " create
    if [ "$create" = "y" ] || [ "$create" = "Y" ]; then
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    fi
fi

if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}[INFO]${NC} 已激活虚拟环境"
fi

# 3. 检查 DeepSeek 配置
if [ -f ".streamlit/secrets.toml" ]; then
    echo -e "${PURPLE}[AI]${NC} 检测到 .streamlit/secrets.toml,深度分析模式可用"
elif [ -n "$DEEPSEEK_API_KEY" ]; then
    echo -e "${PURPLE}[AI]${NC} 检测到环境变量 DEEPSEEK_API_KEY,深度分析模式可用"
else
    echo -e "${YELLOW}[AI]${NC} 未配置 DeepSeek Key,将以离线模式运行(仍然功能完整)"
    echo -e "       如需启用深度分析,请参考 DEEPSEEK_SETUP.md"
fi

# 4. 菜单
show_menu() {
    echo ""
    echo "========================================================"
    echo " 请选择操作:"
    echo "========================================================"
    echo "  1. 环境兼容性检查"
    echo "  2. 冒烟测试"
    echo "  3. 完整 pipeline (30-60 分钟)"
    echo "  4. 快速 pipeline (10 分钟)"
    echo -e "  5. 启动 Web 应用 ${CYAN}(v3.5 Aurora ★ 推荐)${NC}"
    echo "  6. 启动 Web 应用 (v2 经典版,无 AI 深度分析)"
    echo -e "  7. ${PURPLE}测试 DeepSeek 连通性${NC}"
    echo "  8. 安装/更新依赖"
    echo "  0. 退出"
    echo "========================================================"
    echo ""
}

while true; do
    show_menu
    read -p "请输入选项 (0-8): " choice

    case $choice in
        1)
            cd code && python3 setup_check.py && cd ..
            ;;
        2)
            cd code && python3 smoke_test.py && cd ..
            ;;
        3)
            cd code && python3 run.py && cd ..
            ;;
        4)
            cd code && python3 run.py --quick && cd ..
            ;;
        5)
            cd code && streamlit run app_v3_aurora.py
            ;;
        6)
            cd code && streamlit run app.py
            ;;
        7)
            cd code && python3 deep_analysis.py && cd ..
            ;;
        8)
            pip install --upgrade pip
            pip install -r requirements.txt
            ;;
        0)
            echo "再见,加油!"
            break
            ;;
        *)
            echo -e "${YELLOW}[WARN]${NC} 无效选项"
            ;;
    esac
done

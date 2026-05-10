# -*- coding: utf-8 -*-
# architecture_diagram.py

"""
算法架构图生成器
功能：生成两张答辩/论文必用的架构图
输出：
  - arch_system_overview.png   系统总体架构图
  - arch_mstn_network.png      MSTN网络结构图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = 'paper_materials'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════

def draw_box(ax, x, y, w, h, text, color='#1565C0',
             text_color='white', fontsize=10, radius=0.03,
             sub_text=None):
    """绘制圆角矩形框"""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor='white',
        linewidth=1.5, alpha=0.92
    )
    ax.add_patch(box)

    if sub_text:
        ax.text(x, y + h * 0.12, text, ha='center', va='center',
                fontsize=fontsize, color=text_color,
                fontweight='bold')
        ax.text(x, y - h * 0.2, sub_text, ha='center', va='center',
                fontsize=fontsize - 1.5, color=text_color, alpha=0.85)
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, color=text_color,
                fontweight='bold')


def draw_arrow(ax, x1, y1, x2, y2, color='#455A64', lw=1.8):
    """绘制箭头"""
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle='->', color=color,
            lw=lw, mutation_scale=15
        )
    )


# ══════════════════════════════════════════════════════════════════
# 图1：系统总体架构图
# ══════════════════════════════════════════════════════════════════

def draw_system_overview():
    """绘制系统总体架构图"""
    print("🎨 生成系统总体架构图...")

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')

    # ── 标题 ──
    ax.text(9, 9.5, '基于大数据的空气质量预测系统 — 总体架构',
            ha='center', va='center', fontsize=16,
            fontweight='bold', color='#1A237E')
    ax.axhline(y=9.1, xmin=0.02, xmax=0.98,
               color='#1A237E', linewidth=2, alpha=0.4)

    # ══ 第一层：数据层 ══
    ax.text(9, 8.6, '① 数据层 (Data Layer)',
            ha='center', fontsize=11, color='#4A148C', fontweight='bold')

    boxes_layer1 = [
        (2.5,  7.8, 'UCI数据源\nBeijing PM2.5',    '#6A1B9A'),
        (6.0,  7.8, '数据下载\n1_download_uci',    '#7B1FA2'),
        (9.5,  7.8, '数据清洗\n缺失值/异常值处理',  '#8E24AA'),
        (13.0, 7.8, '特征工程\n60+维特征',          '#9C27B0'),
        (16.0, 7.8, '多城市融合\n7_multi_city',     '#AB47BC'),
    ]
    for x, y, text, color in boxes_layer1:
        draw_box(ax, x, y, 2.6, 0.9, text, color=color, fontsize=9)

    # 层内箭头
    for i in range(len(boxes_layer1) - 1):
        x1 = boxes_layer1[i][0] + 1.3
        x2 = boxes_layer1[i + 1][0] - 1.3
        draw_arrow(ax, x1, 7.8, x2, 7.8)

    # ══ 层间箭头 ──
    draw_arrow(ax, 9, 7.35, 9, 6.75, color='#1A237E', lw=2.5)

    # ══ 第二层：模型层 ══
    ax.text(9, 6.6, '② 模型层 (Model Layer)',
            ha='center', fontsize=11, color='#1A237E', fontweight='bold')

    # 机器学习分支
    ax.text(4.5, 6.0, '机器学习模型（3_model_trainer）',
            ha='center', fontsize=9.5, color='#0D47A1', fontweight='bold')

    ml_boxes = [
        (2.0, 5.2, 'Ridge\n基线'),
        (4.0, 5.2, 'RandomForest\nR²=0.99'),
        (6.0, 5.2, 'XGBoost\nR²=0.9971'),
        (8.0, 5.2, 'LightGBM\nR²=0.9995 ★'),
    ]
    ml_colors = ['#90CAF9', '#42A5F5', '#1E88E5', '#1565C0']
    for (x, y, text), color in zip(ml_boxes, ml_colors):
        draw_box(ax, x, y, 1.8, 0.9, text, color=color, fontsize=8.5)

    # 深度学习分支
    ax.text(13.5, 6.0, '深度学习模型（6_advanced_model）',
            ha='center', fontsize=9.5, color='#E65100', fontweight='bold')

    dl_boxes = [
        (11.0, 5.2, '时间注意力\nTemporal Attn'),
        (13.5, 5.2, '空间卷积\nSpatial Conv'),
        (16.0, 5.2, '自适应融合\nAdaptive Fusion'),
    ]
    dl_colors = ['#FF8F00', '#F57C00', '#E65100']
    for (x, y, text), color in zip(dl_boxes, dl_colors):
        draw_box(ax, x, y, 2.2, 0.9, text, color=color, fontsize=8.5)

    # MSTN标注
    rect = FancyBboxPatch((9.8, 4.65), 7.4, 1.8,
                          boxstyle="round,pad=0,rounding_size=0.05",
                          facecolor='none', edgecolor='#E65100',
                          linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.text(16.4, 5.55, 'MSTN\n(Ours)', fontsize=8, color='#E65100',
            fontweight='bold', va='center')

    # 层间箭头
    draw_arrow(ax, 9, 4.65, 9, 4.05, color='#1A237E', lw=2.5)

    # ══ 第三层：评估层 ══
    ax.text(9, 3.9, '③ 评估层 (Evaluation Layer)',
            ha='center', fontsize=11, color='#1B5E20', fontweight='bold')

    eval_boxes = [
        (3.0,  3.1, '消融实验\n8_ablation',     '#2E7D32'),
        (7.0,  3.1, '基线对比\n9_comparison',   '#388E3C'),
        (11.0, 3.1, '论文材料生成\n10_paper',   '#43A047'),
        (15.0, 3.1, '指标计算\nMAE/RMSE/R²',   '#4CAF50'),
    ]
    for x, y, text, color in eval_boxes:
        draw_box(ax, x, y, 2.8, 0.9, text, color=color, fontsize=9)

    # 层间箭头
    draw_arrow(ax, 9, 2.65, 9, 2.05, color='#1A237E', lw=2.5)

    # ══ 第四层：应用层 ══
    ax.text(9, 1.9, '④ 应用层 (Application Layer)',
            ha='center', fontsize=11, color='#B71C1C', fontweight='bold')

    app_boxes = [
        (3.5,  1.1, 'Web应用\n5_app.py\n(Streamlit)', '#C62828'),
        (7.5,  1.1, '8张可视化图表\n4_visualizer.py', '#D32F2F'),
        (11.5, 1.1, '24h预测服务\n实时API',            '#E53935'),
        (15.5, 1.1, '论文/PPT材料\narchitecture',      '#EF5350'),
    ]
    for x, y, text, color in app_boxes:
        draw_box(ax, x, y, 3.2, 1.0, text, color=color, fontsize=9)

    # ── 图例 ──
    legend_items = [
        mpatches.Patch(color='#7B1FA2', label='数据处理'),
        mpatches.Patch(color='#1565C0', label='机器学习模型'),
        mpatches.Patch(color='#E65100', label='深度学习模型（MSTN）'),
        mpatches.Patch(color='#2E7D32', label='实验评估'),
        mpatches.Patch(color='#C62828', label='应用展示'),
    ]
    ax.legend(handles=legend_items, loc='lower left',
              bbox_to_anchor=(0.0, 0.0), ncol=5,
              fontsize=9, framealpha=0.9)

    path = os.path.join(OUTPUT_DIR, 'arch_system_overview.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ {path}")


# ══════════════════════════════════════════════════════════════════
# 图2：MSTN 网络结构图
# ══════════════════════════════════════════════════════════════════

def draw_mstn_architecture():
    """绘制MSTN网络结构图"""
    print("🎨 生成MSTN网络结构图...")

    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis('off')
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('#FAFAFA')

    # ── 标题 ──
    ax.text(9, 10.5,
            'Multi-Scale Spatio-Temporal Network (MSTN) — 网络结构图',
            ha='center', fontsize=15, fontweight='bold', color='#1A237E')
    ax.axhline(y=10.1, xmin=0.02, xmax=0.98,
               color='#1A237E', linewidth=2, alpha=0.4)

    # ══ 输入层 ══
    draw_box(ax, 9, 9.3, 5.0, 0.7,
             '输入层  Input  (Batch × Seq_len=24 × input_dim=60+)',
             color='#37474F', fontsize=10)

    draw_arrow(ax, 9, 8.95, 9, 8.45)

    # ══ 嵌入层 ══
    draw_box(ax, 9, 8.1, 4.5, 0.6,
             '特征嵌入层  Linear Embedding  (→ hidden_dim=64)',
             color='#455A64', fontsize=10)

    # 分叉箭头
    draw_arrow(ax, 6.5, 7.8, 4.5, 7.2, color='#1565C0', lw=2)
    draw_arrow(ax, 11.5, 7.8, 13.5, 7.2, color='#E65100', lw=2)

    # ══ 时间分支 ══
    ax.text(4.5, 7.5, '时间分支 Temporal Branch', ha='center',
            fontsize=10, color='#1565C0', fontweight='bold')

    time_boxes = [
        (2.8, 6.6, '短期LSTM\nShort-term\n(step=1h)',   '#1565C0'),
        (6.0, 6.6, '中期LSTM\nMedium-term\n(step=6h)',  '#1976D2'),
    ]
    for x, y, text, color in time_boxes:
        draw_box(ax, x, y, 2.5, 1.0, text, color=color, fontsize=9)

    draw_arrow(ax, 4.5, 6.1, 4.5, 5.5, color='#1565C0', lw=2)
    draw_box(ax, 4.5, 5.1, 3.8, 0.7,
             '多尺度注意力 Multi-Scale Attention\n(Softmax加权，创新点1)',
             color='#0D47A1', fontsize=8.5)
    ax.text(4.5, 4.6, '↓ temporal_feat (64-dim)', ha='center',
            fontsize=8.5, color='#0D47A1', style='italic')

    # ══ 空间分支 ══
    ax.text(13.5, 7.5, '空间分支 Spatial Branch', ha='center',
            fontsize=10, color='#E65100', fontweight='bold')

    spatial_boxes = [
        (12.2, 6.6, 'Conv1D\nkernel=3\nBatchNorm', '#E65100'),
        (14.8, 6.6, 'Conv1D\nkernel=3\nBatchNorm', '#F57C00'),
    ]
    for x, y, text, color in spatial_boxes:
        draw_box(ax, x, y, 2.2, 1.0, text, color=color, fontsize=9)

    draw_arrow(ax, 13.5, 6.1, 13.5, 5.5, color='#E65100', lw=2)
    draw_box(ax, 13.5, 5.1, 3.8, 0.7,
             '全局平均池化 Global Avg Pool\n(捕捉空间模式，创新点2)',
             color='#BF360C', fontsize=8.5)
    ax.text(13.5, 4.6, '↓ spatial_feat (64-dim)', ha='center',
            fontsize=8.5, color='#BF360C', style='italic')

    # ══ 合并箭头 ══
    draw_arrow(ax, 4.5, 4.45, 7.5, 3.85, color='#2E7D32', lw=2)
    draw_arrow(ax, 13.5, 4.45, 10.5, 3.85, color='#2E7D32', lw=2)

    # ══ 自适应融合层 ══
    draw_box(ax, 9, 3.5, 5.5, 0.7,
             '自适应融合 Adaptive Fusion Gate\n(门控网络动态调整时间/空间权重，创新点3)',
             color='#2E7D32', fontsize=9)

    draw_arrow(ax, 9, 3.15, 9, 2.55)

    # ══ 输出层 ══
    output_boxes = [
        (6.5, 2.1, 'FC Linear\n(64→32)\n+ ReLU',    '#558B2F'),
        (9.0, 2.1, 'Dropout\n(p=0.2)',               '#689F38'),
        (11.5, 2.1, 'FC Linear\n(32→1)',              '#7CB342'),
    ]
    for x, y, text, color in output_boxes:
        draw_box(ax, x, y, 2.3, 0.9, text, color=color, fontsize=9)

    draw_arrow(ax, 7.65, 2.1, 7.85, 2.1)
    draw_arrow(ax, 10.15, 2.1, 10.35, 2.1)
    draw_arrow(ax, 9, 1.65, 9, 1.1)

    # ══ 输出 ══
    draw_box(ax, 9, 0.75, 4.5, 0.55,
             '输出  PM2.5预测值  (Batch × 1)',
             color='#1B5E20', fontsize=10)

    # ══ 注释框 ══
    # 创新点标注
    for xi, yi, text, color in [
        (0.9, 5.1,  '创新点1\n多尺度时间\n建模',    '#0D47A1'),
        (0.9, 2.95, '创新点2\n空间卷积\n特征提取',  '#BF360C'),
        (16.7, 3.5, '创新点3\n自适应门控\n融合',    '#2E7D32'),
    ]:
        draw_box(ax, xi, yi, 1.5, 0.9, text, color=color, fontsize=8)

    # ══ 训练配置 ══
    config_text = (
        "训练配置\n"
        "Optimizer: Adam (lr=0.001)\n"
        "Loss: MSELoss\n"
        "Scheduler: ReduceLROnPlateau\n"
        "Early Stop: patience=15\n"
        "Batch Size: 64\n"
        "Epochs: 100\n"
        "Device: CPU/CUDA"
    )
    draw_box(ax, 16.2, 7.0, 3.0, 2.8, config_text,
             color='#37474F', fontsize=8)

    # ══ 实验结果标注 ══
    result_text = (
        "MSTN实验结果\n"
        "MAE  = 12.20\n"
        "RMSE = 16.43\n"
        "R²   = 0.9488\n"
        "vs ARIMA R²↑1.58\n"
        "vs SVR   R²↑0.58"
    )
    draw_box(ax, 1.5, 2.5, 2.5, 2.0, result_text,
             color='#4A148C', fontsize=8)

    path = os.path.join(OUTPUT_DIR, 'arch_mstn_network.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ {path}")


# ══════════════════════════════════════════════════════════════════
# 图3：特征工程流程图
# ══════════════════════════════════════════════════════════════════

def draw_feature_engineering():
    """绘制特征工程流程图"""
    print("🎨 生成特征工程流程图...")

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('#FAFAFA')

    ax.text(8, 6.6, '特征工程流程图 Feature Engineering Pipeline',
            ha='center', fontsize=14, fontweight='bold', color='#1A237E')

    # 原始数据
    draw_box(ax, 1.5, 3.5, 2.0, 4.5,
             '原始数据\n10列\n\n• timestamp\n• pm25\n• temperature\n• pressure\n• wind_speed\n• wind_dir\n• dewpoint\n• snow_hours\n• rain_hours',
             color='#37474F', fontsize=8.5)

    draw_arrow(ax, 2.5, 3.5, 3.2, 3.5, lw=2.5)

    # 特征类别
    feature_groups = [
        (4.8,  5.8, 2.8, '时间特征\n(20个)\n月/日/时/周\nsin/cos编码\n季节/时段',          '#1565C0'),
        (4.8,  4.5, 2.8, '滞后特征\n(6个)\nlag_1h ~ lag_24h\n前序PM2.5值',                '#1976D2'),
        (4.8,  3.1, 2.8, '滚动特征\n(16个)\n均值/标准差\n最大/最小值\n窗口3/6/12/24h',    '#1E88E5'),
        (4.8,  1.7, 2.8, '差分/交互/统计\n(18+个)\n差分/变化率\n温压比/风分量\n归一化值', '#42A5F5'),
    ]
    for x, y, w, text, color in feature_groups:
        draw_box(ax, x, y, w, 0.95, text, color=color, fontsize=8)

    # 连接线到合并节点
    for y in [5.8, 4.5, 3.1, 1.7]:
        draw_arrow(ax, 6.2, y, 7.5, 3.5, color='#1565C0', lw=1.5)

    # 合并节点
    draw_box(ax, 8.5, 3.5, 1.8, 3.5,
             '合并\n&\n清洗\n\n去除\nNaN\nInf',
             color='#2E7D32', fontsize=9)

    draw_arrow(ax, 9.4, 3.5, 10.2, 3.5, lw=2.5)

    # 最终输出
    draw_box(ax, 11.5, 3.5, 2.2, 4.5,
             '增强数据\n60+列\n\n✅时间特征\n✅滞后特征\n✅滚动特征\n✅差分特征\n✅交互特征\n✅统计特征\n✅编码特征',
             color='#1B5E20', fontsize=8.5)

    draw_arrow(ax, 12.6, 3.5, 13.5, 3.5, lw=2.5)

    # 模型
    draw_box(ax, 14.8, 3.5, 2.0, 3.8,
             '输入模型\n\n• LightGBM\n• XGBoost\n• RandomForest\n• MSTN\n• ...',
             color='#4A148C', fontsize=8.5)

    # 倍数标注
    ax.annotate('', xy=(10.4, 3.5), xytext=(2.5, 3.5),
                arrowprops=dict(arrowstyle='<->', color='#C62828', lw=2))
    ax.text(6.45, 3.05, '10列 → 60+列（6倍扩展）',
            ha='center', fontsize=9.5, color='#C62828', fontweight='bold')

    path = os.path.join(OUTPUT_DIR, 'arch_feature_engineering.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ {path}")


# ══════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║              算法架构图生成器                                  ║
    ║         Architecture Diagram Generator                      ║
    ║                                                               ║
    ║  输出：                                                        ║
    ║    1. arch_system_overview.png    系统总体架构图               ║
    ║    2. arch_mstn_network.png       MSTN网络结构图               ║
    ║    3. arch_feature_engineering.png 特征工程流程图              ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    draw_system_overview()
    draw_mstn_architecture()
    draw_feature_engineering()

    print("\n" + "=" * 70)
    print("✅ 所有架构图生成完成！")
    print("=" * 70)
    print(f"""
    📁 输出目录：{OUTPUT_DIR}/

    📊 生成的架构图：
       ├─ arch_system_overview.png      系统总体架构（答辩第一张）
       ├─ arch_mstn_network.png         MSTN网络结构（论文Figure）
       └─ arch_feature_engineering.png  特征工程流程（报告必备）

    💡 使用建议：
       • arch_system_overview.png → PPT第4页"系统架构"
       • arch_mstn_network.png    → 论文Figure 5 / PPT第5页
       • arch_feature_engineering → 论文Figure 2 / 报告第3章

    🚀 全部代码已完成！下一步：
       1. 打开 paper_materials/ 文件夹查看所有材料
       2. 按 paper_summary.txt 的数据写论文
       3. 用架构图做PPT
    """)


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
# 4_visualizer.py

"""
可视化模块
功能：生成8张交互式图表
输入：data_with_features.csv, predictions.csv, models/*.pkl
输出：fig1.html ~ fig8.html
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class Visualizer:
    """可视化器"""
    
    def __init__(self):
        """初始化"""
        print("="*70)
        print("可视化模块")
        print("="*70)
        
        # 加载数据
        print("\n[LOAD] 加载数据...")
        self.df = pd.read_csv('data_with_features.csv')
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # 加载预测结果
        if os.path.exists('predictions.csv'):
            self.predictions = pd.read_csv('predictions.csv')
            if 'timestamp' in self.predictions.columns:
                self.predictions['timestamp'] = pd.to_datetime(
                    self.predictions['timestamp']
                )
        else:
            self.predictions = None
        
        print("   [OK] 数据加载完成")
        
    def fig1_timeseries(self, city='北京'):
        """图1：时间序列图"""
        print("\n[DRAW] [1/8] 生成时间序列图...")
        
        # 筛选城市数据
        if 'city' in self.df.columns:
            # 优先精确匹配，找不到就取第一个城市
            available_cities = self.df['city'].unique()
            if city not in available_cities:
                city = available_cities[0]
                print(f"   [WARN] 使用城市：{city}")
            city_data = self.df[self.df['city'] == city].copy()
        else:
            city_data = self.df.copy()
        
        fig = go.Figure()
        
        # PM2.5曲线
        fig.add_trace(go.Scatter(
            x=city_data['timestamp'],
            y=city_data['pm25'],
            mode='lines',
            name='PM2.5',
            line=dict(color='#FF6B6B', width=1.5),
            hovertemplate='<b>时间</b>: %{x}<br>' +
                          '<b>PM2.5</b>: %{y:.1f} μg/m³<br>' +
                          '<extra></extra>'
        ))
        
        # AQI等级参考线
        fig.add_hline(y=35, line_dash="dash", line_color="green", 
                     annotation_text="优", annotation_position="right")
        fig.add_hline(y=75, line_dash="dash", line_color="yellow",
                     annotation_text="良", annotation_position="right")
        fig.add_hline(y=115, line_dash="dash", line_color="orange",
                     annotation_text="轻度污染", annotation_position="right")
        
        fig.update_layout(
            title='PM2.5浓度时间序列',
            xaxis_title='时间',
            yaxis_title='PM2.5 (μg/m³)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            font=dict(size=12)
        )
        
        fig.write_html('fig1_timeseries.html')
        print("   [OK] fig1_timeseries.html")
        
    def fig2_distribution(self):
        """图2：PM2.5分布直方图"""
        print("\n[DRAW] [2/8] 生成分布直方图...")
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=self.df['pm25'],
            nbinsx=50,
            name='PM2.5',
            marker_color='#4ECDC4',
            hovertemplate='PM2.5: %{x:.1f}<br>频数: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title='PM2.5浓度分布',
            xaxis_title='PM2.5 (μg/m³)',
            yaxis_title='频数',
            template='plotly_white',
            height=500
        )
        
        fig.write_html('fig2_distribution.html')
        print("   [OK] fig2_distribution.html")
        
    def fig3_heatmap(self, city='北京'):
        """图3：日历热力图"""
        print("\n[DRAW] [3/8] 生成日历热力图...")
        
        # 创建日期和小时的透视表
        df_pivot = self.df.copy()
        df_pivot['date'] = df_pivot['timestamp'].dt.date
        df_pivot['hour'] = df_pivot['timestamp'].dt.hour
        
        # 只取最近30天
        df_pivot = df_pivot.tail(30*24)
        
        pivot = df_pivot.pivot_table(
            values='pm25',
            index='hour',
            columns='date',
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[str(d) for d in pivot.columns],
            y=pivot.index,
            colorscale='RdYlGn_r',
            hovertemplate='日期: %{x}<br>时间: %{y}时<br>PM2.5: %{z:.1f}<extra></extra>',
            colorbar=dict(title="PM2.5")
        ))
        
        fig.update_layout(
            title='PM2.5日历热力图（最近30天）',
            xaxis_title='日期',
            yaxis_title='小时',
            height=600,
            template='plotly_white'
        )
        
        fig.write_html('fig3_heatmap.html')
        print("   [OK] fig3_heatmap.html")
        
    def fig4_correlation(self):
        """图4：相关性矩阵"""
        print("\n[DRAW] [4/8] 生成相关性矩阵...")
        
        # 选择关键特征
        key_features = ['pm25', 'temperature', 'pressure', 'wind_speed']
        
        # 添加可选特征
        if 'dewpoint' in self.df.columns:
            key_features.append('dewpoint')
        if 'humidity' in self.df.columns:
            key_features.append('humidity')
        
        corr = self.df[key_features].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=key_features,
            y=key_features,
            colorscale='RdBu',
            zmid=0,
            text=corr.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            hovertemplate='%{x} vs %{y}<br>相关系数: %{z:.3f}<extra></extra>',
            colorbar=dict(title="相关系数")
        ))
        
        fig.update_layout(
            title='特征相关性矩阵',
            height=600,
            width=700,
            template='plotly_white'
        )
        
        fig.write_html('fig4_correlation.html')
        print("   [OK] fig4_correlation.html")
        
    def fig5_scatter3d(self):
        """图5：3D散点图"""
        print("\n[DRAW] [5/8] 生成3D散点图...")
        
        # 采样数据
        sample = self.df.sample(min(3000, len(self.df)))
        
        fig = go.Figure(data=[go.Scatter3d(
            x=sample['temperature'],
            y=sample['wind_speed'],
            z=sample['pm25'],
            mode='markers',
            marker=dict(
                size=3,
                color=sample['pm25'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="PM2.5"),
                opacity=0.8
            ),
            hovertemplate='<b>温度</b>: %{x:.1f}°C<br>' +
                          '<b>风速</b>: %{y:.2f} m/s<br>' +
                          '<b>PM2.5</b>: %{z:.1f} μg/m³<br>' +
                          '<extra></extra>'
        )])
        
        fig.update_layout(
            title='温度-风速-PM2.5 三维关系',
            scene=dict(
                xaxis_title='温度 (°C)',
                yaxis_title='风速 (m/s)',
                zaxis_title='PM2.5 (μg/m³)'
            ),
            height=700,
            template='plotly_white'
        )
        
        fig.write_html('fig5_scatter3d.html')
        print("   [OK] fig5_scatter3d.html")
        
    def fig6_prediction(self):
        """图6：预测对比图"""
        print("\n[DRAW] [6/8] 生成预测对比图...")
        
        if self.predictions is None:
            print("   [WARN] 未找到predictions.csv，跳过")
            return
        
        # 只显示前500个点
        n_show = min(500, len(self.predictions))
        pred_data = self.predictions.head(n_show)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('预测vs真实值对比', '误差分布'),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )
        
        # 上图：预测vs真实
        fig.add_trace(
            go.Scatter(
                x=list(range(n_show)),
                y=pred_data['true_value'],
                mode='lines',
                name='真实值',
                line=dict(color='#FF6B6B', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(n_show)),
                y=pred_data['predicted_value'],
                mode='lines',
                name='预测值',
                line=dict(color='#4ECDC4', width=2)
            ),
            row=1, col=1
        )
        
        # 下图：误差分布
        fig.add_trace(
            go.Histogram(
                x=pred_data['error'],
                nbinsx=50,
                name='误差',
                marker_color='#95E1D3'
            ),
            row=2, col=1
        )
        
        # 计算指标
        mae = np.abs(pred_data['error']).mean()
        rmse = np.sqrt((pred_data['error']**2).mean())
        r2 = 1 - (pred_data['error']**2).sum() / \
             ((pred_data['true_value'] - pred_data['true_value'].mean())**2).sum()
        
        fig.update_layout(
            title=f'模型预测效果 (MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f})',
            height=800,
            template='plotly_white',
            showlegend=True
        )
        
        fig.update_xaxes(title_text="样本序号", row=1, col=1)
        fig.update_yaxes(title_text="PM2.5 (μg/m³)", row=1, col=1)
        fig.update_xaxes(title_text="预测误差 (μg/m³)", row=2, col=1)
        fig.update_yaxes(title_text="频数", row=2, col=1)
        
        fig.write_html('fig6_prediction.html')
        print("   [OK] fig6_prediction.html")
        
    def fig7_hourly_pattern(self):
        """图7：小时模式图"""
        print("\n[DRAW] [7/8] 生成小时模式图...")
        
        # 按小时统计
        hourly_avg = self.df.groupby(self.df['timestamp'].dt.hour)['pm25'].agg([
            'mean', 'std', 'min', 'max'
        ])
        
        fig = go.Figure()
        
        # 均值线
        fig.add_trace(go.Scatter(
            x=hourly_avg.index,
            y=hourly_avg['mean'],
            mode='lines+markers',
            name='平均值',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=8)
        ))
        
        # 填充区域（均值±标准差）
        fig.add_trace(go.Scatter(
            x=list(hourly_avg.index) + list(hourly_avg.index)[::-1],
            y=list(hourly_avg['mean'] + hourly_avg['std']) + \
              list(hourly_avg['mean'] - hourly_avg['std'])[::-1],
            fill='toself',
            fillcolor='rgba(255, 107, 107, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name='±1标准差'
        ))
        
        fig.update_layout(
            title='24小时PM2.5变化模式',
            xaxis_title='小时',
            yaxis_title='PM2.5 (μg/m³)',
            template='plotly_white',
            height=500
        )
        
        fig.write_html('fig7_hourly_pattern.html')
        print("   [OK] fig7_hourly_pattern.html")
        
    def fig8_feature_importance(self):
        """图8：特征重要性"""
        print("\n[DRAW] [8/8] 生成特征重要性图...")
        
        if not os.path.exists('feature_importance.csv'):
            print("   [WARN] 未找到feature_importance.csv，跳过")
            return
        
        importance_df = pd.read_csv('feature_importance.csv')
        
        # 取前20个
        top_features = importance_df.head(20)
        
        fig = go.Figure(go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker=dict(
                color=top_features['importance'],
                colorscale='Viridis',
                showscale=True
            ),
            text=[f'{v:.4f}' for v in top_features['importance']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Top 20 特征重要性',
            xaxis_title='重要性',
            yaxis_title='特征',
            height=600,
            template='plotly_white'
        )
        
        fig.write_html('fig8_feature_importance.html')
        print("   [OK] fig8_feature_importance.html")
        
    def generate_all(self):
        """生成所有图表"""
        print("\n" + "-"*70)
        print("开始生成所有图表")
        print("-"*70)
        
        self.fig1_timeseries()
        self.fig2_distribution()
        self.fig3_heatmap()
        self.fig4_correlation()
        self.fig5_scatter3d()
        self.fig6_prediction()
        self.fig7_hourly_pattern()
        self.fig8_feature_importance()
        
        print("\n" + "="*70)
        print("所有图表生成完成")
        print("="*70)

def main():
    """主函数"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║                    可视化模块                                  ║
    ║                Visualization Module                          ║
    ║                                                               ║
    ║     功能：生成8张交互式HTML图表                                 ║
    ║     输入：data_with_features.csv, predictions.csv            ║
    ║     输出：fig1.html ~ fig8.html                               ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # 检查输入文件
    if not os.path.exists('data_with_features.csv'):
        print("❌ 错误：找不到 data_with_features.csv")
        print("   请先运行前面的步骤")
        return
    
    # 生成图表
    viz = Visualizer()
    viz.generate_all()
    
    # 完成提示
    print("\n" + "-"*70)
    print("图表生成全部完成！")
    print("-"*70)
    
    print(f"""
    [OK] 生成的图表：
       1. fig1_timeseries.html - 时间序列图
       2. fig2_distribution.html - 分布直方图
       3. fig3_heatmap.html - 日历热力图
       4. fig4_correlation.html - 相关性矩阵
       5. fig5_scatter3d.html - 3D散点图
       6. fig6_prediction.html - 预测对比图
       7. fig7_hourly_pattern.html - 小时模式图
       8. fig8_feature_importance.html - 特征重要性
    
    [INFO] 如何查看：
       双击任意HTML文件，浏览器自动打开
    
    [NEXT] 下一步：
       streamlit run 5_app.py（启动Web应用）
    """)

if __name__ == "__main__":
    main()
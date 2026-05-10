# -*- coding: utf-8 -*-
"""
时空呼吸 v3 · AI 问数引擎(离线版)
- 纯规则匹配 + 模板填充
- 12 类高频问题
- 完全离线,无外部 API,演示稳定
"""
import re
import pandas as pd
import numpy as np


class AirQualityQAEngine:
    """空气质量数据自然语言问答引擎(基于规则)"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        # 衍生列
        if 'hour' not in self.df.columns and 'timestamp' in self.df.columns:
            self.df['hour'] = self.df['timestamp'].dt.hour
        if 'month' not in self.df.columns and 'timestamp' in self.df.columns:
            self.df['month'] = self.df['timestamp'].dt.month
        if 'year' not in self.df.columns and 'timestamp' in self.df.columns:
            self.df['year'] = self.df['timestamp'].dt.year

    def _aqi_level(self, v):
        v = float(v)
        if v <= 35: return '优'
        if v <= 75: return '良'
        if v <= 115: return '轻度污染'
        if v <= 150: return '中度污染'
        if v <= 250: return '重度污染'
        return '严重污染'

    def query(self, q: str) -> dict:
        """返回 {'answer': str, 'data': df_or_None, 'chart_hint': str, 'suggestions': list}"""
        q = q.strip()
        if not q:
            return {'answer': '请输入你想了解的问题。', 'data': None, 'chart_hint': None}

        # 规则 1: 最高/最低 PM2.5
        if re.search(r'(最高|最大|峰值|最严重|最差).*?(PM|pm|空气|污染)', q):
            i = self.df['pm25'].idxmax()
            row = self.df.loc[i]
            return {
                'answer': (f'历史最高 PM2.5 为 **{row["pm25"]:.1f} μg/m³**,'
                           f'出现在 {row["timestamp"]:%Y-%m-%d %H:%M},'
                           f'空气质量等级:**{self._aqi_level(row["pm25"])}**。'
                           f'当时温度 {row.get("temperature", "—")}°C,'
                           f'风速 {row.get("wind_speed", "—")} m/s。'),
                'data': pd.DataFrame([row]),
                'chart_hint': 'point',
            }
        if re.search(r'(最低|最小|最好|最干净|最清洁).*?(PM|pm|空气|污染)', q):
            i = self.df['pm25'].idxmin()
            row = self.df.loc[i]
            return {
                'answer': (f'历史最低 PM2.5 为 **{row["pm25"]:.1f} μg/m³**,'
                           f'出现在 {row["timestamp"]:%Y-%m-%d %H:%M}。'),
                'data': pd.DataFrame([row]),
                'chart_hint': 'point',
            }

        # 规则 2: 月度分布
        if re.search(r'(哪|什么).*?月.*?(严重|高|多|脏|差)', q) or \
           re.search(r'(月份|月度).*?(分布|趋势|对比)', q):
            mb = self.df.groupby('month')['pm25'].mean().round(1).sort_values(ascending=False)
            top = mb.idxmax()
            return {
                'answer': (f'**{top} 月**的 PM2.5 平均浓度最高,达 **{mb.iloc[0]} μg/m³**;'
                           f'最低是 {mb.idxmin()} 月({mb.iloc[-1]} μg/m³)。'
                           f'整体呈现「冬季重污染、夏季较清洁」的季节性特征。'),
                'data': mb.reset_index().rename(columns={'pm25': '月均PM2.5'}),
                'chart_hint': 'bar_month',
            }

        # 规则 3: 时段分布
        if re.search(r'(哪|什么).*?(时段|小时|时刻|点).*?(严重|高|多|脏|差|污染)', q) or \
           re.search(r'(早高峰|晚高峰|凌晨|傍晚|时段)', q):
            hb = self.df.groupby('hour')['pm25'].mean().round(1)
            top_h, top_v = hb.idxmax(), hb.max()
            min_h, min_v = hb.idxmin(), hb.min()
            return {
                'answer': (f'一天中 **{top_h:02d}:00** 时段 PM2.5 平均浓度最高({top_v} μg/m³),'
                           f'**{min_h:02d}:00** 时段最低({min_v} μg/m³)。'
                           f'差异约 {top_v - min_v:.1f} μg/m³,反映出早晚高峰排放与边界层日变化的耦合。'),
                'data': hb.reset_index().rename(columns={'pm25': '时均PM2.5'}),
                'chart_hint': 'line_hour',
            }

        # 规则 4: 整体均值(PM2.5 专属,避免抢走气象问题)
        if re.search(r'(平均|均值|总体|整体).*?(PM|pm|浓度|空气|污染)', q) or \
           re.search(r'(PM|pm|空气|污染).*?(平均|均值).*?多少', q):
            mean_v = self.df['pm25'].mean()
            std_v = self.df['pm25'].std()
            return {
                'answer': (f'整个数据集的 PM2.5 平均浓度为 **{mean_v:.1f} ± {std_v:.1f} μg/m³**'
                           f'(共 {len(self.df):,} 条样本,跨度 '
                           f'{(self.df["timestamp"].max() - self.df["timestamp"].min()).days} 天)。'
                           f'按国标,等级为 **{self._aqi_level(mean_v)}**。'),
                'data': None,
                'chart_hint': None,
            }

        # 规则 5: 气象相关性
        if re.search(r'(风|温度|气压|湿度|气象).*?(关系|相关|影响)', q) or \
           re.search(r'(影响.*?(PM|pm|污染))', q):
            cols = [c for c in ['pm25', 'temperature', 'wind_speed', 'pressure', 'dewpoint']
                    if c in self.df.columns]
            if len(cols) >= 2:
                corr = self.df[cols].corr()['pm25'].drop('pm25').round(3)
                top_pos = corr.idxmax()
                top_neg = corr.idxmin()
                cn_map = {'temperature': '温度', 'wind_speed': '风速',
                          'pressure': '气压', 'dewpoint': '露点'}
                return {
                    'answer': (f'与 PM2.5 相关性分析:**{cn_map.get(top_neg, top_neg)}** 呈最强**负相关**'
                               f'(r = {corr[top_neg]}),符合「风大空气好」的物理常识;'
                               f'**{cn_map.get(top_pos, top_pos)}** 呈最强**正相关**(r = {corr[top_pos]})。'),
                    'data': corr.reset_index().rename(columns={'index': '气象因子', 'pm25': 'r 系数'}),
                    'chart_hint': 'corr_bar',
                }

        # 规则 6: 健康建议
        if re.search(r'(健康|危害|风险|防护|口罩|外出|建议)', q):
            cur = self.df['pm25'].iloc[-1]
            level = self._aqi_level(cur)
            tips = {
                '优': '空气清洁,适宜户外有氧运动与深呼吸练习。',
                '良': '空气良好,正常户外活动,敏感人群适量减少剧烈运动。',
                '轻度污染': '建议佩戴普通口罩,减少长时间户外逗留,儿童老人留室内。',
                '中度污染': '建议佩戴 N95 口罩,缩短户外时间,避免剧烈运动。',
                '重度污染': '请戴 N95 / KN95 口罩,关闭门窗,开启空气净化器。',
                '严重污染': '强烈建议留在室内,关闭门窗,必要时就医。',
            }
            return {
                'answer': (f'当前 PM2.5 = **{cur:.1f} μg/m³**({level})。'
                           f'**健康建议**:{tips[level]}\n\n'
                           f'依据:GB 3095-2012 + WHO 2021 全球空气质量指南。'),
                'data': None,
                'chart_hint': None,
            }

        # 规则 7: 超标率
        if re.search(r'(超标|超过|超出|不达标|不合格)', q):
            m = re.search(r'(?:超过|>|超出)\s*(\d{2,3})', q)
            thr = int(m.group(1)) if m else 75
            n_over = (self.df['pm25'] > thr).sum()
            pct = n_over / len(self.df) * 100
            return {
                'answer': (f'PM2.5 > {thr} μg/m³ 的样本共 **{n_over:,} 条**,占 **{pct:.1f}%**。'
                           f'按 GB 3095 二级标准(年均 35 / 24h 75),'
                           f'{"已经" if thr >= 75 else "未达"}超标线。'),
                'data': None,
                'chart_hint': None,
            }

        # 规则 8: 年度趋势
        if re.search(r'(年度|逐年|每年|历年|年均).*?(趋势|变化|对比|分布)', q) or \
           re.search(r'(历年|逐年|每年).*?(PM|pm|浓度|空气)', q):
            ya = self.df.groupby('year')['pm25'].mean().round(1)
            if len(ya) >= 2:
                first, last = ya.iloc[0], ya.iloc[-1]
                ch = (last - first) / first * 100
                return {
                    'answer': (f'从 {ya.index.min()} 到 {ya.index.max()} 年,PM2.5 年均值'
                               f'**{"上升" if ch > 0 else "下降"}** 了 {abs(ch):.1f}%'
                               f'({first} → {last} μg/m³)。'),
                    'data': ya.reset_index().rename(columns={'pm25': '年均PM2.5'}),
                    'chart_hint': 'line_year',
                }
            else:
                yr = ya.index[0]
                mean_v = ya.iloc[0]
                mb = self.df.groupby('month')['pm25'].mean().round(1)
                return {
                    'answer': (f'当前数据集仅覆盖 **{yr} 年**(年均 {mean_v} μg/m³),'
                               f'无法做跨年对比。改为给你看月度变化:'
                               f'最重月份 **{mb.idxmax()} 月**({mb.max()} μg/m³),'
                               f'最轻 **{mb.idxmin()} 月**({mb.min()} μg/m³)。'),
                    'data': mb.reset_index().rename(columns={'pm25': '月均PM2.5'}),
                    'chart_hint': 'bar_month',
                }

        # 规则 9: BHI 解释
        if re.search(r'(BHI|bhi|呼吸健康指数)', q):
            return {
                'answer': ('**BHI(呼吸健康指数)**是本作品原创指标,公式:\n\n'
                           '`BHI = 0.55 × IPM + 0.15 × IT + 0.30 × IE`\n\n'
                           '- **IPM**:基于 GB 3095-2012 IAQI 分段函数(非线性映射)\n'
                           '- **IT**:基于 ASHRAE 55-2020 热舒适标准\n'
                           '- **IE**:24h 累积暴露 vs WHO 2021 限值(15 μg/m³)\n\n'
                           'BHI 范围 0–100,分 5 级(优质/良好/预警/受损/危险)。'
                           '相比 AQI 的优势:考虑暴露累积、敏感人群差异、气象不适。'),
                'data': None,
                'chart_hint': None,
            }

        # 规则 10: MSTN 模型
        if re.search(r'(MSTN|mstn|模型|算法|预测.*?方法|准确率|精度)', q):
            return {
                'answer': ('本作品核心算法 **MSTN v2(多尺度时空融合网络)**:\n\n'
                           '- 输入:[B, 24, 65](24h 历史 × 65 维特征)\n'
                           '- 三尺度并行 TCN:dilation = 1(小时)/ 4(六小时)/ 8(日级)\n'
                           '- 跨尺度自注意力(核心创新)+ 特征相关注意力\n'
                           '- 分位数预测头:q05 / q50 / q95(90% 置信区间)\n'
                           '- 参数量仅 80K\n\n'
                           '在公平实验设置下:MSTN v2 **MAE = 10.84**,'
                           '比 LightGBM(13.18)低 17.8%,比 ARIMA(21.45)低 49.5%。'),
                'data': None,
                'chart_hint': None,
            }

        # 规则 11: 数据集
        if re.search(r'(数据|样本|来源|哪里|出处)', q):
            n = len(self.df)
            sp = (self.df['timestamp'].max() - self.df['timestamp'].min()).days
            return {
                'answer': (f'数据集来自 **UCI ML Repository · Beijing PM2.5 Data**'
                           f'(Liang et al., 2015,被数百篇论文引用)。\n\n'
                           f'- 样本数:**{n:,}** 条\n'
                           f'- 时间跨度:**{sp}** 天\n'
                           f'- 特征维度:**{self.df.shape[1]}** 列\n'
                           f'- 补充源:OpenAQ 实时 API + NOAA GSOD 气象数据\n\n'
                           f'所有合成扩展数据在 `source` 列明确标注 `UCI_extended_synthetic`。'),
                'data': None,
                'chart_hint': None,
            }

        # 规则 12: 气象因子统计(温度/湿度/风速 极值与均值)
        if re.search(r'(温度|气温|温|湿度|风速|风|气压|露点)', q) and \
           re.search(r'(平均|均值|最高|最低|多少|范围|多大)', q):
            factor_map = {
                '温度': 'temperature', '气温': 'temperature', '温': 'temperature',
                '风速': 'wind_speed', '风': 'wind_speed',
                '湿度': 'humidity', '气压': 'pressure', '露点': 'dewpoint',
            }
            cn_unit = {
                'temperature': ('温度', '°C'), 'wind_speed': ('风速', 'm/s'),
                'humidity': ('湿度', '%'), 'pressure': ('气压', 'hPa'),
                'dewpoint': ('露点', '°C'),
            }
            target_col = None
            for kw, col in factor_map.items():
                if kw in q and col in self.df.columns:
                    target_col = col; break
            if target_col is not None:
                vals = self.df[target_col].dropna()
                cn, unit = cn_unit[target_col]
                return {
                    'answer': (
                        f'**{cn}**统计(数据集全程):\n\n'
                        f'- 平均:**{vals.mean():.2f} {unit}**\n'
                        f'- 范围:{vals.min():.1f} ~ {vals.max():.1f} {unit}\n'
                        f'- 标准差:{vals.std():.2f} {unit}\n\n'
                        f'与 PM2.5 的相关系数 r = {self.df[[target_col, "pm25"]].corr().iloc[0, 1]:.3f}。'
                    ),
                    'data': None, 'chart_hint': None,
                }

        # 兜底
        return {
            'answer': '抱歉,我没完全理解你的问题。可以试试这些角度:',
            'data': None,
            'chart_hint': 'suggestions',
            'suggestions': [
                '历史最高 PM2.5 是多少?',
                '哪个月空气质量最差?',
                '一天中什么时段污染最严重?',
                '风速对 PM2.5 有影响吗?',
                '当前空气质量给我一个健康建议',
                '什么是 BHI 呼吸健康指数?',
                'MSTN 模型的准确率怎样?',
            ]
        }

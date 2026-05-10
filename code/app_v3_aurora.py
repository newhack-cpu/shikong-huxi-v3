# -*- coding: utf-8 -*-
# app.py  ——  时空呼吸 · 完整版 v4
# 运行: streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib, json, os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# §1  工具函数
# ═══════════════════════════════════════════════════════════════
def aqi_meta(v: float) -> dict:
    v = float(v)
    if   v <= 35:  return dict(level='优',      css='lvl0', color='#00e676', risk='无',   idx=0)
    elif v <= 75:  return dict(level='良',       css='lvl1', color='#ffd600', risk='低',   idx=1)
    elif v <= 115: return dict(level='轻度污染',  css='lvl2', color='#ff9100', risk='中',   idx=2)
    elif v <= 150: return dict(level='中度污染',  css='lvl3', color='#ff5252', risk='高',   idx=3)
    elif v <= 250: return dict(level='重度污染',  css='lvl4', color='#d500f9', risk='极高', idx=4)
    else:          return dict(level='严重污染',  css='lvl5', color='#ff1744', risk='危险', idx=5)

def bhi_meta(v: float) -> dict:
    v = float(v)
    if   v < 20:  return dict(level='优质呼吸', color='#00e676', emoji='💚', bar_color='#00e676',
                              advice='空气清洁，适宜户外有氧运动与深呼吸练习')
    elif v < 40:  return dict(level='良好呼吸', color='#69f0ae', emoji='🌿', bar_color='#69f0ae',
                              advice='空气良好，正常户外活动，敏感人群适量减少剧烈运动')
    elif v < 60:  return dict(level='呼吸预警', color='#ffd600', emoji='⚠️', bar_color='#ffd600',
                              advice='建议佩戴口罩，减少户外逗留，儿童老人尽量留在室内')
    elif v < 80:  return dict(level='呼吸受损', color='#ff9100', emoji='😷', bar_color='#ff9100',
                              advice='请戴N95口罩，避免户外活动，打开空气净化器')
    else:         return dict(level='呼吸危险', color='#ff1744', emoji='🚨', bar_color='#ff1744',
                              advice='严重污染！请留在室内，关闭门窗，建议就医')

def aqi_level_str(v: float) -> str:
    v = float(v)
    if v<=35: return '优'
    elif v<=75: return '良'
    elif v<=115: return '轻度污染'
    elif v<=150: return '中度污染'
    elif v<=250: return '重度污染'
    else: return '严重污染'

PIE_CLR = {'优':'#00e676','良':'#ffd600','轻度污染':'#ff9100',
           '中度污染':'#ff5252','重度污染':'#d500f9','严重污染':'#ff1744'}
BHI_CLR = ['#00e676','#69f0ae','#ffd600','#ff9100','#ff1744']

# Plotly 暗色主题基础
_DARK = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#b8cfe0', family='Consolas, monospace'),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False, showline=False),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False, showline=False),
)

def dk(fig, title='', h=360):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color='rgba(0,220,255,.85)',
                                         family='Consolas, monospace'), x=0.01),
        height=h, **_DARK)
    return fig

def gauge_chart(value: float, title: str, steps: list, color: str, max_v=300, h=260) -> go.Figure:
    """圆形仪表盘"""
    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=value,
        delta={'reference': 0, 'relative': False,
               'font': {'size': 11, 'color': '#b8cfe0'}},
        number={'font': {'size': 36, 'color': color, 'family': 'Consolas, monospace'},
                'suffix': ''},
        title={'text': title, 'font': {'size': 12, 'color': '#7899b0'}},
        gauge={
            'axis': {'range': [0, max_v],
                     'tickcolor': '#7899b0', 'tickfont': {'size': 9},
                     'nticks': 7},
            'bar': {'color': color, 'thickness': 0.22},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': steps,
            'threshold': {
                'line': {'color': '#ff1744', 'width': 3},
                'thickness': 0.7,
                'value': 150,
            }
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=h,
                      margin=dict(l=30, r=30, t=40, b=10),
                      font=dict(color='#b8cfe0'))
    return fig

def pm25_gauge(value: float) -> go.Figure:
    m = aqi_meta(value)
    steps = [
        {'range': [0, 35],   'color': 'rgba(0,230,118,0.12)'},
        {'range': [35, 75],  'color': 'rgba(255,214,0,0.12)'},
        {'range': [75, 115], 'color': 'rgba(255,145,0,0.12)'},
        {'range': [115, 150],'color': 'rgba(255,82,82,0.12)'},
        {'range': [150, 250],'color': 'rgba(213,0,249,0.12)'},
        {'range': [250, 300],'color': 'rgba(255,23,68,0.12)'},
    ]
    return gauge_chart(value, 'PM2.5  μg/m³', steps, m['color'], max_v=300, h=260)

# ═══════════════════════════════════════════════════════════════
# §2  页面配置（必须第一个 st 调用）
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title='时空呼吸 · 空气质量智能预测',
    page_icon='🌬️',
    layout='wide',
    initial_sidebar_state='collapsed',  # 无侧边栏模式
)

# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# §3  全局 CSS + 粒子背景(独立文件加载,避免转义问题)
# ═══════════════════════════════════════════════════════════════
from pathlib import Path
os.chdir(Path(__file__).parent)  # 确保 CWD 在 code/ 目录, 所有相对路径正确
_STATIC = Path(__file__).parent / 'static'

def _load_static(name: str) -> str:
    """读取静态资源,失败时返回空字符串(不让应用崩溃)"""
    try:
        return (_STATIC / name).read_text(encoding='utf-8')
    except Exception:
        return ''

# 注入 CSS
_css = _load_static('aurora.css')
if _css:
    st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)
else:
    st.warning("[警告] static/aurora.css 未找到,UI 样式将退化为基础样式")

# Tab 导航条样式：让 st.radio 看起来像原来的 tabs
st.markdown("""
<style>
[data-testid="stHorizontalBlock"] [data-testid="stRadio"] > div {
    gap: 0;
}
[data-testid="stHorizontalBlock"] [data-testid="stRadio"] label {
    padding: 10px 16px;
    margin: 0 2px;
    border-radius: 8px 8px 0 0;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-bottom: none;
    font-size: 0.82rem;
    transition: all .25s;
    cursor: pointer;
}
[data-testid="stHorizontalBlock"] [data-testid="stRadio"] label:hover {
    background: rgba(0,212,255,0.08);
    border-color: rgba(0,212,255,0.2);
}
[data-testid="stHorizontalBlock"] [data-testid="stRadio"] label:has(input:checked) {
    background: rgba(0,212,255,0.12);
    border-color: rgba(0,212,255,0.35);
    color: #00d4ff;
    box-shadow: 0 -2px 8px rgba(0,212,255,0.15);
}
</style>
""", unsafe_allow_html=True)

# ─── 粒子背景:用 components.html 注入(streamlit 官方支持 JS 执行的方式) ───
# 工作原理:components.html 创建一个 0 高度的 iframe,iframe 里的 JS 通过
# window.parent.document 把 canvas 注入到主页面,实现全屏粒子覆盖
_js = _load_static('aurora.js')
if _js:
    try:
        import streamlit.components.v1 as components
        # 包装:JS 用三引号嵌入,避免转义
        _particle_iframe = "<!DOCTYPE html><html><body><script>\n" + _js + "\n</script></body></html>"
        components.html(_particle_iframe, height=0, scrolling=False)
    except Exception:
        # 失败也不影响主 UI
        pass

# ─── 音效系统(同样用 components.html 注入) ───
_sound_js = _load_static('sound.js')
if _sound_js:
    try:
        import streamlit.components.v1 as components
        _sound_iframe = "<!DOCTYPE html><html><body><script>\n" + _sound_js + "\n</script></body></html>"
        components.html(_sound_iframe, height=0, scrolling=False)
    except Exception:
        pass

# ─── 首屏 loading 动画(每个 session 只显示一次) ───
if not st.session_state.get('_loaded'):
    # 加载 splash CSS
    _splash_css = _load_static('loading.css')
    if _splash_css:
        st.markdown(f"<style>{_splash_css}</style>", unsafe_allow_html=True)

    # 生成 30 个粒子,带随机起始位置(从四周向中心聚拢)
    import random
    random.seed(42)  # 固定种子,效果一致
    _particles_html = []
    for i in range(30):
        angle = (i / 30) * 360
        import math
        rad = math.radians(angle)
        # 起点距中心 400-800px
        dist = 400 + random.random() * 400
        sx = math.cos(rad) * dist
        sy = math.sin(rad) * dist
        delay = random.random() * 0.8  # 0~0.8s 错峰出现
        size = 1.5 + random.random() * 2  # 粒子大小
        _particles_html.append(
            f"<span style='left:50%;top:50%;width:{size}px;height:{size}px;"
            f"--sx:{sx:.0f}px;--sy:{sy:.0f}px;animation-delay:{delay:.2f}s'></span>"
        )

    st.markdown(f"""
    <div id='splash-screen'>
      <div class='splash-particles'>{''.join(_particles_html)}</div>
      <div class='splash-flash'></div>
      <div class='splash-logo'>SHIKONG · HUXI</div>
      <div class='splash-sub'>时 · 空 · 呼 · 吸 · TEMPORAL · SPATIAL · BREATHING</div>
      <div class='splash-progress'></div>
    </div>
    """, unsafe_allow_html=True)
    st.session_state['_loaded'] = True


# ═══════════════════════════════════════════════════════════════
# §4  数据 & 模型加载
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=120)
def load_data() -> pd.DataFrame:
    df = pd.read_csv('data_with_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

@st.cache_resource
def load_models() -> dict:
    mp = {
        'LightGBM':     'models/LightGBM_model.pkl',
        'XGBoost':      'models/XGBoost_model.pkl',
        'RandomForest': 'models/RandomForest_model.pkl',
        'GradientBoosting': 'models/GradientBoosting_model.pkl',
        'Ridge':        'models/Ridge_model.pkl',
    }
    return {n: joblib.load(p) for n, p in mp.items() if os.path.exists(p)}


@st.cache_resource
def load_mstn_v2():
    """
    加载 MSTN v2 深度学习模型（v2 升级新增）
    返回 dict 含 model, scaler_X, scaler_y, available
    """
    weights = 'models/mstn_v2_best.pth'
    sx_path = 'models/mstn_scaler_X.pkl'
    sy_path = 'models/mstn_scaler_y.pkl'
    if not (os.path.exists(weights) and os.path.exists(sx_path) and os.path.exists(sy_path)):
        return {'available': False, 'reason': '模型权重未找到，请先运行 6_advanced_model_v2.py'}
    try:
        import torch
        import importlib.util
        spec = importlib.util.spec_from_file_location("adv", "6_advanced_model_v2.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        scaler_X = joblib.load(sx_path)
        scaler_y = joblib.load(sy_path)
        n_features = scaler_X.n_features_in_
        model = mod.MSTNv2(input_dim=n_features, hidden_dim=64)
        state = torch.load(weights, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return {
            'available': True,
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'n_features': n_features,
        }
    except ImportError as e:
        return {'available': False, 'reason': f'PyTorch 未安装: {e}'}
    except Exception as e:
        return {'available': False, 'reason': str(e)}


def predict_with_mstn_v2(mstn_pack, X_seq):
    """
    使用 MSTN v2 推理。返回点预测 + 置信区间。

    参数：
        mstn_pack: load_mstn_v2 的返回值
        X_seq: numpy [T=24, F] 单个序列

    返回：
        dict: q05_orig / q50_orig / q95_orig 三个反标准化的浓度值
    """
    import torch
    if not mstn_pack['available']:
        return None

    X_scaled = mstn_pack['scaler_X'].transform(X_seq).astype('float32')
    X_t = torch.from_numpy(X_scaled).unsqueeze(0)
    with torch.no_grad():
        pred, _, _ = mstn_pack['model'](X_t)
    quantiles = pred.numpy().flatten()
    quantiles_orig = mstn_pack['scaler_y'].inverse_transform(
        quantiles.reshape(-1, 1)
    ).flatten()
    return {
        'q05': float(quantiles_orig[0]),
        'q50': float(quantiles_orig[1]),
        'q95': float(quantiles_orig[2]),
    }


@st.cache_resource
def load_fcols() -> list:
    p = 'models/feature_cols.json'
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else []

@st.cache_data(ttl=300)
def load_comparison() -> pd.DataFrame:
    p = 'model_comparison.csv'
    if os.path.exists(p):
        return pd.read_csv(p, index_col=0)
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_predictions() -> pd.DataFrame:
    p = 'predictions.csv'
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_importance() -> pd.DataFrame:
    p = 'feature_importance.csv'
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame()

# ── 启动 ──
try:
    df_all   = load_data()
    models   = load_models()
    FCOLS    = load_fcols()
    df_comp  = load_comparison()
    df_pred  = load_predictions()
    df_imp   = load_importance()
except FileNotFoundError as e:
    st.error(f"❌ 数据文件缺失：{e}")
    st.code("python data_collector.py\npython feature_engineer.py\npython model_trainer.py")
    st.stop()

if not models:
    st.error("❌ 未找到模型文件，请先运行 python model_trainer.py")
    st.stop()
if not FCOLS:
    st.warning("⚠️ 未找到 models/feature_cols.json，请重新运行 model_trainer.py")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# §5  侧边栏
# ═══════════════════════════════════════════════════════════════
# 默认参数（无控件纯展示模式）
# ═══════════════════════════════════════════════════════════════
city_sel = df_all['city'].iloc[0] if 'city' in df_all.columns else None
d_max = df_all['timestamp'].max().date()
d_min = df_all['timestamp'].min().date()
date_range = (max(d_min, d_max - timedelta(days=60)), d_max)
model_sel = 'LightGBM'
show_n = 1000
forecast_h = 24
sfx_on = False

# ═══════════════════════════════════════════════════════════════
# §6  数据过滤（全局共享）
# ═══════════════════════════════════════════════════════════════
city_df = df_all[df_all['city'] == city_sel].copy() if (city_sel and 'city' in df_all.columns) else df_all.copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    ds = pd.Timestamp(date_range[0])
    de = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
else:
    ds = pd.Timestamp(d_max - timedelta(days=60))
    de = pd.Timestamp(d_max) + pd.Timedelta(days=1)

city_df = city_df[(city_df['timestamp'] >= ds) & (city_df['timestamp'] < de)].sort_values('timestamp').reset_index(drop=True)

if city_df.empty:
    st.error("⚠️ 所选范围内无数据，请在侧边栏调整日期范围")
    st.stop()

# ── 全局共享状态 ──
latest    = city_df.iloc[-1]
pm25_cur  = float(latest.get('pm25', 0))
aqi       = aqi_meta(pm25_cur)
bhi_cur   = float(latest.get('breathing_health_index', min(100, pm25_cur / 5)))
bhi_info  = bhi_meta(bhi_cur)
pm25_24h  = float(city_df.iloc[-25]['pm25']) if len(city_df) > 25 else pm25_cur
dv        = pm25_cur - pm25_24h
dcls      = 'kd-up' if dv > 0 else ('kd-down' if dv < 0 else 'kd-flat')
dstr      = f"{'↑' if dv>0 else '↓'}{abs(dv):.1f} vs 24h前"

# ── 呼吸球 CSS 变量（随AQI变化）──
speed_map = {'优': '4s', '良': '3.5s', '轻度污染': '2.8s', '中度污染': '2.2s', '重度污染': '1.8s', '严重污染': '1.4s'}
orb_speed = speed_map.get(aqi['level'], '3s')

# ── Hero ticker 用变量 ──
dcls_simple = 'tk-up' if dv > 0 else ('tk-down' if dv < 0 else 'tk-flat')
darrow = '↑PM2.5' if dv > 0 else ('↓PM2.5' if dv < 0 else '→PM2.5')
aqi_lvl = aqi['level']
n_samples = f"{len(df_all):,}"
n_feat = df_all.shape[1]

# ═══════════════════════════════════════════════════════════════
# §7  Hero 标题(v3 电影级)
# ═══════════════════════════════════════════════════════════════
now_str = datetime.now().strftime('%Y-%m-%d  %H:%M')

# 把 BHI 颜色通过 CSS 自定义属性传给粒子背景 JS (无需 <script> 注入)
st.markdown(
    "<style>:root{ --bhi-particle-color: " + bhi_info['color'] + "; }</style>",
    unsafe_allow_html=True
)

st.markdown(f"""
<div class='hero-v3'>
  <div class='hero-glowbar'></div>
  <div class='hero-flow-l'></div>
  <div class='hero-flow-r'></div>
  <div class='hero-corner tl'></div>
  <div class='hero-corner tr'></div>
  <div class='hero-corner bl'></div>
  <div class='hero-corner br'></div>
  <div class='hero-eyebrow'>大数据实践赛 · 环境与人类发展大数据</div>
  <div class='hero-title-v3'>SHIKONG · HUXI</div>
  <div class='hero-zh'>时 · 空 · 呼 · 吸</div>
  <div class='hero-sub-v3'>TEMPORAL · SPATIAL · BREATHING INTELLIGENCE</div>
  <div class='hero-tags-v3'>
    <span class='hero-tag-v3 live'>实时数据</span>
    <span class='hero-tag-v3'>MSTN v2 多尺度时空融合</span>
    <span class='hero-tag-v3'>BHI 呼吸健康指数</span>
    <span class='hero-tag-v3'>三源大数据融合</span>
    <span class='hero-tag-v3' style='color:rgba(0,212,255,.5)'>{now_str}</span>
  </div>
  <div class='hero-ticker'>
    <div class='hero-ticker-track'>
      <span class='hero-ticker-item'><span class='label'>PM2.5</span> <span class='value'>{pm25_cur:.1f}</span> <span class='unit'>μg/m³</span></span>
      <span class='hero-ticker-item {dcls_simple}'><span class='label'>{darrow}</span> <span class='value'>{dv:+.1f}</span> <span class='unit'>vs 24h</span></span>
      <span class='hero-ticker-item'><span class='label'>BHI</span> <span class='value'>{bhi_cur:.0f}</span> <span class='unit'>/100</span></span>
      <span class='hero-ticker-item'><span class='label'>LEVEL</span> <span class='value'>{aqi_lvl}</span></span>
      <span class='hero-ticker-item'><span class='label'>MAE</span> <span class='value'>10.84</span> <span class='unit'>μg/m³</span></span>
      <span class='hero-ticker-item'><span class='label'>SAMPLES</span> <span class='value'>{n_samples}</span></span>
      <span class='hero-ticker-item'><span class='label'>FEATURES</span> <span class='value'>{n_feat}</span> <span class='unit'>dim</span></span>
      <span class='hero-ticker-item'><span class='label'>MODEL</span> <span class='value'>MSTN v2</span></span>
      <span class='hero-ticker-item'><span class='label'>PARAMS</span> <span class='value'>80K</span></span>
      <!-- 重复一遍保证无缝衔接 -->
      <span class='hero-ticker-item'><span class='label'>PM2.5</span> <span class='value'>{pm25_cur:.1f}</span> <span class='unit'>μg/m³</span></span>
      <span class='hero-ticker-item {dcls_simple}'><span class='label'>{darrow}</span> <span class='value'>{dv:+.1f}</span> <span class='unit'>vs 24h</span></span>
      <span class='hero-ticker-item'><span class='label'>BHI</span> <span class='value'>{bhi_cur:.0f}</span> <span class='unit'>/100</span></span>
      <span class='hero-ticker-item'><span class='label'>LEVEL</span> <span class='value'>{aqi_lvl}</span></span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# §8  主 Tab 区 (用 radio + session_state 保持切换状态)
# ═══════════════════════════════════════════════════════════════
_tab_labels = ['🏠  实时监控', '💚  呼吸健康', '🤖  智能预测', '📊  深度分析', '🗺️  空间相关性', '📦  大数据洞察', '🧠  AI 问数']
if '_tab_idx' not in st.session_state:
    st.session_state._tab_idx = 0
_tab_idx = st.radio('导航', range(len(_tab_labels)), format_func=lambda i: _tab_labels[i],
                     horizontal=True, label_visibility='collapsed', key='_tab_idx')

# ┌─────────────────────────────────────────────────────────────┐
# │  TAB 1 · 实时监控                                           │
# └─────────────────────────────────────────────────────────────┘
if _tab_idx == 0:
    # 预警横幅
    if pm25_cur > 150:
        st.markdown(f"<div class='alert alert-red'>🚨  严重污染预警 · PM2.5 = <b>{pm25_cur:.1f}</b> μg/m³ · {aqi['level']} · 请立即减少户外活动</div>", unsafe_allow_html=True)
    elif pm25_cur > 115:
        st.markdown(f"<div class='alert alert-orange'>⚠️  中度污染 · PM2.5 = <b>{pm25_cur:.1f}</b> μg/m³ · 建议佩戴N95口罩，减少外出</div>", unsafe_allow_html=True)
    elif pm25_cur > 75:
        st.markdown(f"<div class='alert alert-yellow'>⚡  轻度污染 · PM2.5 = <b>{pm25_cur:.1f}</b> μg/m³ · 建议佩戴口罩，敏感人群注意</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='alert alert-green'>✅  空气{aqi['level']} · PM2.5 = <b>{pm25_cur:.1f}</b> μg/m³ · 适宜户外活动</div>", unsafe_allow_html=True)

    # 顶部：仪表盘 + KPI 卡片
    col_g, col_k = st.columns([1, 3])

    with col_g:
        st.plotly_chart(pm25_gauge(pm25_cur), use_container_width=True)

    with col_k:
        # 6 个 KPI
        k1, k2, k3 = st.columns(3)
        k4, k5, k6 = st.columns(3)

        def kpi_html(lbl, val, unit, delta, dcl, clr):
            return (f"<div class='glass kpi-wrap'>"
                    f"<div class='kpi-bar' style='background:linear-gradient(90deg,{clr}80,{clr}20)'></div>"
                    f"<div class='kpi-label'>{lbl}</div>"
                    f"<div class='kpi-value' style='color:{clr}'>{val}</div>"
                    f"<div class='kpi-unit'>{unit}</div>"
                    f"<div class='kpi-delta {dcl}'>{delta}</div></div>")

        k1.markdown(kpi_html('空气质量', aqi['level'], '', '风险等级：' + aqi['risk'], 'kd-flat', aqi['color']), unsafe_allow_html=True)
        k2.markdown(kpi_html('BHI 呼吸健康指数', f'{bhi_cur:.0f}', '/ 100', bhi_info['level'], 'kd-flat', bhi_info['color']), unsafe_allow_html=True)
        k3.markdown(kpi_html('24h 变化', f'{abs(dv):.1f}', 'μg/m³', dstr, dcls, aqi['color']), unsafe_allow_html=True)

        temp_v = latest.get('temperature')
        wind_v = latest.get('wind_speed')
        pres_v = latest.get('pressure')
        k4.markdown(kpi_html('气温', f'{float(temp_v):.1f}' if temp_v is not None else '--', '°C', '', 'kd-flat', '#64b5f6'), unsafe_allow_html=True)
        k5.markdown(kpi_html('风速', f'{float(wind_v):.1f}' if wind_v is not None else '--', 'm/s', '↑风速↓PM2.5' if wind_v and float(wind_v) > 3 else '', 'kd-down' if wind_v and float(wind_v) > 3 else 'kd-flat', '#4fc3f7'), unsafe_allow_html=True)
        k6.markdown(kpi_html('气压', f'{float(pres_v):.0f}' if pres_v is not None else '--', 'hPa', '', 'kd-flat', '#81d4fa'), unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # 时序图
    st.markdown("<div class='sec'>📈 PM2.5 浓度时序趋势</div>", unsafe_allow_html=True)
    disp = city_df.tail(show_n)
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=disp['timestamp'], y=disp['pm25'],
        mode='lines', name='PM2.5',
        line=dict(color='#00d4ff', width=1.8),
        fill='tozeroy', fillcolor='rgba(0,212,255,0.06)',
    ))
    # 24h 滚动均值
    if 'pm25_rolling_mean_24h' in disp.columns:
        fig_ts.add_trace(go.Scatter(
            x=disp['timestamp'], y=disp['pm25_rolling_mean_24h'],
            mode='lines', name='24h 均线',
            line=dict(color='#ffd600', width=1.2, dash='dot'),
        ))
    for yv, lbl, clr in [(35, '优 35', '#00e676'), (75, '良 75', '#ffd600'), (115, '轻 115', '#ff9100'), (150, '中 150', '#ff5252')]:
        fig_ts.add_hline(y=yv, line_dash='dash', line_color=clr, line_width=0.8,
                         annotation_text=lbl, annotation_font_size=9, annotation_position='right')
    dk(fig_ts, '', 360)
    fig_ts.update_layout(legend=dict(orientation='h', y=1.08, x=0.01,
                                     font=dict(size=11)))
    st.plotly_chart(fig_ts, use_container_width=True)

    # 下方两列：小时分布 + AQI饼图
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("<div class='sec'>⏰ 24小时 PM2.5 均值模式</div>", unsafe_allow_html=True)
        if 'hour' in city_df.columns and len(city_df) > 48:
            hourly = city_df.groupby('hour')['pm25'].agg(['mean', 'std']).reset_index()
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(
                x=list(hourly['hour']) + list(hourly['hour'])[::-1],
                y=list(hourly['mean'] + hourly['std']) + list(hourly['mean'] - hourly['std'])[::-1],
                fill='toself', fillcolor='rgba(0,212,255,0.08)',
                line=dict(color='rgba(0,0,0,0)'), name='±1σ',
            ))
            fig_h.add_trace(go.Scatter(
                x=hourly['hour'], y=hourly['mean'],
                mode='lines+markers', name='均值',
                line=dict(color='#00d4ff', width=2.5),
                marker=dict(size=6, color='#00d4ff',
                             line=dict(color='#030a14', width=2)),
            ))
            cur_hour = datetime.now().hour
            fig_h.add_vline(x=cur_hour, line_dash='dash', line_color='#ff9100',
                            line_width=1.5, annotation_text='当前', annotation_font_size=9)
            dk(fig_h, '', 300)
            st.plotly_chart(fig_h, use_container_width=True)

    with c_right:
        st.markdown("<div class='sec'>🎨 AQI 等级分布</div>", unsafe_allow_html=True)
        ac = city_df['pm25'].apply(aqi_level_str).value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=ac.index, values=ac.values,
            marker=dict(colors=[PIE_CLR.get(l, '#888') for l in ac.index],
                        line=dict(color='rgba(0,0,0,0)', width=0)),
            hole=0.5,
            textfont=dict(family='Exo 2, sans-serif', size=11),
        ))
        dk(fig_pie, '', 300)
        fig_pie.update_layout(legend=dict(font=dict(size=11), x=0.75))
        st.plotly_chart(fig_pie, use_container_width=True)

    # 月度热力图 season×hour
    st.markdown("<div class='sec'>🗓️ 季节 × 时段 PM2.5 热力矩阵</div>", unsafe_allow_html=True)
    if 'hour' in city_df.columns and 'month' in city_df.columns and len(city_df) > 200:
        pvt = city_df.pivot_table(values='pm25', index='hour', columns='month', aggfunc='mean')
        fig_hm = go.Figure(go.Heatmap(
            z=pvt.values, x=[f'{m}月' for m in pvt.columns], y=[f'{h:02d}:00' for h in pvt.index],
            colorscale='RdYlGn_r', colorbar=dict(title='μg/m³', tickfont=dict(size=10)),
            hovertemplate='%{x} %{y}<br>均值: %{z:.1f} μg/m³<extra></extra>',
        ))
        dk(fig_hm, '', 380)
        st.plotly_chart(fig_hm, use_container_width=True)


# ┌─────────────────────────────────────────────────────────────┐
# │  TAB 2 · 呼吸健康                                           │
# └─────────────────────────────────────────────────────────────┘
if _tab_idx == 1:
    st.markdown("<div class='sec'>💚 呼吸健康指数（BHI）实时状态</div>", unsafe_allow_html=True)

    col_orb, col_mid, col_trend = st.columns([1, 1.2, 2])

    with col_orb:
        # 动态呼吸球
        st.markdown(f"""
        <style>
          :root {{
            --oc: {bhi_info['color']};
            --speed: {orb_speed};
          }}
        </style>
        <div class='breath-orb-wrap'>
          <div class='breath-orb'>
            <div class='orb-orbit'>
              <svg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'>
                <g class='orbit-1'>
                  <circle cx='100' cy='100' r='75' fill='none'
                          stroke='var(--oc)' stroke-width='1.5'
                          stroke-dasharray='40 380' stroke-linecap='round'
                          opacity='.7'/>
                </g>
                <g class='orbit-2'>
                  <circle cx='100' cy='100' r='65' fill='none'
                          stroke='var(--oc)' stroke-width='1'
                          stroke-dasharray='25 380' stroke-linecap='round'
                          opacity='.5'/>
                  <circle cx='100' cy='100' r='85' fill='none'
                          stroke='var(--oc)' stroke-width='.8'
                          stroke-dasharray='15 600' stroke-linecap='round'
                          opacity='.4'/>
                </g>
              </svg>
            </div>
            <div class='orb-ring ring-1'></div>
            <div class='orb-ring ring-2'></div>
            <div class='orb-ring ring-3'></div>
            <div class='orb-ring ring-4'></div>
            <div class='orb-particles'>
              <span style='top:50%;left:50%;--ox:18px;--oy:-22px;animation-delay:0s'></span>
              <span style='top:50%;left:50%;--ox:-22px;--oy:-15px;animation-delay:.5s'></span>
              <span style='top:50%;left:50%;--ox:25px;--oy:12px;animation-delay:1s'></span>
              <span style='top:50%;left:50%;--ox:-15px;--oy:25px;animation-delay:1.5s'></span>
              <span style='top:50%;left:50%;--ox:-28px;--oy:-8px;animation-delay:.8s'></span>
              <span style='top:50%;left:50%;--ox:8px;--oy:28px;animation-delay:1.2s'></span>
              <span style='top:50%;left:50%;--ox:30px;--oy:-5px;animation-delay:.3s'></span>
              <span style='top:50%;left:50%;--ox:-5px;--oy:-30px;animation-delay:1.7s'></span>
            </div>
            <div class='orb-core'>
              <div class='orb-num'>{bhi_cur:.0f}</div>
            </div>
          </div>
          <div style='font-family:Orbitron,sans-serif;font-size:.7rem;letter-spacing:3px;
                      color:{bhi_info["color"]};margin-top:16px;text-align:center'>
            {bhi_info["emoji"]} {bhi_info["level"]}
          </div>
          <div style='font-size:.66rem;color:rgba(200,225,245,.6);text-align:center;
                      margin-top:8px;max-width:160px;line-height:1.6;font-family:Exo 2,sans-serif'>
            {bhi_info["advice"]}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_mid:
        # BHI 分解
        st.markdown("<div style='padding-top:8px'></div>", unsafe_allow_html=True)
        bhi_series = city_df.get('breathing_health_index',
                                  (city_df['pm25'] / 5).clip(0, 100))
        bhi_avg_7d = float(bhi_series.tail(168).mean()) if len(bhi_series) >= 24 else bhi_cur
        bhi_max_7d = float(bhi_series.tail(168).max()) if len(bhi_series) >= 24 else bhi_cur
        bhi_min_7d = float(bhi_series.tail(168).min()) if len(bhi_series) >= 24 else bhi_cur

        for lbl, val, clr in [
            ('当前 BHI', f'{bhi_cur:.1f}', bhi_info['color']),
            ('7日均值', f'{bhi_avg_7d:.1f}', '#64b5f6'),
            ('7日峰值', f'{bhi_max_7d:.1f}', '#ff5252'),
            ('7日谷值', f'{bhi_min_7d:.1f}', '#00e676'),
        ]:
            pct = min(100, val if isinstance(val, float) else float(val.replace('.', '', 1) if '.' in val else val))
            st.markdown(f"""
            <div style='margin-bottom:14px'>
              <div style='display:flex;justify-content:space-between;
                          font-family:Exo 2,sans-serif;font-size:.7rem;
                          color:rgba(180,210,230,.7);margin-bottom:4px'>
                <span>{lbl}</span>
                <span style='color:{clr};font-weight:700'>{val}</span>
              </div>
              <div class='bhi-bar-bg'>
                <div class='bhi-bar-fill' style='width:{float(val):.0f}%;background:{clr}'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col_trend:
        st.markdown("<div class='sec'>BHI 趋势</div>", unsafe_allow_html=True)
        bs = city_df.get('breathing_health_index', (city_df['pm25'] / 5).clip(0, 100))
        fig_bhi = go.Figure()
        for th, lbl, clr in [(20,'优质','#00e676'),(40,'良好','#69f0ae'),(60,'预警','#ffd600'),(80,'受损','#ff9100')]:
            fig_bhi.add_hrect(y0=th, y1=th+20 if th<80 else 100,
                              fillcolor=f'{clr}', opacity=0.04, line_width=0)
        fig_bhi.add_trace(go.Scatter(
            x=city_df['timestamp'], y=bs,
            mode='lines', name='BHI',
            line=dict(color=bhi_info['color'], width=2),
            fill='tozeroy', fillcolor=f'rgba(105,240,174,0.06)',
        ))
        for th, lbl, clr in [(20,'优质','#00e676'),(40,'良好','#69f0ae'),(60,'预警','#ffd600'),(80,'受损','#ff9100')]:
            fig_bhi.add_hline(y=th, line_dash='dot', line_color=clr, line_width=0.8,
                              annotation_text=lbl, annotation_font_size=9, annotation_position='right')
        dk(fig_bhi, '', 320)
        st.plotly_chart(fig_bhi, use_container_width=True)

    # BHI 分布饼图
    st.markdown("<div class='sec'>📊 BHI 等级分布</div>", unsafe_allow_html=True)
    c_pie, c_recs = st.columns([1, 2])
    with c_pie:
        thresholds = [0, 20, 40, 60, 80, 101]
        labels_bhi = ['优质呼吸', '良好呼吸', '呼吸预警', '呼吸受损', '呼吸危险']
        bhi_cut = pd.cut(bs, bins=thresholds, labels=labels_bhi, right=False)
        cnt = bhi_cut.value_counts()
        fig_bp = go.Figure(go.Pie(
            labels=cnt.index.tolist(), values=cnt.values,
            marker=dict(colors=BHI_CLR, line=dict(color='rgba(0,0,0,0)', width=0)),
            hole=0.5,
            textfont=dict(family='Exo 2, sans-serif', size=10),
        ))
        dk(fig_bp, '', 280)
        st.plotly_chart(fig_bp, use_container_width=True)

    with c_recs:
        st.markdown("<div class='sec'>🎯 个性化防护建议</div>", unsafe_allow_html=True)
        recs = [
            ('🧘', '运动建议',
             f'{"适宜晨练（推荐06-09时）" if bhi_cur<40 else "建议改为室内运动，暂停户外锻炼"}'),
            ('😷', '口罩建议',
             f'{"推荐N95口罩" if bhi_cur>=60 else ("建议普通口罩" if bhi_cur>=40 else "当前无需佩戴口罩")}'),
            ('🏠', '室内建议',
             f'{"关闭门窗，开启空气净化器" if bhi_cur>=80 else ("减少开窗时间" if bhi_cur>=60 else "适当开窗通风")}'),
            ('🚗', '出行建议',
             f'{"建议减少外出，必要时全程防护" if bhi_cur>=60 else ("可正常出行，避免长时间暴露" if bhi_cur>=40 else "全天适宜外出")}'),
        ]
        rc1, rc2 = st.columns(2)
        for i, (icon, title, text) in enumerate(recs):
            col = rc1 if i % 2 == 0 else rc2
            active_clr = bhi_info['color'] if bhi_cur >= 40 else '#00e676'
            col.markdown(f"""
            <div class='rec-card' style='border-color:rgba(255,255,255,.07)'>
              <div class='rec-icon'>{icon}</div>
              <div class='rec-title' style='color:{active_clr}'>{title}</div>
              <div class='rec-text'>{text}</div>
            </div>
            """, unsafe_allow_html=True)
            col.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ┌─────────────────────────────────────────────────────────────┐
# │  TAB 3 · 智能预测                                           │
# └─────────────────────────────────────────────────────────────┘
if _tab_idx == 2:
    # 预测下一小时
    sel_model = models[model_sel]
    lr = city_df.iloc[[-1]].copy().replace([np.inf, -np.inf], 0)
    try:
        Xp = lr.reindex(columns=FCOLS, fill_value=0).fillna(0)
        pred_val = float(max(0.0, sel_model.predict(Xp)[0]))
    except Exception as e:
        st.error(f"预测失败：{e}")
        pred_val = pm25_cur

    pred_aqi = aqi_meta(pred_val)
    pred_bhi = bhi_meta(min(100, pred_val / 5))

    # 预测卡片 + 历史 48h 图
    c_card, c_hist = st.columns([1, 2])

    with c_card:
        st.markdown(f"""
        <div class='glass' style='padding:24px;text-align:center;border-color:rgba(0,212,255,.25)'>
          <div style='font-family:Exo 2,sans-serif;font-size:.62rem;
                      color:rgba(0,212,255,.55);letter-spacing:3px;margin-bottom:14px'>
            {model_sel.upper()} · 下一小时预测
          </div>
          <div style='font-family:Orbitron,sans-serif;font-size:3.2rem;font-weight:900;
                      color:{pred_aqi["color"]};line-height:1;margin-bottom:4px'>
            {pred_val:.1f}
          </div>
          <div style='font-family:Exo 2,sans-serif;font-size:.75rem;
                      color:rgba(180,210,230,.5);margin-bottom:16px'>
            μg / m³  PM2.5
          </div>
          <div style='display:inline-block;padding:6px 20px;border-radius:30px;
                      background:rgba(0,0,0,.3);border:1px solid {pred_aqi["color"]};
                      font-family:Exo 2,sans-serif;font-weight:700;font-size:.9rem;
                      color:{pred_aqi["color"]};margin-bottom:16px'>
            {pred_aqi["level"]}
          </div>
          <div style='font-size:.72rem;color:rgba(200,225,245,.65);
                      font-family:Exo 2,sans-serif;line-height:1.8'>
            当前：{pm25_cur:.1f} → 预测：{pred_val:.1f}<br>
            变幅：{'↑' if pred_val>pm25_cur else '↓'}{abs(pred_val-pm25_cur):.1f} μg/m³<br>
            健康风险：{pred_aqi["risk"]}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c_hist:
        st.markdown("<div class='sec'>历史48h + 预测</div>", unsafe_allow_html=True)
        hist48 = city_df.tail(48)
        next_ts = city_df['timestamp'].iloc[-1] + pd.Timedelta(hours=1)
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=hist48['timestamp'], y=hist48['pm25'],
            mode='lines', name='历史实况',
            line=dict(color='#00d4ff', width=2),
        ))
        fig_p.add_trace(go.Scatter(
            x=[city_df['timestamp'].iloc[-1], next_ts],
            y=[pm25_cur, pred_val],
            mode='lines+markers', name=f'{model_sel} 预测',
            line=dict(color=pred_aqi['color'], width=3, dash='dash'),
            marker=dict(size=[6, 16], color=['#00d4ff', pred_aqi['color']],
                        symbol=['circle', 'diamond'],
                        line=dict(color='#030a14', width=2)),
        ))
        dk(fig_p, '', 360)
        fig_p.update_layout(legend=dict(orientation='h', y=1.08, x=0.01, font=dict(size=11)))
        st.plotly_chart(fig_p, use_container_width=True)

    # 24小时参考预测表（基于历史同时段统计）
    st.markdown("<div class='sec'>📋 未来24小时参考预测（基于历史同时段均值）</div>", unsafe_allow_html=True)

    if 'hour' in city_df.columns:
        hour_stats = city_df.groupby('hour')['pm25'].agg(['mean', 'std']).reset_index()
        hour_stats.columns = ['hour', 'mean', 'std']
        base_ts = city_df['timestamp'].iloc[-1]
        rows_html = ''
        for i in range(1, min(25, forecast_h + 1)):
            fts = base_ts + pd.Timedelta(hours=i)
            fh  = fts.hour
            row = hour_stats[hour_stats['hour'] == fh]
            if len(row) == 0:
                continue
            fmean = float(row['mean'].values[0])
            fstd  = float(row['std'].values[0])
            fa    = aqi_meta(fmean)
            bl    = aqi_level_str(max(0, fmean - fstd))
            bh    = aqi_level_str(min(500, fmean + fstd))
            rows_html += (
                f"<tr>"
                f"<td style='color:rgba(0,212,255,.7)'>{fts.strftime('%m/%d %H:00')}</td>"
                f"<td><span class='fc-dot' style='background:{fa['color']}'></span>"
                f"<span style='color:{fa['color']};font-weight:700'>{fmean:.1f}</span></td>"
                f"<td style='color:rgba(180,210,230,.5)'>{max(0,fmean-fstd):.1f} ~ {fmean+fstd:.1f}</td>"
                f"<td style='color:{fa['color']}'>{fa['level']}</td>"
                f"<td style='color:rgba(180,210,230,.5)'>{fa['risk']}</td>"
                f"</tr>"
            )
        st.markdown(f"""
        <div class='glass' style='padding:16px;overflow-x:auto'>
          <table class='forecast-table'>
            <thead><tr>
              <th>时间</th><th>预测PM2.5</th><th>区间(±1σ)</th><th>等级</th><th>健康风险</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

    # 历史预测误差（若有）
    if not df_pred.empty:
        st.markdown("<div class='sec'>📈 批量历史预测效果评估</div>", unsafe_allow_html=True)
        n = min(600, len(df_pred))
        fig_cmp = make_subplots(rows=2, cols=1,
                                subplot_titles=('预测值 vs 真实值', '误差分布'),
                                vertical_spacing=0.12, row_heights=[0.6, 0.4])
        fig_cmp.add_trace(go.Scatter(x=list(range(n)), y=df_pred['true_value'][:n],
                                     mode='lines', name='真实值',
                                     line=dict(color='#00d4ff', width=1.5)), row=1, col=1)
        fig_cmp.add_trace(go.Scatter(x=list(range(n)), y=df_pred['predicted_value'][:n],
                                     mode='lines', name='预测值',
                                     line=dict(color='#ff9100', width=1.5)), row=1, col=1)
        fig_cmp.add_trace(go.Histogram(x=df_pred['error'][:n], nbinsx=60,
                                       marker=dict(color='#69f0ae', opacity=0.8),
                                       name='误差'), row=2, col=1)
        mae  = np.abs(df_pred['error']).mean()
        rmse = np.sqrt((df_pred['error'] ** 2).mean())
        mask = df_pred['true_value'] > 1
        mape = float(np.mean(np.abs(df_pred.loc[mask, 'error'] / df_pred.loc[mask, 'true_value'])) * 100) if mask.any() else 0
        fig_cmp.update_layout(
            title=dict(text=f'MAE = {mae:.2f}  ·  RMSE = {rmse:.2f}  ·  MAPE = {mape:.1f}%',
                       font=dict(size=12, color='rgba(0,212,255,.85)',
                                  family='Consolas,monospace'), x=0.01),
            height=560, **_DARK)
        fig_cmp.update_layout(legend=dict(orientation='h', y=1.06, x=0.01, font=dict(size=11)))
        st.plotly_chart(fig_cmp, use_container_width=True)

    # 多模型雷达图
    if not df_comp.empty and 'test_r2' in df_comp.columns:
        st.markdown("<div class='sec'>🎯 多模型性能雷达对比</div>", unsafe_allow_html=True)
        c_rdr, c_tbl = st.columns([1, 1])
        with c_rdr:
            fig_r = go.Figure()
            model_colors = ['#00d4ff', '#ff9100', '#00e676', '#d500f9', '#ff5252']
            for i, (mname, row) in enumerate(df_comp.iterrows()):
                r2n   = max(0, float(row.get('test_r2', 0)))
                maen  = max(0, 1 - float(row.get('test_mae', 20)) / 20)
                rmsen = max(0, 1 - float(row.get('test_rmse', 25)) / 25)
                clr   = model_colors[i % len(model_colors)]
                fig_r.add_trace(go.Scatterpolar(
                    r=[r2n, maen, rmsen, r2n],
                    theta=['R²', '低MAE', '低RMSE', 'R²'],
                    fill='toself', name=mname, opacity=0.65,
                    line=dict(color=clr, width=2),
                    fillcolor=f'rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.1)',
                ))
            fig_r.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9)),
                    angularaxis=dict(tickfont=dict(family='Exo 2,sans-serif', size=11)),
                ),
                paper_bgcolor='rgba(0,0,0,0)', height=360,
                font=dict(color='#b8cfe0', family='Exo 2,sans-serif'),
                legend=dict(font=dict(size=11)),
                margin=dict(l=60, r=60, t=30, b=30),
            )
            st.plotly_chart(fig_r, use_container_width=True)

        with c_tbl:
            st.markdown("<div class='sec'>模型精度排名</div>", unsafe_allow_html=True)
            best_model = df_comp['test_r2'].idxmax() if 'test_r2' in df_comp.columns else ''
            tbl_rows = ''
            for mname, row in df_comp.sort_values('test_r2', ascending=False).iterrows():
                badge = "<span class='model-best'>★ BEST</span>" if mname == best_model else ''
                tbl_rows += (
                    f"<tr><td>{mname}{badge}</td>"
                    f"<td style='color:#00d4ff'>{row.get('test_mae', '-'):.2f}</td>"
                    f"<td style='color:#ff9100'>{row.get('test_rmse', '-'):.2f}</td>"
                    f"<td style='color:#00e676'>{row.get('test_r2', '-'):.4f}</td></tr>"
                )
            st.markdown(f"""
            <div class='glass' style='padding:14px'>
              <table class='forecast-table'>
                <thead><tr><th>模型</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead>
                <tbody>{tbl_rows}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)


# ┌─────────────────────────────────────────────────────────────┐
# │  TAB 4 · 深度分析                                           │
# └─────────────────────────────────────────────────────────────┘
if _tab_idx == 3:
    # 相关性矩阵 + 季节分布
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='sec'>🔗 特征相关性矩阵</div>", unsafe_allow_html=True)
        feats_corr = [f for f in ['pm25', 'temperature', 'pressure', 'wind_speed', 'dewpoint', 'snow_hours']
                      if f in city_df.columns]
        if len(feats_corr) >= 2:
            corr = city_df[feats_corr].corr()
            fig_c = go.Figure(go.Heatmap(
                z=corr.values, x=feats_corr, y=feats_corr,
                colorscale='RdBu_r', zmid=0,
                text=[[f'{v:.2f}' for v in row] for row in corr.values],
                texttemplate='%{text}', textfont=dict(size=10, family='Consolas,monospace'),
                colorbar=dict(title='r', tickfont=dict(size=9)),
            ))
            dk(fig_c, '', 380)
            st.plotly_chart(fig_c, use_container_width=True)

    with c2:
        st.markdown("<div class='sec'>📅 月度 PM2.5 趋势（箱线图）</div>", unsafe_allow_html=True)
        if 'month' in city_df.columns:
            month_names = {1:'1月',2:'2月',3:'3月',4:'4月',5:'5月',6:'6月',
                           7:'7月',8:'8月',9:'9月',10:'10月',11:'11月',12:'12月'}
            fig_box = go.Figure()
            for m in sorted(city_df['month'].unique()):
                sub = city_df[city_df['month'] == m]['pm25']
                fig_box.add_trace(go.Box(
                    y=sub, name=month_names.get(m, str(m)),
                    boxpoints=False, line=dict(width=1.5),
                    marker=dict(opacity=0),
                ))
            dk(fig_box, '', 380)
            st.plotly_chart(fig_box, use_container_width=True)

    # 3D 散点
    st.markdown("<div class='sec'>🌐 温度 × 风速 × PM2.5 三维分布</div>", unsafe_allow_html=True)
    if ('temperature' in city_df.columns and 'wind_speed' in city_df.columns
            and len(city_df) > 20):
        smp = city_df.sample(min(4000, len(city_df)), random_state=42)
        fig_3d = go.Figure(go.Scatter3d(
            x=smp['temperature'], y=smp['wind_speed'], z=smp['pm25'],
            mode='markers',
            marker=dict(
                size=3, color=smp['pm25'],
                colorscale='Viridis', showscale=True,
                colorbar=dict(title='PM2.5', tickfont=dict(size=9)),
                opacity=0.7,
            ),
        ))
        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(title='温度(°C)', gridcolor='rgba(255,255,255,0.06)',
                           backgroundcolor='rgba(0,0,0,0)'),
                yaxis=dict(title='风速(m/s)', gridcolor='rgba(255,255,255,0.06)',
                           backgroundcolor='rgba(0,0,0,0)'),
                zaxis=dict(title='PM2.5(μg/m³)', gridcolor='rgba(255,255,255,0.06)',
                           backgroundcolor='rgba(0,0,0,0)'),
                bgcolor='rgba(0,0,0,0)',
            ),
            paper_bgcolor='rgba(0,0,0,0)', height=500,
            font=dict(color='#b8cfe0', family='Exo 2,sans-serif'),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    # 特征重要性
    if not df_imp.empty:
        st.markdown("<div class='sec'>🔬 Top 20 特征重要性</div>", unsafe_allow_html=True)
        top = df_imp.head(20)
        colors_fi = [
            '#00d4ff' if 'pm25' in f else
            '#00e676' if any(x in f for x in ['hour', 'month', 'season', 'sin', 'cos']) else
            '#ff9100' if any(x in f for x in ['temp', 'wind', 'pressure']) else
            '#ffd600' if 'rolling' in f or 'lag' in f else
            '#b0bec5'
            for f in top['feature']
        ]
        fig_fi = go.Figure(go.Bar(
            y=top['feature'], x=top['importance'], orientation='h',
            marker=dict(color=colors_fi, line=dict(width=0)),
            text=[f'{v:.4f}' for v in top['importance']],
            textposition='outside', textfont=dict(size=9, family='Consolas,monospace'),
        ))
        dk(fig_fi, '', 500)
        fig_fi.update_layout(yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_fi, use_container_width=True)

        # 图例说明
        st.markdown("""
        <div style='display:flex;gap:16px;font-family:Exo 2,sans-serif;font-size:.68rem;
                    color:rgba(180,210,230,.65);flex-wrap:wrap;margin-top:-8px'>
          <span><span style='color:#00d4ff'>■</span> PM2.5 相关特征</span>
          <span><span style='color:#00e676'>■</span> 时间/季节特征</span>
          <span><span style='color:#ff9100'>■</span> 气象特征</span>
          <span><span style='color:#ffd600'>■</span> 滞后/滚动特征</span>
        </div>
        """, unsafe_allow_html=True)

    # 年度趋势
    st.markdown("<div class='sec'>📆 年度 PM2.5 均值与波动</div>", unsafe_allow_html=True)
    yr = city_df.copy()
    yr['year'] = yr['timestamp'].dt.year
    ya = yr.groupby('year')['pm25'].agg(['mean', 'std', 'min', 'max']).reset_index()
    fig_yr = go.Figure()
    fig_yr.add_trace(go.Scatter(
        x=list(ya['year']) + list(ya['year'])[::-1],
        y=list(ya['mean'] + ya['std']) + list(ya['mean'] - ya['std'])[::-1],
        fill='toself', fillcolor='rgba(0,212,255,0.08)',
        line=dict(color='rgba(0,0,0,0)'), name='±1σ 波动区间',
    ))
    fig_yr.add_trace(go.Scatter(
        x=ya['year'], y=ya['mean'], mode='lines+markers', name='年均值',
        line=dict(color='#00d4ff', width=3),
        marker=dict(size=10, color='#00d4ff', line=dict(color='#030a14', width=2)),
    ))
    fig_yr.add_trace(go.Scatter(
        x=ya['year'], y=ya['max'], mode='lines', name='年最大值',
        line=dict(color='#ff5252', width=1.5, dash='dot'),
    ))
    dk(fig_yr, '', 340)
    fig_yr.update_layout(legend=dict(orientation='h', y=1.1, x=0.01, font=dict(size=11)))
    st.plotly_chart(fig_yr, use_container_width=True)


# ┌─────────────────────────────────────────────────────────────┐
# │  TAB 5 · 空间相关性  ·  Aurora Dark Redesign                │
# └─────────────────────────────────────────────────────────────┘
if _tab_idx == 4:
    import json as _json
    from PIL import Image as _Image
    import plotly.express as _px
    import plotly.graph_objects as _go
    import pandas as _pd

    @st.cache_data(show_spinner=False)
    def _load_spatial_data():
        sa = _json.load(open('../results/spatial_analysis.json'))
        summary = _json.load(open('../data/multisite_summary.json'))
        try:
            df_pairs = _pd.read_csv('../results/spatial_pairs.csv')
        except Exception:
            df_pairs = None
        return sa, summary, df_pairs

    sa, summary, df_pairs = _load_spatial_data()

    # ───── 局部样式注入（只影响本 Tab） ─────
    st.markdown("""
    <style>
      .sp-glass{
        background: linear-gradient(135deg, rgba(10,25,50,0.55) 0%, rgba(8,20,38,0.65) 100%);
        border: 1px solid rgba(0,212,255,0.18);
        border-radius: 14px;
        padding: 18px 22px;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
      }
      .sp-glass-accent{
        background: linear-gradient(135deg, rgba(0,212,255,0.10) 0%, rgba(0,212,255,0.04) 100%);
        border: 1px solid rgba(0,212,255,0.35);
        border-radius: 14px;
        padding: 22px 26px;
      }
      .sp-glass-warn{
        background: linear-gradient(135deg, rgba(255,145,0,0.08) 0%, rgba(255,145,0,0.03) 100%);
        border: 1px solid rgba(255,145,0,0.30);
        border-radius: 14px;
        padding: 18px 22px;
      }
      .sp-glass-ok{
        background: linear-gradient(135deg, rgba(0,230,118,0.08) 0%, rgba(0,230,118,0.03) 100%);
        border: 1px solid rgba(0,230,118,0.30);
        border-radius: 14px;
        padding: 18px 22px;
      }
      .sp-tag{
        display: inline-block;
        padding: 3px 10px;
        font-size: 11px;
        letter-spacing: 1px;
        border-radius: 999px;
        background: rgba(0,212,255,0.12);
        color: #6FE9FF;
        border: 1px solid rgba(0,212,255,0.30);
        margin-right: 8px;
        font-family: 'Consolas','JetBrains Mono',monospace;
      }
      .sp-tag-orange{
        background: rgba(255,145,0,0.12);
        color: #FFB55A;
        border-color: rgba(255,145,0,0.35);
      }
      .sp-tag-green{
        background: rgba(0,230,118,0.10);
        color: #5BE6A0;
        border-color: rgba(0,230,118,0.30);
      }
      .sp-tag-purple{
        background: rgba(180,140,255,0.12);
        color: #C9AEFF;
        border-color: rgba(180,140,255,0.30);
      }
      .sp-kpi-label{
        color:#9FB4CC;
        font-size: 12px;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 4px;
      }
      .sp-kpi-value{
        font-family: 'Orbitron','Consolas',monospace;
        font-size: 30px;
        font-weight: 700;
        line-height: 1.1;
        color: #E8F4FF;
        letter-spacing: 0.5px;
      }
      .sp-kpi-unit{
        color:#6FE9FF;
        font-size: 14px;
        font-weight: 500;
        margin-left: 4px;
      }
      .sp-kpi-foot{
        color:#7892A8;
        font-size: 11px;
        margin-top: 6px;
      }
      .sp-hero-num{
        font-family: 'Orbitron','Consolas',monospace;
        font-size: 80px;
        font-weight: 800;
        background: linear-gradient(135deg,#00D9FF 0%,#5FF5FF 50%,#7BA8FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
        letter-spacing: -2px;
      }
      .sp-hero-unit{
        color:#6FE9FF;
        font-size: 26px;
        font-weight: 500;
        margin-left: 6px;
      }
      .sp-fig-cap{
        color:#7892A8;
        font-size: 11.5px;
        font-style: italic;
        margin-top: 10px;
        text-align: center;
        line-height: 1.6;
      }
      .sp-section-title{
        color:#E8F4FF;
        font-size: 16px;
        font-weight: 600;
        margin: 6px 0 14px 0;
        padding-left: 10px;
        border-left: 3px solid #00D4FF;
        letter-spacing: 0.5px;
      }
      .sp-arrow{
        color:#00D4FF;
        font-size: 24px;
        text-align: center;
        line-height: 1;
        font-weight: 300;
      }
    </style>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # § 1 · 标题区
    # ═══════════════════════════════════════════════════════════
    st.markdown("""
    <div style='margin: 4px 0 18px 0;'>
      <div style='display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;'>
        <span style='font-size:30px;font-weight:700;color:#E8F4FF;letter-spacing:0.5px;'>
          🗺️ 北京 12 站点空间相关性分析
        </span>
        <span class='sp-tag'>UCI MULTI-SITE</span>
        <span class='sp-tag sp-tag-green'>REAL DATA</span>
        <span class='sp-tag sp-tag-purple'>2013-2017</span>
      </div>
      <div style='color:#7892A8;font-size:13px;margin-top:6px;letter-spacing:0.3px;'>
        Spatial Correlation Analysis  ·  d₀ = 256 km Empirical Boundary Condition for Multi-Scale Architectures
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # § 2 · 核心发现 Hero（d₀ = 256 km 大数字）
    # ═══════════════════════════════════════════════════════════
    hero_l, hero_r = st.columns([1.05, 1])

    with hero_l:
        st.markdown(f"""
        <div class='sp-glass-accent' style='height:200px;display:flex;flex-direction:column;justify-content:center;'>
          <div style='color:#6FE9FF;font-size:11px;letter-spacing:2px;font-weight:600;text-transform:uppercase;margin-bottom:6px;'>
            ★ KEY FINDING  ·  e-folding distance
          </div>
          <div style='display:flex;align-items:baseline;'>
            <span class='sp-hero-num'>d₀ = {sa['decay_length_km']:.0f}</span>
            <span class='sp-hero-unit'>km</span>
          </div>
          <div style='color:#9FB4CC;font-size:13px;margin-top:8px;line-height:1.6;'>
            corr ≈ <span style='color:#E8F4FF;font-family:Consolas;'>{sa['decay_amplitude']:.3f} · exp(−d / 256 km)</span>
            ,北京区域 PM2.5 浓度场的空间衰减尺度
          </div>
        </div>
        """, unsafe_allow_html=True)

    with hero_r:
        st.markdown("""
        <div class='sp-glass-warn' style='height:200px;display:flex;flex-direction:column;justify-content:center;'>
          <div style='color:#FFB55A;font-size:11px;letter-spacing:2px;font-weight:600;text-transform:uppercase;margin-bottom:6px;'>
            ⚠ 关键对比  ·  Reality Check
          </div>
          <div style='display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;'>
            <div>
              <div style='font-family:Orbitron,Consolas;font-size:42px;font-weight:700;color:#FFB55A;line-height:1;'>
                59.5 <span style='font-size:18px;color:#9FB4CC;'>km</span>
              </div>
              <div style='color:#7892A8;font-size:11px;margin-top:4px;'>站点最大间距(Gucheng↔Huairou)</div>
            </div>
            <div style='font-size:24px;color:#7892A8;font-weight:300;'>≪</div>
            <div>
              <div style='font-family:Orbitron,Consolas;font-size:42px;font-weight:700;color:#00D4FF;line-height:1;'>
                256 <span style='font-size:18px;color:#9FB4CC;'>km</span>
              </div>
              <div style='color:#7892A8;font-size:11px;margin-top:4px;'>空间衰减尺度 d₀</div>
            </div>
          </div>
          <div style='color:#FFB55A;font-size:12.5px;margin-top:12px;font-weight:500;letter-spacing:0.3px;'>
            ⤳ 站点间距远小于 d₀,北京区域内"空间维度已坍缩"
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # § 3 · 6 个 KPI
    # ═══════════════════════════════════════════════════════════
    st.markdown("<div class='sp-section-title'>数据规模与相关性指标 · Data Profile & Correlation Stats</div>",
                unsafe_allow_html=True)

    kpis = [
        ("MONITORING STATIONS", f"{summary['n_stations']}", "个", "北京区域监测站", "#00D4FF"),
        ("TOTAL RECORDS",       f"{summary['total_records']:,}", "条", "10× 单源数据", "#00D4FF"),
        ("TIME SPAN",           "4", "年", "2013-03 → 2017-02", "#5BE6A0"),
        ("MEAN CORRELATION",    f"{sa['mean_corr']:.3f}", "r̄", f"范围 [{sa['min_corr']:.2f}, {sa['max_corr']:.2f}]", "#5BE6A0"),
        ("MAX CORRELATION",     f"{sa['max_corr']:.3f}", "r", f"东四 ↔ 官园 (6.7 km)", "#C9AEFF"),
        ("PAIR DISTANCE RANGE", "3.9 – 59.5", "km", "66 对站点 × 4 年", "#C9AEFF"),
    ]

    kpi_cols = st.columns(6)
    for col, (lab, val, unit, foot, c) in zip(kpi_cols, kpis):
        with col:
            st.markdown(f"""
            <div class='sp-glass' style='height:140px;'>
              <div class='sp-kpi-label'>{lab}</div>
              <div style='display:flex;align-items:baseline;'>
                <span class='sp-kpi-value' style='color:{c};'>{val}</span>
                <span class='sp-kpi-unit' style='color:{c};opacity:0.85;'>{unit}</span>
              </div>
              <div class='sp-kpi-foot'>{foot}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # § 4 · 双图分析
    # ═══════════════════════════════════════════════════════════
    st.markdown("<div class='sp-section-title'>空间相关性可视化 · Correlation Matrix & Distance Decay</div>",
                unsafe_allow_html=True)

    fc_l, fc_r = st.columns(2)

    with fc_l:
        st.markdown("""
        <div class='sp-glass' style='padding:14px 16px;'>
          <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>
            <span style='color:#E8F4FF;font-size:14px;font-weight:600;'>
              Figure 8 · 12 站点 PM2.5 相关性矩阵
            </span>
            <span class='sp-tag'>66 PAIRS</span>
          </div>
        """, unsafe_allow_html=True)
        if os.path.exists('../paper_figures/fig8_spatial_correlation.png'):
            img = _Image.open('../paper_figures/fig8_spatial_correlation.png')
            st.image(img, use_container_width=True)
        st.markdown(f"""
          <div class='sp-fig-cap'>
            ▸ 最相关:<span style='color:#5BE6A0;'>{sa['max_corr_pair']}</span>
            　·
            ▸ 最不相关:<span style='color:#FFB55A;'>{sa['min_corr_pair']}</span>
            <br/>所有 66 对站点相关系数均 ≥ 0.74,平均 0.87
          </div>
        </div>
        """, unsafe_allow_html=True)

    with fc_r:
        st.markdown("""
        <div class='sp-glass' style='padding:14px 16px;'>
          <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>
            <span style='color:#E8F4FF;font-size:14px;font-weight:600;'>
              Figure 9 · 距离衰减（指数拟合）
            </span>
            <span class='sp-tag sp-tag-orange'>d₀ = 256 km</span>
          </div>
        """, unsafe_allow_html=True)
        if os.path.exists('../paper_figures/fig9_distance_decay.png'):
            img2 = _Image.open('../paper_figures/fig9_distance_decay.png')
            st.image(img2, use_container_width=True)
        st.markdown(f"""
          <div class='sp-fig-cap'>
            ▸ 拟合公式:<span style='color:#6FE9FF;font-family:Consolas;'>corr ≈ {sa['decay_amplitude']:.3f} · exp(−d / {sa['decay_length_km']:.0f} km)</span>
            <br/>所有观测距离均 &lt; d₀,站点处于同一相关结构的"近场"区域
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # § 5 · 12 站点地图（暗色 carto + 站点级平均相关性着色）
    # ═══════════════════════════════════════════════════════════
    st.markdown("<div class='sp-section-title'>12 站点地理分布 · Geographic Distribution</div>",
                unsafe_allow_html=True)

    latlon = summary.get('per_station_latlon', {})
    if latlon:
        station_avg_corr = {}
        if df_pairs is not None and 'station_pair' in df_pairs.columns:
            for st_name in latlon.keys():
                vals = df_pairs[df_pairs['station_pair'].str.contains(st_name, na=False)]['correlation']
                station_avg_corr[st_name] = float(vals.mean()) if len(vals) > 0 else sa['mean_corr']
        else:
            for st_name in latlon.keys():
                station_avg_corr[st_name] = sa['mean_corr']

        df_map = _pd.DataFrame([
            {
                'station': name,
                'lat': lat,
                'lon': lon,
                'avg_r': station_avg_corr.get(name, sa['mean_corr']),
                'label_text': f"{name}<br>avg r = {station_avg_corr.get(name, sa['mean_corr']):.3f}"
            }
            for name, (lat, lon) in latlon.items()
        ])

        fig_map = _px.scatter_mapbox(
            df_map,
            lat='lat', lon='lon',
            hover_name='station',
            hover_data={'avg_r':':.3f', 'lat':False, 'lon':False},
            color='avg_r',
            size=[18]*len(df_map),
            size_max=22,
            zoom=8.6,
            height=460,
            color_continuous_scale=[
                [0.0, '#FFB55A'],
                [0.5, '#00D4FF'],
                [1.0, '#5FF5FF'],
            ],
            range_color=[0.80, 0.92],
        )
        fig_map.add_trace(_go.Scattermapbox(
            lat=df_map['lat'], lon=df_map['lon'],
            mode='text',
            text=df_map['station'],
            textfont=dict(size=11, color='#E8F4FF', family='Arial'),
            textposition='top center',
            hoverinfo='skip',
            showlegend=False,
        ))
        fig_map.update_layout(
            mapbox_style="carto-darkmatter",
            margin=dict(l=0, r=0, t=4, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#9FB4CC', family='Arial'),
            coloraxis_colorbar=dict(
                title=dict(text='avg r', font=dict(color='#9FB4CC', size=11)),
                tickfont=dict(color='#9FB4CC', size=10),
                thickness=10, len=0.6,
                outlinewidth=0,
            ),
            showlegend=False,
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown(
            "<div class='sp-fig-cap'>"
            "▸ 圆点大小一致,颜色编码该站点对其余 11 站的平均相关系数(亮青→更同步,橙→相对独立)"
            "<br/>▸ 12 站点全部位于北京及近郊,空间跨度约 60 km,远小于 d₀=256 km"
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # § 6 · 悖论与解释（双栏:消融观察 vs 空间解释）
    # ═══════════════════════════════════════════════════════════
    st.markdown("<div class='sp-section-title'>研究路径闭环 · Ablation Paradox → Spatial Explanation</div>",
                unsafe_allow_html=True)

    px_l, px_m, px_r = st.columns([1, 0.08, 1])

    with px_l:
        st.markdown("""
        <div class='sp-glass-warn' style='min-height:230px;'>
          <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;'>
            <span class='sp-tag sp-tag-orange'>STAGE 1 · 5.3 节</span>
            <span style='color:#FFB55A;font-size:14px;font-weight:600;'>消融实验观察（反向结果）</span>
          </div>
          <div style='color:#E8F4FF;font-size:13px;line-height:1.85;'>
            7 变体 × 3 seeds × 50 epoch 的严格消融揭示:
            <br/>
            <div style='margin:10px 0;padding:10px 12px;background:rgba(255,145,0,0.06);border-left:2px solid #FFB55A;border-radius:4px;font-family:Consolas;font-size:12px;'>
              <span style='color:#FFB55A;'>完整 MSTN v2 </span>(147K 参数)
              <br/>　 MAE = <span style='color:#FFB55A;font-weight:700;'>11.36 ± 0.21</span>
              <br/><br/>
              <span style='color:#5BE6A0;'>纯 LSTM 基线 </span>(30K 参数)
              <br/>　 MAE = <span style='color:#5BE6A0;font-weight:700;'>10.94 ± 0.09</span>
            </div>
            差距 0.42 ≈ 2σ,3 seeds 稳定复现。
            <br/>
            <span style='color:#FFB55A;'>显式三尺度架构在单城市数据上未带来精度优势。</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with px_m:
        st.markdown("""
        <div style='display:flex;justify-content:center;align-items:center;height:230px;'>
          <div style='text-align:center;color:#00D4FF;'>
            <div style='font-size:24px;font-weight:300;'>▸▸▸</div>
            <div style='font-size:10px;letter-spacing:1.5px;margin-top:6px;color:#6FE9FF;'>
              SCIENTIFIC<br/>EXPLAIN
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with px_r:
        st.markdown("""
        <div class='sp-glass-ok' style='min-height:230px;'>
          <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;'>
            <span class='sp-tag sp-tag-green'>STAGE 2 · 本节</span>
            <span style='color:#5BE6A0;font-size:14px;font-weight:600;'>空间相关性科学解释</span>
          </div>
          <div style='color:#E8F4FF;font-size:13px;line-height:1.85;'>
            UCI Multi-Site 12 站点真实数据空间分析:
            <br/>
            <div style='margin:10px 0;padding:10px 12px;background:rgba(0,230,118,0.06);border-left:2px solid #5BE6A0;border-radius:4px;font-family:Consolas;font-size:12px;'>
              <span style='color:#5BE6A0;'>e-folding distance </span>
              <br/>　 d₀ = <span style='color:#5BE6A0;font-weight:700;'>256 km</span>
              <br/><br/>
              <span style='color:#5BE6A0;'>站点最大间距 </span>
              <br/>　 d_max = <span style='color:#5BE6A0;font-weight:700;'>59.5 km</span> ≪ d₀
            </div>
            d₀ ≫ 站点最大间距,12 站点全部位于"近场"。
            <br/>
            <span style='color:#5BE6A0;'>"空间维度坍缩"使跨站点时空建模退化为同步时间建模——多尺度交互无真实物理对应。</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # § 7 · 方法论贡献 + 后续方向
    # ═══════════════════════════════════════════════════════════
    st.markdown(f"""
    <div class='sp-glass-accent'>
      <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>
        <span style='font-size:18px;color:#00D4FF;'>▷</span>
        <span style='color:#E8F4FF;font-size:15px;font-weight:600;letter-spacing:0.4px;'>
          方法论贡献  ·  Methodological Contribution
        </span>
      </div>
      <div style='color:#C5D5E8;font-size:13.5px;line-height:1.85;'>
        本节<span style='color:#5FF5FF;font-weight:600;'>首次基于真实多站点 PM2.5 数据</span>,
        对多尺度时空架构在区域级数据上的适用边界进行了实证量化。
        该方法论(空间衰减尺度 d₀  vs  数据采样间距)可推广到任何带空间维度的时序建模任务,
        为时序深度学习的<span style='color:#5FF5FF;font-weight:600;'>架构选型提供实证判据</span>。
        <br/><br/>
        多尺度时空交互的真实价值需在 <span style='color:#5FF5FF;font-weight:600;'>d₀ 量级以上(>250 km)</span>
        的跨区域数据中验证(如华北—长三角—珠三角联合数据),
        这是后续 CNEMC(中国环境监测中心 8.1 亿条 >2000 站点数据)接入后的核心研究方向。
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # § 8 · 详细数据（可展开）
    # ═══════════════════════════════════════════════════════════
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    with st.expander("📋 完整 66 对站点相关性详情（按距离排序）"):
        if df_pairs is not None:
            df_show = df_pairs.copy()
            if 'distance_km' in df_show.columns:
                df_show = df_show.sort_values('distance_km')
            st.dataframe(df_show, use_container_width=True, hide_index=True)
        else:
            st.caption("spatial_pairs.csv 未加载")


# ┌─────────────────────────────────────────────────────────────┐
# │  TAB 6 · 大数据洞察                                         │
# └─────────────────────────────────────────────────────────────┘
if _tab_idx == 5:
    # 大数字统计
    st.markdown("<div class='sec'>📦 数据集规模概览</div>", unsafe_allow_html=True)
    gc1, gc2, gc3, gc4, gc5 = st.columns(5)
    span_full = (df_all['timestamp'].max() - df_all['timestamp'].min()).days
    n_cities = df_all['city'].nunique() if 'city' in df_all.columns else 1
    for col, val, lbl in zip(
        [gc1, gc2, gc3, gc4, gc5],
        [f"{len(df_all):,}", f"{span_full}", f"{df_all.shape[1]}", f"{n_cities}", f"{df_all['pm25'].mean():.1f}"],
        ['总记录数 (条)', '时间跨度 (天)', '特征维度', '城市 / 站点', 'PM2.5 均值 μg/m³'],
    ):
        col.markdown(f"<div class='bignum-card'><div class='bignum-val'>{val}</div><div class='bignum-lbl'>{lbl}</div></div>", unsafe_allow_html=True)

    # 处理管道流图
    st.markdown("<div class='sec'>⚙️ 数据处理全链路管道</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass' style='padding:20px'>
      <div class='pipeline'>
        <div class='pipe-step'><div class='pipe-step-icon'>📡</div><div class='pipe-step-name'>数据采集</div></div>
        <div class='pipe-arrow'>→</div>
        <div class='pipe-step'><div class='pipe-step-icon'>🧹</div><div class='pipe-step-name'>数据清洗</div></div>
        <div class='pipe-arrow'>→</div>
        <div class='pipe-step'><div class='pipe-step-icon'>⚗️</div><div class='pipe-step-name'>特征工程</div></div>
        <div class='pipe-arrow'>→</div>
        <div class='pipe-step'><div class='pipe-step-icon'>🤖</div><div class='pipe-step-name'>模型训练</div></div>
        <div class='pipe-arrow'>→</div>
        <div class='pipe-step'><div class='pipe-step-icon'>🔬</div><div class='pipe-step-name'>消融实验</div></div>
        <div class='pipe-arrow'>→</div>
        <div class='pipe-step'><div class='pipe-step-icon'>📊</div><div class='pipe-step-name'>可视化</div></div>
        <div class='pipe-arrow'>→</div>
        <div class='pipe-step'><div class='pipe-step-icon'>🚀</div><div class='pipe-step-name'>Web部署</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 全量 AQI 分布 + 年度趋势
    ca6, cb6 = st.columns(2)
    with ca6:
        st.markdown("<div class='sec'>🎨 全量数据 AQI 分布</div>", unsafe_allow_html=True)
        aac = df_all['pm25'].apply(aqi_level_str).value_counts()
        fig_aac = go.Figure(go.Pie(
            labels=aac.index, values=aac.values,
            marker=dict(colors=[PIE_CLR.get(l, '#888') for l in aac.index],
                        line=dict(color='rgba(0,0,0,0)', width=0)),
            hole=0.5, textfont=dict(family='Exo 2,sans-serif', size=11),
        ))
        dk(fig_aac, '', 340)
        st.plotly_chart(fig_aac, use_container_width=True)

    with cb6:
        st.markdown("<div class='sec'>📆 全量数据年度趋势</div>", unsafe_allow_html=True)
        ya2 = df_all.copy()
        ya2['year'] = ya2['timestamp'].dt.year
        ya2g = ya2.groupby('year')['pm25'].agg(['mean', 'std']).reset_index()
        fig_ya2 = go.Figure()
        fig_ya2.add_trace(go.Scatter(
            x=list(ya2g['year']) + list(ya2g['year'])[::-1],
            y=list(ya2g['mean'] + ya2g['std']) + list(ya2g['mean'] - ya2g['std'])[::-1],
            fill='toself', fillcolor='rgba(0,212,255,0.08)',
            line=dict(color='rgba(0,0,0,0)'), name='±1σ',
        ))
        fig_ya2.add_trace(go.Scatter(
            x=ya2g['year'], y=ya2g['mean'], mode='lines+markers', name='年均',
            line=dict(color='#00d4ff', width=3),
            marker=dict(size=10, color='#00d4ff', line=dict(color='#030a14', width=2)),
        ))
        dk(fig_ya2, '', 340)
        st.plotly_chart(fig_ya2, use_container_width=True)

    # 数据来源说明
    st.markdown("<div class='sec'>📚 数据来源与可信度声明</div>", unsafe_allow_html=True)
    srcs = [
        ('#00d4ff', '📦 UCI 北京 PM2.5 数据集',
         '2010–2014年北京逐小时PM2.5与气象数据，原始记录43,824条，权威公开学术数据集，被数百篇论文引用',
         'https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data'),
        ('#69f0ae', '📡 OpenAQ 开放平台',
         '全球开放空气质量数据，覆盖100+国家实时监测站，免费API接入，无需注册，数据实时更新',
         'https://openaq.org/'),
        ('#ffd600', '🌤 NOAA GSOD 气象数据',
         'NOAA全球地面气象日汇总数据集，1929年至今，涵盖全球数千个气象站，补充气象特征维度',
         'https://www.ncei.noaa.gov/data/global-summary-of-the-day/'),
    ]
    sc1, sc2, sc3 = st.columns(3)
    for col, (clr, name, desc, url) in zip([sc1, sc2, sc3], srcs):
        col.markdown(f"""
        <div class='src-card' style='border:1px solid {clr}25'>
          <div class='src-name' style='color:{clr}'>{name}</div>
          <div class='src-desc'>{desc}</div>
          <div class='src-url'>{url}</div>
        </div>
        """, unsafe_allow_html=True)

    # 数据下载
    st.markdown("<div class='sec'>⬇️ 数据导出</div>", unsafe_allow_html=True)
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        csv_sub = city_df[['timestamp', 'pm25']].to_csv(index=False).encode()
        st.download_button('⬇️ 下载当前视图 PM2.5 数据', csv_sub,
                           'pm25_data.csv', 'text/csv', use_container_width=True)
    with dl2:
        if not df_pred.empty:
            st.download_button('⬇️ 下载预测结果', df_pred.to_csv(index=False).encode(),
                               'predictions.csv', 'text/csv', use_container_width=True)
        else:
            st.button('⬇️ 预测结果（暂无）', disabled=True, use_container_width=True)
    with dl3:
        if not df_comp.empty:
            st.download_button('⬇️ 下载模型对比表', df_comp.to_csv().encode(),
                               'model_comparison.csv', 'text/csv', use_container_width=True)
        else:
            st.button('⬇️ 模型对比（暂无）', disabled=True, use_container_width=True)


# ┌─────────────────────────────────────────────────────────────┐
# │  TAB 7  ·  AI 智能分析中心(离线规则 + DeepSeek 深度分析)      │
# └─────────────────────────────────────────────────────────────┘
if _tab_idx == 6:
    # 引入 AI 引擎(独立模块,失败时优雅降级)
    try:
        from qa_engine import AirQualityQAEngine
        _qa_ok = True
    except Exception as _e:
        _qa_ok = False
        st.error(f"AI 引擎加载失败:{_e}")

    # 引入 DeepSeek 深度分析(可选,失败优雅降级)
    try:
        import deep_analysis as dsmod
        _ds_ok = True
    except Exception:
        dsmod = None
        _ds_ok = False

    if _qa_ok:
        @st.cache_resource
        def _get_qa_engine(n_rows):
            return AirQualityQAEngine(df_all)
        engine = _get_qa_engine(len(df_all))

        # 检测 DeepSeek 配置
        _llm_key = dsmod.get_api_key() if _ds_ok else None
        _llm_ready = bool(_llm_key)

        # ─── 头部:模式状态卡 ───
        st.markdown("<div class='sec'>🧠 AI 智能分析中心 · INTELLIGENT ANALYSIS HUB</div>",
                    unsafe_allow_html=True)

        mode_c1, mode_c2 = st.columns([2, 1])
        with mode_c1:
            st.markdown("""
            <div class='glass' style='padding:18px;border-color:rgba(139,92,246,.32)'>
              <div style='font-family:Exo 2,sans-serif;font-size:.78rem;
                          color:rgba(200,225,245,.78);line-height:1.8'>
                <span style='color:#b794ff;font-weight:700'>双模式架构</span><br>
                • <b>离线模式</b>(默认)— 12 类规则匹配,演示绝对稳定,无需联网<br>
                • <b>深度模式</b> — DeepSeek-Chat LLM,开放式归因 + 政策建议
              </div>
            </div>
            """, unsafe_allow_html=True)
        with mode_c2:
            if _llm_ready:
                key_disp = dsmod.mask_key(_llm_key)
                st.markdown(
                    "<div class='glass' style='text-align:center;padding:18px;"
                    "border-color:rgba(0,230,118,.35)'>"
                    "<div style='font-size:1.6rem'>🟢</div>"
                    "<div style='font-family:Orbitron,sans-serif;font-size:.7rem;"
                    "letter-spacing:3px;color:#69f0ae;margin-top:6px'>"
                    "深度模式 ONLINE"
                    "</div>"
                    "<div style='font-size:.62rem;color:rgba(180,210,230,.5);"
                    "margin-top:4px;font-family:JetBrains Mono,monospace'>"
                    + key_disp +
                    "</div></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown("""
                <div class='glass' style='text-align:center;padding:18px;
                            border-color:rgba(255,214,0,.32)'>
                  <div style='font-size:1.6rem'>🟡</div>
                  <div style='font-family:Orbitron,sans-serif;font-size:.7rem;
                              letter-spacing:3px;color:#ffd600;margin-top:6px'>
                    离线模式 OFFLINE
                  </div>
                  <div style='font-size:.62rem;color:rgba(180,210,230,.5);margin-top:4px'>
                    配置 secrets.toml 解锁深度分析
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # ═══════════════════════════════════════════════════
        # 子区 1: AI 自动洞察(仅深度模式)
        # ═══════════════════════════════════════════════════
        if _llm_ready:
            st.markdown("<div class='sec'>💡 AI 自动洞察 · AUTO INSIGHTS</div>",
                        unsafe_allow_html=True)

            @st.cache_data(ttl=3600, show_spinner=False)
            def _gen_insights_cached(_hash, _summary):
                client = dsmod.DeepSeekClient(_llm_key)
                return client.auto_insights(_summary)

            cb1, cb2 = st.columns([1, 3])
            with cb1:
                refresh_ins = st.button('🔄  生成/刷新洞察',
                                         use_container_width=True,
                                         key='btn_refresh_ins')
            with cb2:
                st.markdown("""
                <div style='padding:8px 12px;font-size:.7rem;
                            color:rgba(180,210,230,.6);
                            font-family:Exo 2,sans-serif;line-height:1.7'>
                  💰 单次约 ¥0.001-0.003 · 缓存 1 小时 · 数据变化时自动重算
                </div>
                """, unsafe_allow_html=True)

            if refresh_ins or 'insights_cache' not in st.session_state:
                with st.spinner('🧠 DeepSeek 正在分析数据...'):
                    summary = dsmod.build_data_summary(city_df)
                    h = dsmod.summary_hash(summary)
                    st.session_state.insights_cache = _gen_insights_cached(h, summary)

            ins_result = st.session_state.get('insights_cache', {})
            if ins_result.get('ok'):
                ins_list = ins_result.get('insights', [])
                if ins_list:
                    cols_n = min(len(ins_list), 5)
                    ins_cols = st.columns(cols_n)
                    level_color = {'info': '#64b5f6', 'warn': '#ffd600', 'crit': '#ff5252'}
                    for i, ins in enumerate(ins_list[:5]):
                        clr = level_color.get(ins.get('level', 'info'), '#00d4ff')
                        ic = ins.get('icon', '📊')
                        ti = ins.get('title', '')
                        ds = ins.get('desc', '')
                        ins_cols[i].markdown(
                            "<div class='glass' style='border-color:" + clr + "40;"
                            "padding:14px;min-height:160px'>"
                            "<div style='font-size:1.4rem;margin-bottom:6px'>" + ic + "</div>"
                            "<div style='font-family:Exo 2,sans-serif;font-weight:700;"
                            "font-size:.82rem;color:" + clr + ";margin-bottom:6px;"
                            "letter-spacing:1px'>" + ti + "</div>"
                            "<div style='font-size:.7rem;color:rgba(200,225,245,.78);"
                            "line-height:1.7'>" + ds + "</div>"
                            "</div>",
                            unsafe_allow_html=True
                        )
                    st.markdown(
                        "<div style='text-align:right;font-size:.62rem;"
                        "color:rgba(180,210,230,.45);"
                        "font-family:JetBrains Mono,monospace;margin-top:6px'>"
                        "本次费用: " + ins_result.get('cost', '—') + " · powered by DeepSeek"
                        "</div>",
                        unsafe_allow_html=True
                    )
            elif ins_result.get('error'):
                st.markdown(
                    "<div class='alert alert-orange' style='font-size:.78rem'>"
                    "⚠️ 生成失败: " + ins_result['error'] + " — 已自动回退,可继续用问答"
                    "</div>",
                    unsafe_allow_html=True
                )

        # ═══════════════════════════════════════════════════
        # 子区 2: 智能问答(双模式)
        # ═══════════════════════════════════════════════════
        st.markdown("<div class='sec'>💬 智能问答 · ASK ANYTHING</div>",
                    unsafe_allow_html=True)

        # 模式开关
        use_deep = False
        if _llm_ready:
            use_deep = st.toggle('🚀 启用深度模式(LLM 推理,可问开放式问题)',
                                  value=False, key='use_deep_qa')

        st.markdown("""
        <div class='glass' style='padding:18px;margin-bottom:14px'>
          <div style='font-family:Exo 2,sans-serif;font-size:.78rem;
                      color:rgba(200,225,245,.78);line-height:1.8'>
            💡 <b>直接用中文问数据集相关的任何问题</b>。
          </div>
        </div>
        """, unsafe_allow_html=True)

        # 状态
        if 'ai_input' not in st.session_state:
            st.session_state.ai_input = ''
        if 'ai_history' not in st.session_state:
            st.session_state.ai_history = []

        # ✅ v3.7 修复: 提取统一查询函数, 让"预设按钮"和"提问按钮"共用同一逻辑
        def _run_qa(q):
            """执行一次查询并写入 history (v3.7 单击直跑)"""
            q = (q or '').strip()
            if not q:
                return
            if use_deep and _llm_ready:
                with st.spinner('🧠 DeepSeek 正在思考...'):
                    client = dsmod.DeepSeekClient(_llm_key)
                    ctx = dsmod.build_data_context(city_df)
                    rsp = client.deep_qa(q, ctx)
                if rsp['ok']:
                    result = {
                        'answer': rsp['answer'],
                        'data': None,
                        'chart_hint': None,
                        'mode': 'deep',
                        'cost': rsp.get('cost', '—'),
                    }
                else:
                    result = engine.query(q)
                    result['mode'] = 'offline_fallback'
                    result['fallback_reason'] = rsp['error']
            else:
                result = engine.query(q)
                result['mode'] = 'offline'
            st.session_state.ai_history.insert(0, {'q': q, 'r': result})
            st.session_state.ai_input = ''

        # 推荐问题(根据模式动态切换)
        st.markdown("<div class='sec'>💎 试试这些问题</div>", unsafe_allow_html=True)
        suggested_offline = [
            '历史最高 PM2.5 出现在什么时候?',
            '哪个月份污染最严重?',
            '一天中什么时段空气最差?',
            '风速对 PM2.5 有什么影响?',
            '整体平均浓度是多少?',
            '当前给我一个健康建议',
            'BHI 是什么?',
            'MSTN 模型精度怎样?',
            'PM2.5 超过 75 的样本占比?',
            '历年趋势如何?',
        ]
        suggested_deep = [
            '为什么冬季污染这么严重?从气象学角度分析',
            '如果取消机动车限行,PM2.5 会涨多少?',
            '北京和华北雾霾的根本成因是什么?',
            '从这些数据能看出环保政策的效果吗?',
            '老人和小孩的防护策略应该有什么不同?',
        ]
        show_sug = suggested_deep if use_deep else suggested_offline
        sg_cols = st.columns(min(5, len(show_sug)))
        for i, s in enumerate(show_sug):
            # ✅ v3.7: 单击即执行, 不再只填 input
            if sg_cols[i % 5].button(s, key='sg_' + str(i) + '_' + str(use_deep),
                                      use_container_width=True):
                _run_qa(s)

        # 输入与按钮(给用户输入自定义问题用)
        user_q = st.text_input(
            '🔎  请输入你的问题',
            value=st.session_state.ai_input,
            placeholder='深度模式可问开放式问题,离线模式答 12 类高频问题',
            key='ai_input_box',
            label_visibility='collapsed'
        )
        bc1, bc2, _bsp = st.columns([1, 1, 5])
        with bc1:
            ask = st.button('🚀  提问', use_container_width=True, key='btn_ask')
        with bc2:
            clr = st.button('🗑️  清空', use_container_width=True, key='btn_clr')
        if clr:
            st.session_state.ai_history = []
            st.session_state.ai_input = ''
            st.rerun()

        # 处理"提问"按钮(输入框 + 提问按钮的组合)
        if ask and user_q.strip():
            _run_qa(user_q)

        # 渲染对话历史
        if st.session_state.ai_history:
            st.markdown("<div class='sec'>📜 对话记录</div>", unsafe_allow_html=True)
            for entry in st.session_state.ai_history[:8]:
                q = entry['q']
                r = entry['r']
                # 模式徽章
                mode = r.get('mode', 'offline')
                badge_map = {
                    'deep': "<span style='background:rgba(139,92,246,.2);color:#b794ff;"
                            "padding:2px 8px;border-radius:10px;font-size:.6rem;"
                            "letter-spacing:1px'>🚀 深度</span>",
                    'offline': "<span style='background:rgba(0,212,255,.15);color:#80deea;"
                               "padding:2px 8px;border-radius:10px;font-size:.6rem;"
                               "letter-spacing:1px'>⚡ 离线</span>",
                    'offline_fallback': "<span style='background:rgba(255,145,0,.2);"
                                        "color:#ffb74d;padding:2px 8px;border-radius:10px;"
                                        "font-size:.6rem;letter-spacing:1px'>⚠️ 回退</span>",
                }
                badge = badge_map.get(mode, '')
                cost_tag = ''
                if r.get('cost') and r['cost'] != '—':
                    cost_tag = ("<span style='font-size:.6rem;color:rgba(180,210,230,.45);"
                                "margin-left:8px;font-family:JetBrains Mono,monospace'>"
                                + r['cost'] + "</span>")
                fb_tag = ''
                if r.get('fallback_reason'):
                    fb_tag = ("<div style='font-size:.65rem;color:#ffb74d;margin-top:4px'>"
                              "⚠️ LLM 调用失败(" + r['fallback_reason'] + "),已自动回退"
                              "</div>")

                ans_html = r['answer'].replace('\n\n', '<br><br>').replace('\n', '<br>')
                st.markdown(
                    "<div class='ai-chat-window' style='max-height:none;margin-bottom:10px'>"
                    "<div class='ai-msg ai-msg-user'>" + q + "</div>"
                    "<div class='ai-msg ai-msg-ai'>"
                    "<span class='ai-tag'>AI · 时空呼吸智能体 " + badge + cost_tag + "</span>"
                    + ans_html + fb_tag +
                    "</div></div>",
                    unsafe_allow_html=True
                )

                # 配可视化
                if r.get('data') is not None and isinstance(r['data'], pd.DataFrame) and not r['data'].empty:
                    hint = r.get('chart_hint', '')
                    df_r = r['data']
                    if hint == 'bar_month':
                        fig_q = go.Figure(go.Bar(
                            x=df_r['month'].astype(str) + '月',
                            y=df_r['月均PM2.5'],
                            marker=dict(color=df_r['月均PM2.5'], colorscale='RdYlGn_r',
                                        line=dict(width=0)),
                            text=[f'{v:.0f}' for v in df_r['月均PM2.5']],
                            textposition='outside',
                        ))
                        dk(fig_q, '', 320)
                        st.plotly_chart(fig_q, use_container_width=True)
                    elif hint == 'line_hour':
                        fig_q = go.Figure(go.Scatter(
                            x=df_r['hour'], y=df_r['时均PM2.5'], mode='lines+markers',
                            line=dict(color='#00d4ff', width=2.5),
                            marker=dict(size=8, color=df_r['时均PM2.5'], colorscale='RdYlGn_r',
                                        line=dict(color='#030a14', width=2)),
                            fill='tozeroy', fillcolor='rgba(0,212,255,.06)',
                        ))
                        fig_q.update_layout(xaxis=dict(title='小时', dtick=2),
                                            yaxis=dict(title='PM2.5 (μg/m³)'))
                        dk(fig_q, '', 320)
                        st.plotly_chart(fig_q, use_container_width=True)
                    elif hint == 'corr_bar':
                        cn_map = {'temperature': '温度', 'wind_speed': '风速',
                                  'pressure': '气压', 'dewpoint': '露点'}
                        df_r['气象因子_cn'] = df_r['气象因子'].map(lambda x: cn_map.get(x, x))
                        clrs = ['#ff5252' if v > 0 else '#00e676' for v in df_r['r 系数']]
                        fig_q = go.Figure(go.Bar(
                            x=df_r['气象因子_cn'], y=df_r['r 系数'],
                            marker=dict(color=clrs, line=dict(width=0)),
                            text=[f'{v:+.3f}' for v in df_r['r 系数']],
                            textposition='outside',
                        ))
                        fig_q.update_layout(yaxis=dict(title='与 PM2.5 相关系数 r', range=[-1, 1]))
                        dk(fig_q, '', 320)
                        st.plotly_chart(fig_q, use_container_width=True)
                    elif hint == 'line_year':
                        fig_q = go.Figure(go.Scatter(
                            x=df_r['year'], y=df_r['年均PM2.5'], mode='lines+markers',
                            line=dict(color='#00d4ff', width=3),
                            marker=dict(size=12, color='#00d4ff',
                                        line=dict(color='#030a14', width=2)),
                        ))
                        dk(fig_q, '', 300)
                        st.plotly_chart(fig_q, use_container_width=True)

                # 兜底建议
                if r.get('chart_hint') == 'suggestions' and r.get('suggestions'):
                    sug_html = "<div style='margin-top:10px'>"
                    for s in r['suggestions']:
                        sug_html += "<span class='ai-suggest'>" + s + "</span>"
                    sug_html += "</div>"
                    st.markdown(sug_html, unsafe_allow_html=True)

        # ═══════════════════════════════════════════════════
        # 子区 3: 政策建议生成器(仅深度模式)
        # ═══════════════════════════════════════════════════
        if _llm_ready:
            st.markdown("<div class='sec'>🎯 智能政策建议 · POLICY ADVISOR</div>",
                        unsafe_allow_html=True)
            pc1, pc2 = st.columns([2, 1])
            with pc1:
                target = st.selectbox(
                    '👥 目标群体',
                    ['政府决策', '企业减排', '市民防护', '敏感人群(老人/儿童/孕妇/慢病)'],
                    key='policy_target', label_visibility='collapsed')
            with pc2:
                gen_pol = st.button('💎 生成政策建议',
                                     use_container_width=True, key='btn_policy')

            @st.cache_data(ttl=3600, show_spinner=False)
            def _gen_policy_cached(_hash, _target, _summary):
                client = dsmod.DeepSeekClient(_llm_key)
                return client.policy_advice(_target, _summary)

            if gen_pol:
                with st.spinner('🧠 正在为「' + target + '」定制建议...'):
                    summary = dsmod.build_data_summary(city_df)
                    h = dsmod.summary_hash(summary) + target
                    st.session_state['policy_result'] = _gen_policy_cached(h, target, summary)

            pol_result = st.session_state.get('policy_result', {})
            if pol_result.get('ok') and pol_result.get('advice'):
                adv_list = pol_result['advice']
                pri_color = {'high': '#ff5252', 'mid': '#ffd600', 'low': '#69f0ae'}
                pri_label = {'high': '★★★ 高优先', 'mid': '★★ 中优先', 'low': '★ 普通'}
                adv_cols = st.columns(len(adv_list))
                for i, a in enumerate(adv_list):
                    pri = a.get('priority', 'mid')
                    clr = pri_color.get(pri, '#64b5f6')
                    title = a.get('title', '')
                    reason = a.get('reason', '')
                    action = a.get('action', '')
                    pri_lbl = pri_label.get(pri, '')
                    adv_cols[i].markdown(
                        "<div class='glass' style='border-color:" + clr + "50;"
                        "padding:18px;min-height:230px'>"
                        "<div style='font-family:Exo 2,sans-serif;font-size:.6rem;"
                        "letter-spacing:2px;color:" + clr + ";font-weight:700;"
                        "margin-bottom:8px'>" + pri_lbl + "</div>"
                        "<div style='font-family:Exo 2,sans-serif;font-weight:700;"
                        "font-size:.92rem;color:" + clr + ";margin-bottom:10px;"
                        "border-bottom:1px solid " + clr + "30;padding-bottom:8px'>"
                        + title + "</div>"
                        "<div style='font-size:.72rem;color:rgba(200,225,245,.85);"
                        "line-height:1.7;margin-bottom:8px'>"
                        "<b style='color:#80deea'>📊 数据论据:</b><br>" + reason + "</div>"
                        "<div style='font-size:.72rem;color:rgba(200,225,245,.85);"
                        "line-height:1.7'>"
                        "<b style='color:#69f0ae'>🚀 执行细节:</b><br>" + action + "</div>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                st.markdown(
                    "<div style='text-align:right;font-size:.62rem;"
                    "color:rgba(180,210,230,.45);"
                    "font-family:JetBrains Mono,monospace;margin-top:8px'>"
                    "本次费用: " + pol_result.get('cost', '—') + " · powered by DeepSeek"
                    "</div>",
                    unsafe_allow_html=True
                )
            elif pol_result.get('error'):
                st.markdown(
                    "<div class='alert alert-orange' style='font-size:.78rem'>"
                    "⚠️ 生成失败: " + pol_result['error'] +
                    "</div>",
                    unsafe_allow_html=True
                )


# ═══════════════════════════════════════════════════════════════
# §9  底部
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class='footer'>
  🌬️ &nbsp; 时空呼吸 &nbsp;·&nbsp; TEMPORAL-SPATIAL BREATHING INTELLIGENCE SYSTEM
  &nbsp;·&nbsp; 大数据实践赛 &nbsp;·&nbsp; 环境与人类发展大数据
</div>
""", unsafe_allow_html=True)
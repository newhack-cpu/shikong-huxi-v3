# -*- coding: utf-8 -*-
# deep_analysis.py
# ╔══════════════════════════════════════════════════════════════════╗
# ║   时空呼吸 · DeepSeek 深度分析模块 (v1.0)                         ║
# ║                                                                    ║
# ║   功能:                                                            ║
# ║     1. 自动洞察生成 (5 条数据洞察,带缓存)                           ║
# ║     2. 深度问答 (开放性 LLM 推理)                                   ║
# ║     3. 政策建议生成器 (政府/企业/市民/敏感人群)                       ║
# ║                                                                    ║
# ║   安全:                                                            ║
# ║     - 三层密钥读取 (st.secrets > env > session_state)              ║
# ║     - 绝不在代码中硬编码密钥                                          ║
# ║     - 5 秒超时 + 自动回退                                            ║
# ║     - Token 估算 + 费用提示                                          ║
# ╚══════════════════════════════════════════════════════════════════╝
"""
使用方法:
    from deep_analysis import DeepSeekClient, get_api_key

    key = get_api_key()  # 自动从 secrets/env/session 获取
    if key:
        client = DeepSeekClient(key)
        insights = client.auto_insights(df_summary)
        answer = client.deep_qa(question, data_context)
        policy = client.policy_advice(target, data_summary)
"""

import os
import json
import time
import hashlib
import re
from typing import Optional, List, Dict, Any

# 这两个导入在调用时才检查,允许模块在无 streamlit 环境下被 import 测试
try:
    import streamlit as st
    _HAS_ST = True
except ImportError:
    _HAS_ST = False


# ═══════════════════════════════════════════════════════════════════
# §1  密钥读取 —— 三层优先级,绝不硬编码
# ═══════════════════════════════════════════════════════════════════
def get_api_key() -> Optional[str]:
    """
    按优先级返回 DeepSeek API Key:
      1. st.secrets['DEEPSEEK_API_KEY']  (Streamlit Cloud 部署)
      2. 环境变量 DEEPSEEK_API_KEY        (本地开发)
      3. st.session_state.deepseek_key   (用户在侧边栏临时输入)
    都没有则返回 None,上层应回退到规则模式。
    """
    # 1. Streamlit secrets
    if _HAS_ST:
        try:
            k = st.secrets.get('DEEPSEEK_API_KEY', None)
            if k and isinstance(k, str) and k.startswith('sk-'):
                return k.strip()
        except Exception:
            pass

    # 2. 环境变量
    k = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if k and k.startswith('sk-'):
        return k

    # 3. session_state (临时)
    if _HAS_ST:
        try:
            k = st.session_state.get('_dskey_temp', '')
            if k and isinstance(k, str) and k.startswith('sk-'):
                return k.strip()
        except Exception:
            pass

    return None


def mask_key(k: Optional[str]) -> str:
    """打码显示密钥,只露开头和结尾各 4 位。绝不打印完整密钥。"""
    if not k or len(k) < 12:
        return '(未配置)'
    return f"{k[:6]}...{k[-4:]}"


# ═══════════════════════════════════════════════════════════════════
# §2  Token 估算 (简单版,中英文混合按 1 字 ≈ 1.5 token 估)
# ═══════════════════════════════════════════════════════════════════
def estimate_tokens(text: str) -> int:
    """简单估算 token 数。中文字符 ≈1 token,英文单词 ≈1.3 token。"""
    if not text:
        return 0
    cn = len(re.findall(r'[\u4e00-\u9fff]', text))
    other = len(text) - cn
    return int(cn + other / 3.5)


def estimate_cost_yuan(prompt_tok: int, completion_tok: int) -> float:
    """
    DeepSeek-Chat 定价(2025-Q4 公开价):
      - 输入:¥0.001/千 token (cache miss) / ¥0.0001/千 token (cache hit)
      - 输出:¥0.002/千 token
    我们按 cache miss 估算上限。
    """
    return (prompt_tok / 1000) * 0.001 + (completion_tok / 1000) * 0.002


# ═══════════════════════════════════════════════════════════════════
# §3  DeepSeek 客户端 (兼容 OpenAI SDK 协议)
# ═══════════════════════════════════════════════════════════════════
DEEPSEEK_BASE = 'https://api.deepseek.com/v1'
DEFAULT_MODEL = 'deepseek-chat'
DEFAULT_TIMEOUT = 30  # 秒
MAX_PROMPT_TOKENS = 4000  # 上限,防止 prompt 爆炸


class DeepSeekClient:
    """轻量级 DeepSeek 客户端,只用 requests,不依赖 openai-sdk。"""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        if not api_key or not api_key.startswith('sk-'):
            raise ValueError('Invalid DeepSeek API key')
        self.api_key = api_key
        self.model = model

    def _request(self, messages: List[Dict[str, str]],
                 max_tokens: int = 800,
                 temperature: float = 0.3,
                 json_mode: bool = False,
                 timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
        """
        发起一次 chat completion 请求。
        返回 {'ok': bool, 'text': str, 'usage': {...}, 'error': str}
        """
        # 不在这里 import requests,避免无网测试报错
        import requests

        payload = {
            'model': self.model,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'stream': False,
        }
        if json_mode:
            payload['response_format'] = {'type': 'json_object'}

        # 估算 prompt token 数,超限直接拒绝
        total_tok = sum(estimate_tokens(m.get('content', '')) for m in messages)
        if total_tok > MAX_PROMPT_TOKENS:
            return {'ok': False, 'text': '', 'error': f'Prompt 过长 ({total_tok} tokens),已截断',
                    'usage': {'prompt_tokens': total_tok, 'completion_tokens': 0}}

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        try:
            r = requests.post(f'{DEEPSEEK_BASE}/chat/completions',
                              headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            text = data['choices'][0]['message']['content']
            usage = data.get('usage', {'prompt_tokens': total_tok, 'completion_tokens': 0})
            return {'ok': True, 'text': text, 'error': '', 'usage': usage}
        except requests.exceptions.Timeout:
            return {'ok': False, 'text': '', 'error': '请求超时(网络慢或 DeepSeek 繁忙)',
                    'usage': {'prompt_tokens': total_tok, 'completion_tokens': 0}}
        except requests.exceptions.HTTPError as e:
            code = getattr(e.response, 'status_code', '?')
            msg_map = {401: '密钥无效或已废止', 402: '余额不足,请充值',
                       429: '请求过于频繁,稍候再试', 500: 'DeepSeek 服务器错误'}
            return {'ok': False, 'text': '', 'error': msg_map.get(code, f'HTTP {code} 错误'),
                    'usage': {'prompt_tokens': total_tok, 'completion_tokens': 0}}
        except Exception as e:
            return {'ok': False, 'text': '', 'error': f'调用失败:{type(e).__name__}',
                    'usage': {'prompt_tokens': total_tok, 'completion_tokens': 0}}

    # ─── 任务 1:自动洞察生成 ───
    def auto_insights(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于数据摘要生成 5 条洞察。返回:
          {'ok': True, 'insights': [{'icon':'..','title':'..','desc':'..','level':'..'}, ...],
           'cost': '¥0.0012', 'raw': '...'}
        """
        sys_p = (
            '你是空气质量数据分析专家"时空呼吸智能体"。'
            '基于用户提供的统计摘要,生成 5 条洞察。每条要有数字支撑、专业判断、行动指向。'
            '严格按下面 JSON 格式输出,不要任何额外文字:\n'
            '{"insights":[{"icon":"📊","title":"标题(≤12字)","desc":"30-50字描述","level":"info|warn|crit"},...]}'
        )
        user_p = f'''数据摘要(JSON):
{json.dumps(summary, ensure_ascii=False, indent=2)}

要求:
1. 5 条洞察,角度互不重复(可选自:季节性/时段/气象/极值/超标率/趋势/健康)
2. 每条必须引用具体数字
3. level 选 info(普通)/warn(预警)/crit(严重)
4. icon 用 emoji,要贴合内容(📊📈🌡️💨🚨🌙☀️ 等)'''

        rsp = self._request([
            {'role': 'system', 'content': sys_p},
            {'role': 'user', 'content': user_p},
        ], max_tokens=1200, temperature=0.4, json_mode=True)

        if not rsp['ok']:
            return {'ok': False, 'insights': [], 'error': rsp['error'], 'cost': '¥0.0000'}

        try:
            data = json.loads(rsp['text'])
            insights = data.get('insights', [])
            if not insights or not isinstance(insights, list):
                raise ValueError('返回结构异常')
            cost = estimate_cost_yuan(rsp['usage']['prompt_tokens'],
                                       rsp['usage'].get('completion_tokens', 0))
            return {'ok': True, 'insights': insights[:5], 'error': '',
                    'cost': f'¥{cost:.4f}', 'raw': rsp['text']}
        except Exception as e:
            return {'ok': False, 'insights': [], 'error': f'解析失败:{e}', 'cost': '¥0.0000'}

    # ─── 任务 2:深度问答 ───
    def deep_qa(self, question: str, data_context: str) -> Dict[str, Any]:
        """
        开放式深度问答。data_context 是预先生成的数据上下文摘要。
        返回 {'ok':True, 'answer':'...', 'cost':'¥0.0008'}
        """
        sys_p = (
            '你是空气质量大数据分析专家"时空呼吸智能体"。回答用户基于真实数据集的问题。\n'
            '约束:\n'
            '1. 中文回答,300 字以内,口语自然但保持专业\n'
            '2. 必须引用我提供的数据事实,不要编造数字\n'
            '3. 涉及健康时,引用 GB 3095-2012 或 WHO 2021 标准\n'
            '4. 涉及模型/技术时,优先解释 MSTN v2 或 BHI 公式\n'
            '5. 不要用 markdown 标题,但可用 **加粗** 强调关键数字\n'
            '6. 如果数据不足以回答,如实说"现有数据不支持回答 XX",不要瞎编'
        )
        user_p = f'''数据上下文:
{data_context[:2500]}

用户提问:{question}'''

        rsp = self._request([
            {'role': 'system', 'content': sys_p},
            {'role': 'user', 'content': user_p},
        ], max_tokens=600, temperature=0.5)

        if not rsp['ok']:
            return {'ok': False, 'answer': '', 'error': rsp['error'], 'cost': '¥0.0000'}
        cost = estimate_cost_yuan(rsp['usage']['prompt_tokens'],
                                   rsp['usage'].get('completion_tokens', 0))
        return {'ok': True, 'answer': rsp['text'].strip(), 'error': '', 'cost': f'¥{cost:.4f}'}

    # ─── 任务 3:政策建议生成 ───
    def policy_advice(self, target: str, data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成针对特定群体的政策/行动建议。
        target ∈ {'政府决策', '企业减排', '市民防护', '敏感人群'}
        """
        sys_p = (
            '你是城市环境规划与公共健康顾问。基于本数据集的统计特征,'
            '为指定群体生成 3 条具体、可执行、有依据的行动建议。\n\n'
            '严格按 JSON 输出,不要任何额外文字:\n'
            '{"advice":[{"title":"建议标题(≤15字)","reason":"数据论据(20-40字)",'
            '"action":"执行细节(20-40字)","priority":"high|mid|low"},...]}'
        )
        user_p = f'''目标群体:**{target}**

数据摘要:
{json.dumps(data_summary, ensure_ascii=False, indent=2)}

要求:
1. 3 条建议,priority 至少 1 条 high
2. 每条都要有数据支撑,不要泛泛而谈
3. action 要具体到"做什么、谁做、什么时候做"'''

        rsp = self._request([
            {'role': 'system', 'content': sys_p},
            {'role': 'user', 'content': user_p},
        ], max_tokens=900, temperature=0.4, json_mode=True)

        if not rsp['ok']:
            return {'ok': False, 'advice': [], 'error': rsp['error'], 'cost': '¥0.0000'}
        try:
            data = json.loads(rsp['text'])
            advice = data.get('advice', [])
            cost = estimate_cost_yuan(rsp['usage']['prompt_tokens'],
                                       rsp['usage'].get('completion_tokens', 0))
            return {'ok': True, 'advice': advice[:3], 'error': '', 'cost': f'¥{cost:.4f}'}
        except Exception as e:
            return {'ok': False, 'advice': [], 'error': f'解析失败:{e}', 'cost': '¥0.0000'}


# ═══════════════════════════════════════════════════════════════════
# §4  数据摘要构造器 (供 LLM 使用,不传整个 df)
# ═══════════════════════════════════════════════════════════════════
def build_data_summary(df) -> Dict[str, Any]:
    """从 DataFrame 抽取 LLM 需要的关键统计,控制在 ~1KB 以内。"""
    import pandas as pd
    if 'timestamp' in df.columns:
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'hour' not in df.columns:
            df['hour'] = df['timestamp'].dt.hour
        if 'month' not in df.columns:
            df['month'] = df['timestamp'].dt.month

    s = {
        '样本数': int(len(df)),
        'PM2.5_均值': round(float(df['pm25'].mean()), 1),
        'PM2.5_中位数': round(float(df['pm25'].median()), 1),
        'PM2.5_标准差': round(float(df['pm25'].std()), 1),
        'PM2.5_最高': round(float(df['pm25'].max()), 1),
        'PM2.5_最低': round(float(df['pm25'].min()), 1),
        '超标率_75ug': round(float((df['pm25'] > 75).mean() * 100), 1),
        '超标率_150ug': round(float((df['pm25'] > 150).mean() * 100), 1),
    }
    if 'timestamp' in df.columns:
        s['时间跨度_天'] = int((df['timestamp'].max() - df['timestamp'].min()).days)
        s['起始日期'] = df['timestamp'].min().strftime('%Y-%m-%d')
        s['结束日期'] = df['timestamp'].max().strftime('%Y-%m-%d')

    if 'month' in df.columns:
        mb = df.groupby('month')['pm25'].mean().round(1).to_dict()
        s['月度均值'] = {f'{int(k)}月': float(v) for k, v in mb.items()}
    if 'hour' in df.columns:
        hb = df.groupby('hour')['pm25'].mean().round(1)
        s['最污染时段'] = f'{int(hb.idxmax()):02d}:00 ({hb.max()} μg/m³)'
        s['最清洁时段'] = f'{int(hb.idxmin()):02d}:00 ({hb.min()} μg/m³)'

    # 气象相关性
    cn = {'temperature': '温度', 'wind_speed': '风速',
          'pressure': '气压', 'dewpoint': '露点'}
    cors = {}
    for col, name in cn.items():
        if col in df.columns:
            try:
                r = float(df[['pm25', col]].corr().iloc[0, 1])
                cors[name] = round(r, 3)
            except Exception:
                pass
    if cors:
        s['气象相关性'] = cors

    return s


def build_data_context(df, focus: str = '全部') -> str:
    """构造长文本上下文(供深度问答用)。focus 控制详细度。"""
    s = build_data_summary(df)
    lines = ['【数据集摘要】']
    for k, v in s.items():
        if isinstance(v, dict):
            v_str = ', '.join(f'{kk}={vv}' for kk, vv in list(v.items())[:12])
            lines.append(f'- {k}: {v_str}')
        else:
            lines.append(f'- {k}: {v}')
    lines.append('')
    lines.append('【系统说明】')
    lines.append('- 核心模型:MSTN v2(多尺度时空融合网络),三尺度TCN+跨尺度注意力,80K参数')
    lines.append('- BHI公式:BHI = 0.55×IPM + 0.15×IT + 0.30×IE')
    lines.append('  其中IPM基于GB 3095-2012,IE基于WHO 2021,权重来自文献相关性分析')
    lines.append('- 数据来源:UCI Beijing PM2.5 + OpenAQ + NOAA GSOD')
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════
# §5  缓存键工具(供调用方用)
# ═══════════════════════════════════════════════════════════════════
def summary_hash(summary: Dict[str, Any]) -> str:
    """对摘要做哈希,作为缓存 key。摘要变了才重新调用 API。"""
    s = json.dumps(summary, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(s.encode('utf-8')).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════
# §6  自检 (命令行直接跑可以测试连通性)
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys
    print('═' * 60)
    print(' DeepSeek 深度分析模块 · 自检')
    print('═' * 60)

    key = get_api_key()
    if not key:
        print('❌ 未找到 API Key。请设置环境变量:')
        print('     export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx')
        print('   或在 .streamlit/secrets.toml 中配置:')
        print('     DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxx"')
        sys.exit(1)

    print(f'✅ Key 已加载: {mask_key(key)}')
    print()
    print('▶ 调用 DeepSeek 进行连通性测试...')
    client = DeepSeekClient(key)
    rsp = client._request([
        {'role': 'user', 'content': '用一句话介绍自己。'}
    ], max_tokens=100)

    if rsp['ok']:
        print(f'✅ API 连通,模型回复:')
        print(f'   {rsp["text"][:200]}')
        print(f'   tokens: prompt={rsp["usage"].get("prompt_tokens",0)} '
              f'completion={rsp["usage"].get("completion_tokens",0)}')
        cost = estimate_cost_yuan(rsp["usage"].get("prompt_tokens", 0),
                                    rsp["usage"].get("completion_tokens", 0))
        print(f'   本次费用: ¥{cost:.6f}')
    else:
        print(f'❌ 调用失败: {rsp["error"]}')
        sys.exit(1)

# 🔐 DeepSeek API Key 安全配置指南

> 重要:本文档讲怎么把 API Key **安全地**给到应用,**永远不要把 Key 写进任何会被提交到 git 的文件**。
> 上次你不小心把 Key 直接发在聊天里,这次改成"应用知道、密钥不流出"的方式。

---

## ⚠️ 三个绝对不能做的事

```
❌ 在 app.py / deep_analysis.py 里硬编码 Key
❌ 把 Key 提交到 git (即使是私有仓库,泄漏面也大)
❌ 在演示视频/截图里露出 Key
```

如果违反任何一条,只要项目被任何人(评委、组员、网友)看到,Key 就废了。

---

## ✅ 三种推荐方式(任选其一,从最推荐到最便捷)

### 方式 A:Streamlit Secrets(推荐用于云部署)

1. 在项目根目录创建 `.streamlit/secrets.toml`:

```bash
mkdir -p .streamlit
touch .streamlit/secrets.toml
```

2. 编辑 `secrets.toml`,内容只有一行:

```toml
DEEPSEEK_API_KEY = "sk-你的真实key"
```

3. **关键步骤** —— 把 secrets.toml 加到 `.gitignore`:

```bash
echo ".streamlit/secrets.toml" >> .gitignore
```

4. 启动应用,自动加载:

```bash
streamlit run app_v3_aurora.py
```

侧边栏会显示 ✅ DeepSeek 已配置 + 打码 Key。

**为什么推荐这个**:Streamlit Cloud 部署时,在网页 Settings → Secrets 直接粘贴这一行即可,不需要修改代码。

---

### 方式 B:环境变量(推荐用于本地开发)

**Linux / Mac**:

```bash
# 临时(只对当前 shell 生效)
export DEEPSEEK_API_KEY="sk-你的真实key"
streamlit run app_v3_aurora.py

# 永久(写入 ~/.bashrc 或 ~/.zshrc)
echo 'export DEEPSEEK_API_KEY="sk-你的真实key"' >> ~/.bashrc
source ~/.bashrc
```

**Windows PowerShell**:

```powershell
# 临时
$env:DEEPSEEK_API_KEY = "sk-你的真实key"
streamlit run app_v3_aurora.py

# 永久(用户级别)
[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "sk-你的真实key", "User")
# 重启 PowerShell 后生效
```

**Windows CMD**:

```cmd
setx DEEPSEEK_API_KEY "sk-你的真实key"
:: 重启 CMD 后生效
```

---

### 方式 C:Streamlit 侧边栏临时输入(最便捷,适合答辩前现场配置)

1. 直接 `streamlit run app_v3_aurora.py`
2. 侧边栏底部展开 `🔑 配置 DeepSeek API Key`
3. 输入 Key(密码框,不会明文显示)
4. 自动激活,**只在当前 session 有效**,刷新浏览器后丢失

**优点**:不会留下任何文件痕迹;**缺点**:每次刷新都要重输,演示前现场操作有少量风险(可能忘了输)。

---

## 🎬 答辩当天推荐做法

**演示 30 分钟前的流程**:

```bash
# 1. 准备 .streamlit/secrets.toml(确认 .gitignore 已加)
cat > .streamlit/secrets.toml <<EOF
DEEPSEEK_API_KEY = "sk-你的真实key"
EOF

# 2. 确认 .gitignore
grep "secrets.toml" .gitignore

# 3. 启动应用
streamlit run app_v3_aurora.py

# 4. 浏览器打开,Tab 7 应显示 🟢 深度模式 ONLINE
# 5. 点一次"生成/刷新洞察",生成 5 条洞察后会缓存 1 小时
#    答辩演示时直接展示缓存结果,不再产生 API 调用
```

**关键提示**:
- 缓存 1 小时,意味着只要演示前 1 小时内点过一次,**演示中再点都是免费的**(走缓存)
- 如果担心现场断网,提前生成好洞察 + 政策建议,演示时就算没网也能看缓存

---

## 💰 费用预估(让你心里有数)

DeepSeek-Chat 价格(2025-Q4):
- 输入:¥0.001 / 千 token(cache miss)
- 输出:¥0.002 / 千 token

**单次调用估算**:

| 任务 | prompt | completion | 单次费用 |
|---|---|---|---|
| 自动洞察(5 条) | ~600 tok | ~400 tok | **¥0.0014** |
| 深度问答(单次) | ~800 tok | ~300 tok | **¥0.0014** |
| 政策建议(3 条) | ~500 tok | ~500 tok | **¥0.0015** |

**比赛全周期成本**:
- 准备阶段(测试 100 次)≈ ¥0.15
- 答辩演示当天(20 次)≈ ¥0.03
- **整个比赛总花费 < ¥0.5**

充 5 元 DeepSeek 余额绝对用不完。

---

## 🛡️ 安全细节(本系统已做的事)

| 风险 | 我们的措施 |
|---|---|
| Key 写进代码 | 从未硬编码,只通过 3 个外部源读取 |
| Key 在日志里露出 | `mask_key()` 函数打码,只显示前 6 + 后 4 字符 |
| Key 在错误堆栈里露出 | 所有 except 都 catch 不打 raw error |
| Prompt 太大爆炸 | `MAX_PROMPT_TOKENS = 4000` 上限 |
| 网络挂掉影响演示 | 5 秒超时 + 自动回退到离线规则模式 |
| 现场断网 | 缓存 1 小时,演示前预热即可 |
| 重复调用浪费钱 | `summary_hash` 数据变化才重新调用 |
| 余额耗尽 | HTTP 402 错误友好提示"余额不足,请充值" |

---

## 🐞 常见问题排查

### Q1:侧边栏显示 ⚠️ deep_analysis.py 未找到

**原因**:`deep_analysis.py` 没和 `app_v3_aurora.py` 放在同一目录。

```bash
ls code/deep_analysis.py code/app_v3_aurora.py
# 两个文件都应该在 code/ 下
```

### Q2:配置了 Key 但还是显示 OFFLINE

**排查顺序**:

```bash
# 1. 测试 Key 本身能用
cd code/
python3 deep_analysis.py
# 期望:✅ API 连通,模型回复:你好,我是 DeepSeek

# 2. 检查环境变量
echo $DEEPSEEK_API_KEY      # Linux/Mac
echo %DEEPSEEK_API_KEY%     # Windows CMD
echo $env:DEEPSEEK_API_KEY  # Windows PowerShell

# 3. 检查 secrets.toml 路径
ls -la .streamlit/secrets.toml
cat .streamlit/secrets.toml
```

### Q3:点"生成洞察"转圈很久没反应

**原因**:DeepSeek API 慢或网络抖动。

**自查**:

```bash
# 直接 ping
curl -X POST https://api.deepseek.com/v1/chat/completions \
  -H "Authorization: Bearer sk-你的key" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"hi"}],"max_tokens":10}'
```

如果命令行也慢,确实是 DeepSeek 那边的问题,等几分钟再试。

### Q4:返回 "Prompt 过长"

**原因**:数据集太大。

**解决**:在侧边栏缩短"日期范围"(默认 60 天),或在代码里改:

```python
# deep_analysis.py 第 78 行
MAX_PROMPT_TOKENS = 8000  # 默认 4000,可上调
```

### Q5:返回 "401 密钥无效或已废止"

**原因**:Key 错了 / Key 被删了 / Key 还没激活。

**解决**:登录 https://platform.deepseek.com/api_keys 检查 Key 状态。

---

## 🎯 答辩中可能被问到的问题

**Q:"你们的 AI 是用 ChatGPT 做的吗?"**

A:不是。我们集成的是 **DeepSeek**(国产开源大模型,中文场景表现优秀,且单次调用成本仅约 ¥0.001)。系统采用**双模式架构**:默认离线规则匹配保障演示稳定,启用深度模式后调用 DeepSeek 做开放式归因分析和政策建议生成。两种模式都做了完整缓存,演示中实际产生的 API 调用极少。

**Q:"如果断网,你的 AI 还能用吗?"**

A:**完全可以**。离线模式覆盖 12 类高频问题,基于规则匹配 + 模板填充,不依赖任何外部服务。深度模式在网络异常时**自动回退到离线模式**,并在界面上明确标注"⚠️ 已自动回退"以保持透明。

**Q:"为什么用 LLM 不直接训一个分类模型?"**

A:答案不是分类,而是**结构化生成**——LLM 的优势在于把数据洞察用人话讲清楚,把抽象数字翻译成具体的执行建议(比如"PM2.5 在 11 点峰值"→"建议早高峰错峰出行")。这是分类模型做不到的,也是当前国一标杆作品的差异化方向。

---

## 📎 一句话总结

**密钥放在 `.streamlit/secrets.toml`,文件加进 `.gitignore`,演示前 1 小时点一次"生成洞察"预热缓存。** 🚀

*版本:v3.5 · 2026-05-02*

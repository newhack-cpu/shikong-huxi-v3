# 🚀 时空呼吸 v3.5 · Streamlit Cloud 部署指南

> 把项目部署到 Streamlit Cloud,获得永久公网 URL
> 提交比赛时,可直接链接 URL,**评委手机扫码就能体验**(国奖加分项)

---

## 📋 部署前置条件

| 条件 | 说明 |
|---|---|
| GitHub 账号 | 免费,没有的去 https://github.com/signup |
| Streamlit Cloud 账号 | 免费,用 GitHub 登录 https://share.streamlit.io |
| 项目跑得通 | 本地 `streamlit run app_v3_aurora.py` 没问题 |
| (可选)DeepSeek Key | 启用深度 AI 模式,本地测试好后部署 |

---

## ⚡ 5 分钟一键部署流程

### 第 1 步:推送项目到 GitHub

```bash
cd shikong_huxi_v2

# 初始化 git(如果还没初始化)
git init
git add .
git commit -m "feat: 时空呼吸 v3.5 完整版,含 AI 双模式"

# 创建 GitHub 仓库(用 GitHub 网站或 gh CLI)
# 假设仓库名为 shikong-huxi
git remote add origin https://github.com/<你的用户名>/shikong-huxi.git
git branch -M main
git push -u origin main
```

### 第 2 步:登录 Streamlit Cloud

1. 打开 https://share.streamlit.io
2. 点击 "Sign in with GitHub" → 授权
3. 登录后看到主页 "Workspace"

### 第 3 步:新建 App

1. 点击右上角 "Create app"
2. 选择 "Deploy a public app from GitHub"
3. 填写表单:
   - **Repository**:`<你的用户名>/shikong-huxi`
   - **Branch**:`main`
   - **Main file path**:`code/app_v3_aurora.py`
   - **App URL**(可选):自定义子域名,如 `shikong-huxi`
4. 点击 "Deploy!"

### 第 4 步:等待首次构建(约 5-10 分钟)

构建过程会:
1. 拉取你的代码
2. 读取 `requirements.txt` 安装依赖
3. 启动 streamlit 进程
4. 分配公网 URL(如 `https://shikong-huxi.streamlit.app`)

可在右上角点 "Manage app" 看实时构建日志。

---

## 🔐 配置 DeepSeek API Key(可选,但推荐)

为了部署版本启用 AI 深度模式:

1. 在 Streamlit Cloud 控制台,选择你的 app
2. 点击 "⋮" → "Settings" → "Secrets"
3. 在文本框粘贴:
   ```toml
   DEEPSEEK_API_KEY = "sk-你的真实key"
   ```
4. 点击 "Save"
5. App 会自动重启,几秒后生效

**安全保证**:Streamlit secrets 不会被任何用户看到,也不会出现在代码里。

---

## ⚠️ 常见部署问题

### 问题 1:requirements.txt 缺包

**症状**:构建日志报 `ModuleNotFoundError`

**解决**:
```bash
# 在本地确保 requirements.txt 完整
pip freeze | grep -iE "streamlit|pandas|plotly|joblib|requests" > requirements_check.txt
# 对比 requirements.txt,缺什么补什么
```

需要确保以下包都在(版本可放宽):

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
scikit-learn>=1.3.0
joblib>=1.3.0
requests>=2.31.0
```

### 问题 2:模型文件过大,推送失败

**症状**:`git push` 报 "file too large"(GitHub 单文件 100MB 限制)

**解决**:
- 选项 A:用 Git LFS 管理大文件
  ```bash
  git lfs install
  git lfs track "*.pkl"
  git add .gitattributes
  git commit -m "track pkl with lfs"
  git push
  ```
- 选项 B:把训练好的模型放外部托管(HuggingFace / OSS),启动时下载
- 选项 C:**推荐** — 提交一个轻量"Demo 模型"专门给云端用,只要文件 < 50MB

### 问题 3:启动后看到 SettingsError

**症状**:页面显示 `streamlit.errors.StreamlitSecretNotFoundError`

**解决**:你的代码尝试读 secrets,但 Cloud 没配置。
- 临时:按上文配置 DEEPSEEK_API_KEY
- 或:确认你的代码用了 `try/except`(我们的 `deep_analysis.py` 已处理)

### 问题 4:数据文件读不到

**症状**:`FileNotFoundError: data_with_features.csv`

**解决**:确认 CSV 在 git 里(`git ls-files | grep csv`)。
路径用相对路径 `'data_with_features.csv'`,不要用绝对路径。

### 问题 5:页面打开慢

**原因**:Streamlit Cloud 免费版冷启动 ~30 秒

**优化**:
- 使用 `@st.cache_data` / `@st.cache_resource` 缓存
- 模型用 `joblib.load` 一次性加载到 `st.cache_resource`
- 避免每次 rerun 都重读大文件

---

## 🧪 部署后自查清单

部署成功后,**手机扫码访问**,逐项检查:

| 项 | 检查 | 备注 |
|---|---|---|
| 首屏 | 极光粒子背景出现 | 不应该看到 JS 代码裸露 |
| Hero | "SHIKONG · HUXI" 流光显示 | 标题双向流光动画 |
| 7 个 Tab | 全部能正常切换 | 注意 Tab 7 |
| KPI 卡 | 数字加载正常 | 数据连通 |
| 呼吸球 | 涟漪 + 数字心跳 | CSS 动画 |
| AI 问数 | 推荐问题点击有响应 | 离线模式工作 |
| 深度模式 | 顶部 🟢 ONLINE | 已配 Key 才有 |
| 政策建议 | 选群体能生成 | 需深度模式 |

---

## 🌐 自定义域名(高级,可选)

Streamlit Cloud 默认域名是 `<app>.streamlit.app`,如果想用自己的域名:

1. 你需要拥有一个域名(如 `airquality.com`)
2. 在 Streamlit Cloud Settings → Domain → Add custom domain
3. 按提示在你的 DNS 提供商加 CNAME 记录指向 streamlit
4. 等待 DNS 生效(几分钟到几小时)

**比赛建议**:免费 streamlit.app 子域名就够,自定义域名不是加分项。

---

## 📊 资源限制(免费版)

| 资源 | 限制 |
|---|---|
| 内存 | 1 GB |
| CPU | 共享(突发) |
| 存储 | < 1 GB(代码 + 数据) |
| 并发用户 | 软性 100 左右 |
| 单 app 数量 | 免费 1 个,Pro 无限 |
| 冷启动时间 | ~30s(无访问后) |

如果遇到 OOM,优化数据加载(下采样、按需读)而非升级套餐。

---

## 🎯 答辩环节使用

### 推荐 URL 格式

```
https://shikong-huxi.streamlit.app
```

### 二维码生成

用 https://www.qrcode-monkey.com 生成漂亮二维码,放到 PPT 封底:

> "扫码体验 · LIVE DEMO"

评委用手机扫一下,**3 秒内进入你的系统**——这是国一作品的标准操作。

### 演示备选方案(关键!)

万一答辩当天 Streamlit Cloud 临时故障:

1. **本地备份**:笔记本运行 `streamlit run app_v3_aurora.py`(必备)
2. **离线视频**:U 盘里准备 10 分钟演示视频
3. **截图 PPT**:PPT 内嵌系统全套截图
4. **手机热点**:不要依赖会场 WiFi

---

## 📝 部署 Checklist(打勾完成)

- [ ] GitHub 仓库已创建并 push
- [ ] Streamlit Cloud 账号已创建
- [ ] App 已部署成功,有公网 URL
- [ ] DeepSeek Key 已配置(如启用深度模式)
- [ ] 手机扫码测试通过
- [ ] 7 个 Tab 全部正常
- [ ] 二维码已生成并加入 PPT 封底
- [ ] 本地备份方案已就绪

---

## 🎁 其他部署选项(备选)

如果 Streamlit Cloud 因特殊原因不可用:

| 平台 | 难度 | 优势 | 劣势 |
|---|---|---|---|
| **Streamlit Cloud** ★ | 1/5 | 官方支持,5 分钟部署 | 免费版冷启动 |
| Hugging Face Spaces | 2/5 | AI 友好,免费 GPU 选项 | 中国大陆访问慢 |
| Render | 3/5 | 750h 免费 | 配置稍复杂 |
| 阿里云 / 腾讯云 ECS | 4/5 | 国内访问快,稳 | 要钱(¥30/月起) |
| Docker + 自有服务器 | 5/5 | 完全可控 | 运维成本高 |

**推荐**:首选 Streamlit Cloud,国内访问体验中等但稳定;如果评委多在国内且对速度敏感,可考虑同时部署到阿里云。

---

*v1 · 2026-05-04 · 时空呼吸部署文档*

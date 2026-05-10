# -*- coding: utf-8 -*-
# Streamlit Cloud 部署指南

> 把 Web 应用部署到 Streamlit Cloud，让评委直接打开网址即可访问 Web 演示。  
> **这是国奖一等奖团队的"标配"加分项**：不需要在评审现场依赖你的笔记本网络。

---

## 为什么这件事很值得做

**国奖评审现场可能遇到的尴尬**：
- 评委想看 Web 演示，但你笔记本网络不通
- 你的演示视频太长，评委想自己点击试试
- 答辩时评委追问"你这个 Web 跑起来吗？"

**Streamlit Cloud 部署后**：
- 评委直接打开 `your-app.streamlit.app`，立刻看到完整界面
- 24/7 在线，不依赖你的笔记本
- **作品报告和 PPT 都可以放二维码 → 评委手机直接扫码体验**

---

## 部署 5 步（约 30 分钟）

### Step 1：把代码推到 GitHub

```bash
# 在你本机
cd shikong_huxi_v2/
git init
git add .
git commit -m "时空呼吸 v2 国奖冲刺版"

# 在 GitHub 创建一个仓库（公开或私有都行）
# 然后：
git remote add origin https://github.com/你的用户名/shikong-huxi.git
git push -u origin main
```

如果不想用 GitHub，可以用 Gitee（国内访问更稳定）。

### Step 2：注册 Streamlit Cloud

打开 https://streamlit.io/cloud  
用 GitHub 账号登录（免费版每月 1000 小时使用，对答辩演示完全够用）

### Step 3：选择仓库 + 主文件

在 Streamlit Cloud 的 New App 界面：
- **Repository**: 你的 GitHub 仓库名
- **Branch**: main
- **Main file path**: `code/app.py`（注意要写完整路径）
- **App URL**: 自定义一个好记的名字，例如 `shikong-huxi`

### Step 4：配置 requirements.txt

Streamlit Cloud 会自动读取仓库根目录的 `requirements.txt` 安装依赖。  
但默认的 `torch`、`xgboost` 包很大，可能装不上。

**推荐**：在仓库根目录新建 `streamlit_requirements.txt`，内容更精简：

```text
streamlit>=1.28
pandas>=2.0
numpy>=1.24
plotly>=5.15
joblib>=1.3
scikit-learn>=1.3
matplotlib>=3.7
```

然后在 Streamlit Cloud 的 Advanced settings 中指定 `streamlit_requirements.txt`。

> 注意：如果 Web 应用需要加载 PyTorch 模型（MSTN v2），torch 必须装。
> 但 torch 包太大（~2GB），Streamlit Cloud 部署可能超时。
> 解决方案：把 PyTorch 模型的预测结果离线生成 CSV，Web 直接读 CSV，不在云端跑推理。

### Step 5：处理大文件 / 模型

Streamlit Cloud 仓库限制 1GB，所以：
- ✅ `data_with_features.csv`（约 30MB）可以直接放
- ✅ 传统 ML 模型 `*.pkl`（每个 < 50MB）可以
- ❌ MSTN v2 权重 `mstn_v2_best.pth`（如果 > 100MB）需要外部托管

**外部托管方案**：
1. 上传到阿里云 OSS / 腾讯云 COS / 七牛云
2. 在 `app.py` 启动时下载到本地缓存

```python
import urllib.request
@st.cache_resource
def download_model():
    url = "https://your-cdn.com/mstn_v2_best.pth"
    local = "models/mstn_v2_best.pth"
    if not os.path.exists(local):
        urllib.request.urlretrieve(url, local)
    return local
```

---

## 部署后的检查

部署成功后：
- [ ] 访问 `https://your-app.streamlit.app`，所有 6 个标签页都能加载
- [ ] 用手机访问，移动端布局正常（Streamlit 默认响应式）
- [ ] 几个关键按钮（如"刷新数据"、"切换模型"）能交互
- [ ] 控制台 F12 看是否有 JavaScript 错误

---

## 二维码生成（用于 PPT 和报告）

部署 URL 出来后，生成二维码：
- https://www.qrcode-monkey.com/  
- 或在 Python 里：
  ```python
  import qrcode
  img = qrcode.make("https://shikong-huxi.streamlit.app")
  img.save("qr_demo.png")
  ```

把 `qr_demo.png` 放到：
- 答辩 PPT 第 19 页（Web 应用展示页）
- 作品报告封面或附录
- 演示视频结尾

---

## 答辩用话术

> "我们的 Web 应用已经部署到 Streamlit Cloud，评委老师可以扫描这个二维码
> 用手机直接体验，也可以打开 shikong-huxi.streamlit.app 在电脑上查看。"

这一句话能让评委立刻感受到项目的"完成度"——从研究原型升级为可访问的产品。

---

## 备用方案：如果 Streamlit Cloud 部署失败

**Plan B**：用 ngrok 临时暴露你本机的 Web 应用
```bash
# 在你本机
streamlit run app.py
# 另一个终端
ngrok http 8501
```
ngrok 会给你一个临时 URL（如 `https://abc123.ngrok.io`），评委打开就能访问你的本机应用。

**Plan C**：用 30 秒的 Web 演示视频替代实时访问
- 录一段从打开 URL 到点击各标签的 30 秒视频
- 嵌入到主演示视频或 PPT 中
- 万一现场网络挂了，至少有视频备份

---

## 完成度等级

| 等级 | 你做到了哪一步 |
|---|---|
| ⭐ 基础 | 本地能跑 streamlit run app.py |
| ⭐⭐ 中等 | 录了 Web 演示视频 |
| ⭐⭐⭐ 高 | 部署到 Streamlit Cloud + 二维码 |
| ⭐⭐⭐⭐ 国奖一等奖团队 | 部署 + 二维码 + 移动端测试 + 答辩演练 |

你目前是 ⭐ 级别，做到 ⭐⭐⭐ 大概需要 30 分钟，对国奖加分非常显著。

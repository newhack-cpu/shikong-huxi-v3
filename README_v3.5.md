# 🌬️ 时空呼吸 v3.5 · 完整版

> 修复了 v3.4 的 streamlit JS 裸露 bug,粒子背景通过 components.html 正确执行
>
> 版本:v3.5 · 2026-05-04 · components iframe 注入版

---

## ⚡ 30 秒跑通

```bash
# Linux/Mac
./start_unix.sh   # 选 5

# Windows
start_windows.bat # 选 5
```

打开 http://localhost:8501,看到极光粒子背景 + SHIKONG · HUXI 流光标题就成功了。

---

## 🐞 修复经过(三次坑)

### 坑 1:CSS 注入用 repr() 编码 → 字面 \n 出现
**修复**:CSS 抽到独立文件 `static/aurora.css`,Python 用 `Path.read_text()` 加载。

### 坑 2:`<script>` 在 st.markdown 中被裸露成文本
**根因**:streamlit 的 markdown 出于安全原因,**不执行 `<script>`**,而是当文本输出。
**修复**:粒子 JS 必须用 **`streamlit.components.v1.html`** 注入,它会创建一个 iframe,iframe 里能正常执行 JS。

### 坑 3:iframe 里的粒子只在 iframe 矩形里渲染
**修复**:iframe 里的 JS 通过 `window.parent.document` 访问主页面 body,在主页面创建 fixed 全屏 canvas,粒子动画在主页面上下文运行。streamlit 的 iframe 与主页面同源,无 CORS 问题。

---

## 📦 关键文件

```
shikong_huxi_v2/
├── README_v3.5.md
├── DEEPSEEK_SETUP.md             ← AI 深度分析配置
├── 时空呼吸_v3_视觉预览.html       ← 双击浏览器看效果(含粒子)
├── code/
│   ├── app.py                    ← v2 经典版,完全没动
│   ├── app_v3_aurora.py          ← v3.5 主应用
│   ├── qa_engine.py              ← AI 离线问答(12 类)
│   ├── deep_analysis.py          ← DeepSeek 深度分析(可选)
│   └── static/
│       ├── aurora.css            ← 全部样式(660 行)
│       └── aurora.js             ← 粒子背景(components 兼容版)
├── .streamlit/secrets.toml.template
├── .gitignore
└── start_unix.sh / start_windows.bat
```

---

## 🌌 视觉特性

- **极光背景**:双层径向渐变流动
- **粒子背景**:JS canvas 80 颗粒子上升,**颜色随 BHI 等级变**
- **网格底纹**:CSS 兜底
- **Hero 流光**:6s 双向 shimmer 动画
- **呼吸球**:4 层涟漪 + 8 颗轨道粒子(纯 CSS)
- **玻璃卡 hover**:上浮 + 顶部光线
- **Tab 高亮**:选中态发光阴影

---

## 🧠 Tab 7 · AI 智能分析

- **离线模式**(默认):12 类规则匹配,完全离线,演示稳定
- **深度模式**(配 DeepSeek Key):自动洞察 + 政策建议 + 开放问答
- **容错**:断网自动回退,UI 标"⚠️ 已自动回退"

详见 `DEEPSEEK_SETUP.md`

---

## ⚠️ 跑不起来排查

| 症状 | 解决 |
|---|---|
| 缺包 | `pip install -r requirements.txt` |
| 端口冲突 | `streamlit run app_v3_aurora.py --server.port 8502` |
| 没看到粒子背景 | F12 → Console 看是否有 `[particles] OK` 日志 |
| 看到 JS 代码裸露 | 不应该出现(已删 script 注入,改用 components.html) |
| 页面纯文本无样式 | 确认 `code/static/aurora.css` 存在 |
| AI 模式 OFFLINE | `python3 deep_analysis.py` 看错误 |

---

让每一次呼吸,都被技术守护。🌬️

*v3.5 · 2026-05-04*

# -*- coding: utf-8 -*-
# 🛠️ 使用手册：从解压到提交的完整流程

> 本文档假设你刚解压完 `时空呼吸_国奖冲刺包_v2.zip`，想知道下一步做什么。  
> 全文按"立刻能跑的命令 + 出现什么算成功"的方式组织。

---

## ⚠️ Windows 用户 必读：编码问题（GBK vs UTF-8）

如果你在 Windows 上运行 `pip install -r requirements.txt` 时看到这种错误：

```
UnicodeDecodeError: 'gbk' codec can't decode byte 0xXX in position XX:
illegal multibyte sequence
```

**这是为什么**：

- Windows 中文版默认编码是 GBK（CP936）
- 但 Python 项目文件用 UTF-8 编写
- 老版本 pip / Python 在读文件时用了 GBK，遇到 UTF-8 的中文字节就崩了

**3 个解决方案**（从快到慢，任选其一）：

### 方案 A（最快，强烈推荐）：临时设环境变量
```bat
:: 在 cmd 中
set PYTHONUTF8=1
pip install -r requirements.txt
```

```powershell
# 在 PowerShell 中
$env:PYTHONUTF8 = "1"
pip install -r requirements.txt
```

### 方案 B（永久解决，下次开机也有效）
```bat
setx PYTHONUTF8 1
:: 然后关闭并重新打开 cmd
pip install -r requirements.txt
```

### 方案 C（升级 pip）
```bat
python -m pip install --upgrade pip
:: 升到 23.0+ 后会自动检测 UTF-8 BOM
pip install -r requirements.txt
```

### 方案 D（一次性临时用 UTF-8 模式跑 Python）
```bat
python -X utf8 -m pip install -r requirements.txt
```

> 💡 **快速诊断**：运行 `python code/setup_check.py`，它会自动检测你系统的编码状况，并给出针对性建议。

---

## 第一步：环境准备（10 分钟）

### 1.1 检查 Python 版本

```bash
python --version
```

需要 Python 3.10 或更高。如果没有，去 https://python.org 下载 3.10/3.11/3.12 任一版本。

### 1.2 创建虚拟环境（强烈推荐，避免污染系统 Python）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

激活后命令行前应有 `(venv)` 前缀。

### 1.3 环境兼容性自检

```bash
cd shikong_huxi_v2/code/
python setup_check.py
```

如果显示问题，按提示修复。**Windows 用户务必先看上面"编码问题"章节**。

### 1.4 安装依赖

```bash
# 如果是 Windows 且看到 GBK 错误，先：
set PYTHONUTF8=1

pip install -r requirements.txt
```

如果 PyTorch 安装报错（最常见），用这个命令：

```bash
# CPU 版本（推荐，能跑就行）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 或者 GPU 版本（需要 NVIDIA 显卡）
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 1.5 验证安装

```bash
cd code/
python smoke_test.py
```

**如果你看到这个就成功了**：
```
✅ 全部通过！可以直接运行 python run.py 启动完整流程
```

如果看到 ❌，按提示安装缺失的依赖。

---

## 第二步：第一次跑通项目（30 分钟）

### 选项 A：完整流程（推荐，30-60 分钟）

```bash
cd code/
python run.py
```

会依次跑 12 个 Step（v2 升级版）：
- Step -1: 环境兼容性检查（新增）
- Step  0: 多源数据采集（可选）
- Step  1: UCI 数据下载（必需，约 10MB）
- Step  2: 特征工程（含 BHI v2，~3 分钟）
- Step  3: 5 个传统 ML 模型训练（~2 分钟）
- Step  4: **公平对比实验（揭示数据泄漏，新增）★**
- Step  5: 8 张交互图表生成
- Step  6: **MSTN v2 训练（~5-15 分钟，关键！）★**
- Step  7: 多城市数据采集
- Step  8: **真·消融实验（~10-20 分钟，关键！）★**
- Step  9: 基线对比
- Step 10: 7 张论文级图表
- Step 11: **推理速度 benchmark（新增）**

### 选项 B：快速版（10 分钟，仅供调试）

```bash
python run.py --quick
```

消融实验只跑 1 次种子、10 epoch，**结果不能用于报告**，只能用于"证明流程能跑通"。

### 选项 C：离线模式

```bash
python run.py --skip-dl
```

跳过所有需要网络的步骤。

### 选项 D：只跑某一步

```bash
python run.py --only 4    # 只跑公平对比
python run.py --only 8    # 只跑消融实验
python run.py --from 6    # 从 Step 6 开始
```

### 跑完后，应该有这些输出文件

```
shikong_huxi_v2/code/
├── air_quality_data.csv          ← UCI 清洗后数据
├── data_with_features.csv        ← 65 维特征
├── multi_city_real.csv           ← 多城市数据
├── model_comparison.csv          ← 5 模型对比
├── fair_comparison_results.csv   ← ★ 公平对比真实结果
├── ablation_results_v2.csv       ← ★ 消融真实结果
├── ablation_results_v2.tex       ← ★ LaTeX 三线表
├── baseline_comparison.csv       ← 基线对比
├── mstn_v2_predictions.csv       ← MSTN v2 预测值
├── inference_benchmark.csv       ← ★ 推理速度
├── defense_report.md             ← 答辩用报告
├── fair_comparison_summary.md    ← 公平对比说明
├── models/
│   ├── *.pkl                     ← 5 个传统 ML 模型权重
│   └── mstn_v2_best.pth          ← MSTN v2 权重
└── paper_figures/                ← 7 张论文级图表
```

---

## 第三步：用真实数据更新所有材料（最关键！）

### 提取关键数字

跑完 `run.py` 后，**最重要的是这几个文件**：

```bash
# 1. 看消融真实结果
cat code/ablation_results_v2.csv

# 2. 看公平对比结果（揭示数据泄漏的核心证据）
cat code/fair_comparison_results.csv

# 3. 看基线对比结果
cat code/baseline_comparison.csv

# 4. 看推理速度
cat code/inference_benchmark.csv
```

记下其中 MSTN v2 的 MAE / RMSE / R² / MAPE 等数字。

### 更新数字对齐主表

打开 `docs/01_信息表填写答案与数字对齐主表.md`：
- 找到"模型对比（待真·训练后填入）"这张表
- 把所有 `TBD` 替换成你刚跑出来的真实数字
- **这张表是你的"单一真值来源"**——后面所有材料的数字都以这张表为准

### 更新答辩 PPT

打开 `submission/时空呼吸_答辩PPT_v2.pptx`：
- **第 8 页（消融柱状图）**：把图中数字换成 `ablation_results_v2.csv` 里的值
- **第 14 页（基线对比表）**：把表中所有数字换成 `fair_comparison_results.csv` 里的值
- **第 15 页（多步预测）**：等你跑完多步预测后再更新

> 💡 如果你想完全重新生成 PPT，可以编辑 `code/build_ppt.js`：
> ```bash
> cd code/
> # 用 VSCode/Notepad++ 打开 build_ppt.js
> # ctrl+F 搜索 "10.84" "0.8312" 等占位数字，替换成真实值
> npm install pptxgenjs   # 首次需要装依赖
> node build_ppt.js
> # 生成新的 时空呼吸_答辩PPT_v2.pptx
> ```

### 更新报告

打开 `submission/时空呼吸_作品报告_v3_原版.docx`，按
`docs/04_报告关键章节升级版.md` 替换关键章节。

重点替换：
- 第 5.2 节 表 5-1（基线对比）
- 第 5.3 节 表 5-2（多步预测）
- 第 5.4 节 表 5-3（消融实验）

存为 PDF：`submission/时空呼吸_作品报告_v4.pdf`

---

## 第四步：录制 10 分钟演示视频

### 准备工作

1. 在你的电脑上把 Web 应用跑起来：
   ```bash
   cd code/
   streamlit run app.py
   ```

2. 准备录屏工具：
   - **OBS Studio**（免费，推荐）
   - **剪映**（剪辑+字幕一站式）

3. 写好稿子（已为你准备）：
   - 打开 `docs/05_演示视频脚本_10分钟分镜.md`
   - 总时长卡在 9:45-10:00

### 拍摄步骤

按分镜表逐段录制（详见 `docs/05`）。

### 剪辑

用剪映：
- 拼接所有片段
- 加 BGM（用剪映自带免版权音乐）
- 加双语字幕（中文为主，英文为辅）
- 导出 MP4，1080p，8Mbps

存到：`submission/时空呼吸_演示视频.mp4`

---

## 第五步：填写作品信息概要表

打开 `submission/04-2_作品信息概要表_模板.docx`：
- 对照 `docs/01_信息表填写答案与数字对齐主表.md` 中的答案逐项填写
- 特别注意第 4 项 "AI 工具使用情况说明"
- 导出 PDF：`submission/时空呼吸_作品信息概要表.pdf`

---

## 第六步：部署 Web 到 Streamlit Cloud（强烈推荐，⭐⭐⭐ 加分项）

按 `docs/07_Streamlit_Cloud部署指南.md` 操作，约 30 分钟。

部署后：
- 把云端 URL 加到答辩 PPT
- 生成二维码贴到报告封面 / PPT 第 19 页
- **现场答辩 plan B**：万一笔记本网络挂了，评委手机也能看 Web 演示

---

## 第七步：答辩准备

### 至少做 3 遍这件事

打开 `docs/03_答辩30问_标准答案库.md`，**和你的队员对答练习**：

- **第一遍**：每个人通读所有 30 问，了解全貌
- **第二遍**：每个人重点准备自己负责章节的 5-10 问
- **第三遍**：互相对答，模拟真实评委追问

### 录音回放

把对答过程录下来回听，找：
- 哪里卡顿了？
- 哪里逻辑不清？
- 哪里说"嗯""那个"过多？

---

## 第八步：最终自检 + 提交

### 自检

打开 `docs/08_国奖审计自检清单.md`，逐项打勾。  
☑️ ≥ 25 项 → 国奖一等奖准备就绪。

### 提交

按 `docs/06_提交清单与最后冲刺.md` 中的目录结构组织文件。

---

## 我做完了，但还有问题怎么办？

### 检查日志

```bash
cd code/
python run.py 2>&1 > run.log     # Linux/Mac
# 或
python run.py > run.log 2>&1     # Windows
```

如果某一步失败，`run.log` 会有详细错误。

### 单步重跑

```bash
python run.py --only 8    # 只跑消融实验
python run.py --only 6    # 只跑 MSTN v2 训练
python run.py --from 6    # 从 Step 6 开始
```

### 重置环境

如果环境彻底坏了：
```bash
# 删除所有生成的中间文件
rm code/*.csv code/models/*.pkl code/models/*.pth      # Linux/Mac
del code\*.csv code\models\*.pkl code\models\*.pth     # Windows

# 然后重跑
python run.py
```

---

## 📌 最重要的 5 件事（再说一遍）

1. **跑完真实实验**（ run.py），把所有 `TBD` 替换成真实数字
2. **数字对齐**，报告/PPT/答辩稿全部用同一套数字
3. **AI 使用诚实声明**，作品信息表第 4 项不可省
4. **答辩 30 问烂熟于胸**，团队互答 3 遍
5. **部署 Streamlit Cloud + 二维码**（30 分钟，国奖加分项）

---

## 🆘 紧急联系

如果在使用过程中遇到问题，按优先级查：

1. `python code/setup_check.py` — 自动诊断环境问题
2. `docs/06_提交清单与最后冲刺.md` 的 FAQ 章节
3. `docs/03_答辩30问_标准答案库.md` 看是否答辩相关
4. 代码注释（每个 v2 文件开头都有详细说明）

---

**祝国奖捷报！** 🏆🌬️

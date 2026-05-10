# Windows GBK 编码问题排除指南

> 你在 `pip install -r requirements.txt` 时遇到 GBK 错误？  
> 本文档把所有可能的原因和解决方案都列出来。**先读"快速解决"，再读"原理"**。

---

## 🚀 快速解决（90% 情况按这个就行）

### 方法 1：直接用我们提供的安装脚本（最省心）

```cmd
:: 在项目根目录双击或命令行运行
install_windows.bat
```

这个脚本帮你做了：
- 自动设置 UTF-8 编码（解决 GBK 报错）
- 自动配置清华镜像（避免下载超时）
- 分步安装依赖（避开一次性 -r 的编码 trap）
- 单独装 PyTorch（最大的包）

### 方法 2：手动 3 行命令解决

```cmd
chcp 65001
set PYTHONUTF8=1
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 🔍 GBK 报错的原理（科普）

### 为什么会报错？

你看到的错误大概率是这个：

```
UnicodeDecodeError: 'gbk' codec can't decode byte 0xXX in position XX:
illegal multibyte sequence
```

这不是我们 `requirements.txt` 的问题（我们的文件是纯 ASCII），而是 **pip 安装某个包时**的问题。

### 完整的报错链条

1. **你运行**：`pip install xxx`
2. **pip 下载**：`xxx.tar.gz` 包到本地
3. **pip 解压**：自动读 `setup.py` 或 `README.md`
4. **关键问题**：pip **用系统默认编码读这些文件**
5. **Windows 中文版默认编码**是 `GBK`（cp936）
6. **但人家这些文件是 UTF-8 写的**（包含 `é`, `ü`, `™` 等字符）
7. **GBK 解码 UTF-8** → ❌ UnicodeDecodeError

### 为什么 Linux/Mac 没这个问题？

Linux/Mac 的默认 locale 是 `en_US.UTF-8`，pip 读文件时用 UTF-8，所以不会报错。

只有 **Windows + 中文区域设置** 才会触发。

---

## 🛠️ 8 种修复方案（按效果排序）

### ✅ 方案 1：升级 pip 到 23.1+（**根治**）

```cmd
python -m pip install --upgrade pip
```

pip 23.1+ 在 Windows 上**自动用 UTF-8 读文件**，从根本上解决问题。

### ✅ 方案 2：临时设置 PYTHONUTF8=1

```cmd
set PYTHONUTF8=1
pip install -r requirements.txt
```

PEP 540 引入的"UTF-8 模式"，让 Python 在所有平台上都用 UTF-8。

### ✅ 方案 3：永久启用 UTF-8 模式（推荐）

控制面板 → 区域 → 管理 → 更改系统区域设置 → ☑ **Beta: 使用 UTF-8 提供全球语言支持**

重启电脑生效，从此再也不会遇到 GBK 问题（包括其他软件）。

> ⚠️ 注意：少数老软件可能不兼容，开启后如有问题可关闭。

### ✅ 方案 4：用国内镜像源

```cmd
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么这个有时候能解决？** 因为国内镜像缓存的 wheel 包通常已经预编译，pip 不需要读 setup.py，绕过了 GBK 问题。

更多镜像选择：
| 镜像 | URL |
|---|---|
| 清华 | https://pypi.tuna.tsinghua.edu.cn/simple |
| 阿里 | https://mirrors.aliyun.com/pypi/simple/ |
| 中科大 | https://pypi.mirrors.ustc.edu.cn/simple |
| 豆瓣 | https://pypi.doubanio.com/simple |

### ✅ 方案 5：单独装出错的包

如果只有某个包报 GBK 错（比如 `xxx`）：

```cmd
pip install xxx --no-build-isolation
:: 或者指定预编译版本
pip install xxx --only-binary=:all:
```

`--no-build-isolation` 跳过包内的 setup.py 构建过程；
`--only-binary` 只用预编译 wheel，不下载源码包。

### ✅ 方案 6：用 conda 替代 pip

```cmd
conda install numpy pandas scipy scikit-learn matplotlib
conda install -c conda-forge xgboost lightgbm statsmodels
```

conda 不读 setup.py，从根本上没有 GBK 问题。

### ✅ 方案 7：分步装，避开问题包

我们的 `install_windows.bat` 用的就是这个方案——**逐个装**，避免一次性装 30 个包时其中某个出问题导致整个安装失败。

```cmd
pip install numpy
pip install pandas
pip install scipy
:: ... 一个一个装
```

### ⚠️ 方案 8：修改源代码（不推荐，但作为最后手段）

如果某个包的 setup.py 真的有非 UTF-8 编码：

```python
:: 在 Python 启动时强制 UTF-8
import sys
sys.stdout.reconfigure(encoding='utf-8')
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
```

---

## 🎯 决策树：你应该用哪种方案？

```
你在中国大陆吗？
├── 是 → 用方案 1 (升级 pip) + 方案 4 (清华镜像)
│         即: python -m pip install --upgrade pip
│             pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
│
└── 否 → 用方案 1 + 方案 2
          即: python -m pip install --upgrade pip
              set PYTHONUTF8=1
              pip install -r requirements.txt
```

如果上述都失败：
- 用 `install_windows.bat`（一站式）
- 或换 conda 环境（方案 6）

---

## 🐛 我遇到的具体错误是哪一类？

### 错误 A：`'gbk' codec can't decode byte ... in position ...`

→ 标准 GBK 解码错误，按上述 8 种方案任一即可解决。

### 错误 B：`Could not find a version that satisfies the requirement xxx`

→ 这不是编码问题，是网络问题。用清华镜像（方案 4）。

### 错误 C：`Microsoft Visual C++ 14.0 or greater is required`

→ 缺少 C++ 编译器（某些包需要从源码编译）。  
解决：装 [Microsoft C++ Build Tools](https://aka.ms/vs/17/release/vs_BuildTools.exe)，或用 conda（方案 6）。

### 错误 D：`error: Microsoft Visual C++ 14.0 is required`（装 statsmodels 时）

→ 同上，statsmodels 老版本需要 C++ 编译。  
**最简单解决**：装预编译版 `pip install statsmodels --only-binary=:all:`

### 错误 E：`Error: Out of memory` 或卡死（装 PyTorch 时）

→ PyTorch CPU 版有 800MB+，下载慢且占内存。
**单独装它**：

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

或者先跳过：把 requirements.txt 里 `torch` 注释掉，最后再装。

---

## 📞 还是不行？

按这个清单逐项发问题给我：

1. Windows 版本？（10/11，专业版/家庭版）
2. Python 版本？（运行 `python --version`）
3. pip 版本？（运行 `pip --version`）
4. **完整错误信息**（最少最后 30 行）
5. 你试过哪些方案？

---

**记住：90% 的 GBK 问题用 `install_windows.bat` 一键搞定。**

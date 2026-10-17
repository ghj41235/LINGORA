# LINGORA 运行环境

本项目**唯一第三方依赖**是 [jieba](https://github.com/fxsjy/jieba)（中文分词），其余全部为 Python 标准库。

## 要求
- Python **3.10** 及以上

## 一键配置（推荐 · 离线）

| 系统 | 操作 |
| --- | --- |
| Windows | 双击 `setup.bat` |
| Linux / Mac | `bash setup.sh` |

脚本会：
1. 在 `environment/.venv` 创建**独立虚拟环境**（不污染全局 Python、不碰项目代码）；
2. 从**本地 wheel** 离线安装 jieba（无需联网，绕开国内下载要 VPN 的问题）。

## 手动配置（联网）

```bash
pip install -r requirements.txt
```

## 运行

先 `cd` 到项目根目录，然后：

```bash
# Windows
environment\.venv\Scripts\python.exe main.py

# Linux / Mac
environment/.venv/bin/python main.py
```

> 手动配置（未用 venv）时，直接用 `python main.py` 即可。

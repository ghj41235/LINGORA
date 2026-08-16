"""
罗小黑 键盘模型 AI — 入口脚本
从项目根目录运行：python main.py
"""
import sys
import os

# 确保 src 在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.runtime.main import main

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
战场二 一词多原语键路由验收脚本

验收口径（投 data/猫.frag 17 条后）：
- 「猫是什么」  → IS 键（哺乳动物）
- 「猫能干什么」→ CAN 键（捕鼠）
- 「猫怕什么」  → DISLIKE 键（烫/狗）
- 「猫有什么」  → HAVE 键（爪子）
- 「猫属于什么」→ IS 键（猫科动物）

传导式路由（环环相扣，非查表）：疑问键亮 → 线 → 目标原语键亮。

不 mock 任何机制：真实运行时 + 真实投喂 + 真实对话。
用法：python verify_cat.py   # 退出码 0 = 全过
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.runtime.main import LuoXiaoheiRuntime                 # noqa: E402
from src.inout.frag import ingest_frag_file_to_garden          # noqa: E402


# (问题, 目标原语, 期望命中的子串们)
CASES = [
    ("猫是什么",     "IS",      ["哺乳动物", "食肉动物"]),
    ("猫能干什么",   "CAN",     ["捕鼠", "跳跃"]),
    ("猫怕什么",     "DISLIKE", ["烫", "狗"]),
    ("猫有什么",     "HAVE",    ["爪子", "尾巴"]),
    ("猫属于什么",   "IS",      ["猫科动物"]),
]


def build_runtime():
    tmpdir = tempfile.mkdtemp(prefix="luoxiaohei_cat_")
    rt = LuoXiaoheiRuntime(
        filepath=os.path.join(tmpdir, "cat.luoxiaohei"),
        data_dir=os.path.join(tmpdir, "data"),
    )
    frag_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "猫.frag")
    ingest_frag_file_to_garden(rt.garden, frag_path, source="master")
    return rt


def ask(rt, text: str) -> str:
    rt._cool_residuals()
    buf = io.StringIO()
    with redirect_stdout(buf):
        rt._process_input(text)
    out = buf.getvalue()
    lines = [
        l.strip() for l in out.splitlines()
        if l.strip() and not l.strip().startswith("〔")
    ]
    return "\n".join(lines)


def main() -> int:
    rt = build_runtime()
    print(f"〔花园就绪〕{rt.garden.total_keys} 键 / {rt.garden.total_wires} 线")

    failed = 0
    for question, target_prim, expects in CASES:
        reply = ask(rt, question)
        hit = any(e in reply for e in expects)
        mark = "✓" if hit else "✗"
        if not hit:
            failed += 1
        print(f"\n{mark} [{target_prim:8s}] 问：{question}")
        print(f"     答：{reply}")
        if not hit:
            print(f"     （期望命中：{expects}）")

    print("\n" + ("✓ 一词多原语键路由全过" if failed == 0
                  else f"✗ {failed}/{len(CASES)} 题答错"))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

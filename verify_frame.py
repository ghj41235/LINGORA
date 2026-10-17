# -*- coding: utf-8 -*-
"""
verify_frame.py — 验证 MVP-1 言外之力框架（资格层 + 安全网）

起一个全新实例（临时记忆文件，不碰真实存档），先单独验证 frame 识别，
再跑完整对话，逐一断言四个目标：

    "手机"       → 定义（不是"手机屏幕"之类前缀误匹配）
    "英语"       → 安全网「我还不太懂英语。」（光秃未知概念，不该给随机困惑）
    "你是谁"     → 自我介绍「我是一只黑色的小猫妖，叫罗小黑」
    "你爱吃什么" → 小鱼干（亮度赢家层，必须保持不退化）

"你爱吃什么"放最前：它是四个里唯一走亮度赢家层的（query_what 非定义），
会被前一轮话题的结构性强化污染；另外三个都是确定性 frame 回答，不受污染。
（真实 run() 循环每轮 `_cool_residuals()` 只清亮度，不清 Hebbian/证据的结构强化。）

用法（在项目根目录运行）：
    python verify_frame.py

退出码：0 = 四个目标全部命中；1 = 有任一未命中。
"""

import contextlib
import io
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.frame import recognize_frame
from src.inout.input import _tokenize
from src.runtime.main import LuoXiaoheiRuntime

FAIL = 0

# ── 1. frame 识别（纯符号产生式，不依赖运行时） ─────────────────

EXPECT_ACTS = {
    "手机":        ("ostension", "手机", False, False),
    "英语":        ("ostension", "英语", False, False),
    "你是谁":       ("query_who", "self", True, False),
    "你爱吃什么":     ("query_what", "吃", False, False),
    "手机是什么":     ("query_what", "手机", False, True),
    "你好":        ("greeting", None, False, False),
    "师父给我小鱼干":  ("assert", "师父", False, False),
    "小鱼干好吃吗":    ("query_yesno", None, False, False),
    "为什么":       ("query_why", None, False, False),
}

print("=== frame 识别 ===")
for t, (act, topic, is_self, is_defq) in EXPECT_ACTS.items():
    f = recognize_frame(t, _tokenize(t))
    got = (f.act, f.topic, f.is_self_query, f.is_definition_query)
    ok = got == (act, topic, is_self, is_defq)
    if not ok:
        FAIL = 1
    print(f"  {'✓' if ok else '✗'} {t!r:16} -> {got}")

# ── 2. 完整对话 ───────────────────────────────────────────────

print("\n=== 完整对话 ===")
TMP = os.path.join("data", "_verify_frame.luoxiaohei")
if os.path.exists(TMP):
    os.remove(TMP)

r = LuoXiaoheiRuntime(filepath=TMP, data_dir="data")

# 亮度敏感的在最前，确定性的在后（见文件头说明）
CASES = [
    ("你爱吃什么", "小鱼干"),
    ("手机", "手机是一个巴掌大小的东西"),
    ("英语", "我还不太懂英语"),
    ("你是谁", "我是一只黑色的小猫妖"),
]

results = {}
for inp, want in CASES:
    r._cool_residuals()  # 模拟 run() 每轮开始前冷却残留亮度
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r._handle_conversation(inp)
    results[inp] = buf.getvalue()
    # 去掉元信息行后打印回复正文
    body = "\n".join(
        ln for ln in buf.getvalue().splitlines()
        if not ln.startswith("〔") and ln.strip()
    )
    print(f"\n----- 师父> {inp} -----")
    print(body)

print("\n=== 断言 ===")
for inp, want in CASES:
    out = results[inp]
    ok = want in out
    if not ok:
        FAIL = 1
    print(f"  {'✓' if ok else '✗'} {inp!r} 期望含 {want!r}")
    if not ok:
        body = "\n".join(
            ln for ln in out.splitlines()
            if not ln.startswith("〔") and ln.strip()
        )
        print(f"      实际：{body!r}")

if os.path.exists(TMP):
    os.remove(TMP)

print()
print("PASS" if FAIL == 0 else "FAIL")
sys.exit(FAIL)

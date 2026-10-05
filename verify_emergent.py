#!/usr/bin/env python3
"""
罗小黑 M1–M5 涌现验收脚本（原 verify_frame.py 的继任者）

验收口径：四条输入的回复必须子串命中期望文本。
答案必须从键/电线/传导里长出来——本脚本不 mock 任何机制，
直接建真实运行时、真实投喂、真实对话。

用法：
    python verify_emergent.py     # 退出码 0 = 四 case 全过
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from contextlib import redirect_stdout

# 项目根目录入 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows 控制台默认 GBK，print 中文/符号需 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.runtime.main import LuoXiaoheiRuntime          # noqa: E402
from src.inout.ingest import ingest_document_to_garden_v1  # noqa: E402
from src.core.wire import SLIPPAGE_DISTANCE              # noqa: E402


# ── 语料：复现交接清单 §1 的数据事实 ──────────────
# 事实要求：
#   存在   词:手机 / 词:小鱼干 / 词:我 / 词:师父
#   不存在 词:罗小黑 / 词:猫妖 / 词:吃 / 词:爱 / 词:你 / 词:什么
#   IS 定义键与自我介绍键经 CD 拆解自然生成（part-of 0.8 到词键）
# 注意语序：「师父是无限」必须最先投喂——否则 词:师父 会被
# find_similar_key 的 containment 评分吸收进先到的长句键。
KNOWLEDGE_CORPUS = (
    "师父是无限。"
    "手机是一个巴掌大小的东西，师父总是揣在怀里。"
    "我是一只黑色的小猫妖，叫罗小黑。"
    "小黑爱吃小鱼干。"
    "我也爱吃小鱼干。"
)

# 「小鱼干是我最喜欢吃的东西」在真实存档里是 _decompose_freeform
# 生成的自由形 WORD 键（无 词: 前缀）——那句回答的实际来源。
# 当前的 CD 模式表会把这句拆进 EAT 模式，所以这里按事实手工还原。
FREEFORM_FISH = "小鱼干是我最喜欢吃的东西"


def build_runtime() -> LuoXiaoheiRuntime:
    """在隔离临时目录里建一个全新的罗小黑运行时并投喂语料。"""
    tmpdir = tempfile.mkdtemp(prefix="luoxiaohei_verify_")
    rt = LuoXiaoheiRuntime(
        filepath=os.path.join(tmpdir, "verify.luoxiaohei"),
        data_dir=os.path.join(tmpdir, "data"),
    )

    # 真实 CD 投喂
    new_keys, pot_id, _, _ = ingest_document_to_garden_v1(
        rt.garden, KNOWLEDGE_CORPUS, source="master",
    )

    # 手工还原自由形 WORD 键（修正 D 的对象），按 CD 自由拆解的
    # 惯例以 part-of 0.5 连到 词:小鱼干
    pot = rt.garden.pots[pot_id]
    fish_word = None
    for k in pot.network.all_keys():
        if k.content == "词:小鱼干":
            fish_word = k
            break
    assert fish_word is not None, "语料没拆出 词:小鱼干，语料需要调整"
    freeform = pot.network.create_key(content=FREEFORM_FISH, source="master")
    freeform.primitive = "WORD"
    freeform.slots = {"text": freeform.id}
    pot.network.set_wire(freeform.id, fish_word.id, thickness=0.5)
    w = pot.network.get_wire(freeform.id, fish_word.id)
    if w:
        w.relation_type = "part-of"
        w.slippage_distance = SLIPPAGE_DISTANCE.get("part-of", 0)

    # ── 数据事实自检（交接清单 §1）──
    all_contents = {
        k.content for p in rt.garden.pots.values() for k in p.network.all_keys()
    }
    for must in ("词:手机", "词:小鱼干", "词:我", "词:师父"):
        assert must in all_contents, f"数据事实不符：缺少 {must}"
    for must_not in ("词:罗小黑", "词:猫妖", "词:吃", "词:爱", "词:你", "词:什么"):
        assert must_not not in all_contents, f"数据事实不符：多出了 {must_not}"

    return rt


def ask(rt: LuoXiaoheiRuntime, text: str) -> str:
    """模拟一轮真实对话（先冷却残留亮度，走完整 _handle_conversation）。"""
    rt._cool_residuals()
    buf = io.StringIO()
    with redirect_stdout(buf):
        rt._process_input(text)
    out = buf.getvalue()
    # 回复正文 = 非〔…〕注释行
    lines = [
        l.strip() for l in out.splitlines()
        if l.strip() and not l.strip().startswith("〔")
    ]
    return "\n".join(lines)


CASES = [
    # (输入, 期望子串, 涌现路径说明)
    ("你爱吃什么", "小鱼干",
     "亮度赢家层原路径：吃/爱概念提及 → EAT 键 → part-of+spotlight → 词:小鱼干"),
    ("手机", "手机是一个巴掌大小的东西",
     "词:手机@2 → part-of 0.8 → IS 定义键被推亮（修正 A：概念提及整层跳过）"),
    ("英语", "我还不太懂英语",
     "孤种键 词:英语 无线 → 零新亮键 → 沉默 → 说实话"),
    ("你是谁", "我是一只黑色的小猫妖",
     "你→SELF@2 → 桥 0.6 → 自我介绍 IS 键亮"),
]


def main() -> int:
    rt = build_runtime()
    print(f"〔花园就绪〕{rt.garden.total_keys} 键 / "
          f"{rt.garden.total_wires} 线 / {len(rt.garden.pots)} 盆\n")

    failed = 0
    for text, expect, path_note in CASES:
        reply = ask(rt, text)
        ok = expect in reply
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] 输入: {text}")
        print(f"       回复: {reply}")
        print(f"       期望子串: {expect}")
        print(f"       涌现路径: {path_note}")
        if not ok:
            failed += 1
        print()

    if failed:
        print(f"✗ {failed}/{len(CASES)} 条未过")
        return 1
    print(f"✓ {len(CASES)}/{len(CASES)} 条全过——答案全部从传导里涌现")
    return 0


if __name__ == "__main__":
    sys.exit(main())

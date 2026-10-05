#!/usr/bin/env python3
"""
战场二 .frag 碎片入网验收脚本

验收口径：
1. 三元组正确展开成词键 + 原语键 + 命名槽位 + 带 relation_type 的线。
2. 极性正确落在线：不是/没有 → opposite-of(-1)，是/能/导致 → part-of/cause-of(+1)。
3. 传导问答：问「手机能干什么」→ 答案里含「打电话」。

不 mock 任何机制：真实运行时 + 真实投喂 + 真实对话。

用法：
    python verify_frag.py     # 退出码 0 = 全过
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows 控制台默认 GBK，print 中文/符号需 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.runtime.main import LuoXiaoheiRuntime                 # noqa: E402
from src.inout.frag import ingest_frag_file_to_garden          # noqa: E402
from src.core.wire import RELATION_POLARITY                    # noqa: E402


FRAG_TEXT = """@概念
手机 能 打电话
火灾 导致 烧伤
猫 是一种 动物
猫 不是 植物
"""


def build_runtime_with_frag():
    """隔离临时目录里建运行时，投喂 .frag。"""
    tmpdir = tempfile.mkdtemp(prefix="luoxiaohei_frag_")
    rt = LuoXiaoheiRuntime(
        filepath=os.path.join(tmpdir, "verify.luoxiaohei"),
        data_dir=os.path.join(tmpdir, "data"),
    )

    frag_path = os.path.join(tmpdir, "概念.frag")
    with open(frag_path, "w", encoding="utf-8") as f:
        f.write(FRAG_TEXT)

    new_keys, pot_id, is_new_pot = ingest_frag_file_to_garden(
        rt.garden, frag_path, source="master",
    )
    return rt, pot_id, new_keys


def check_structure(rt, pot_id) -> list[str]:
    """检查键/槽位/线/极性正确入网。返回失败信息列表。"""
    fails: list[str] = []
    pot = rt.garden.pots[pot_id]
    net = pot.network
    by_content = {k.content: k for k in net.all_keys()}

    def wire_rel(a: str, b: str):
        ka, kb = by_content.get(f"词:{a}"), by_content.get(f"词:{b}")
        if ka is None or kb is None:
            return None
        return net.get_wire(ka.id, kb.id)

    # ── CAN 键：手机 能 打电话 ──
    if "词:手机" not in by_content or "词:打电话" not in by_content:
        fails.append("缺词键 词:手机 / 词:打电话")
    if "手机能打电话" not in by_content:
        fails.append("缺 CAN 原语键 手机能打电话")
    else:
        can = by_content["手机能打电话"]
        if can.primitive != "CAN":
            fails.append(f"CAN 键 primitive 错：{can.primitive}")
        if can.slots.get("agent") != by_content.get("词:手机").id:
            fails.append("CAN 键 agent 槽位错")
        if can.slots.get("action") != by_content.get("词:打电话").id:
            fails.append("CAN 键 action 槽位错")
        # 原语键↔词键 线 = part-of
        w = net.get_wire(can.id, by_content["词:手机"].id)
        if w is None or w.relation_type != "part-of":
            fails.append("CAN↔词:手机 线应 part-of")

    # ── 词键间线极性：手机↔打电话 = part-of(+1) ──
    w = wire_rel("手机", "打电话")
    if w is None:
        fails.append("缺 手机↔打电话 线")
    elif w.relation_type != "part-of" or RELATION_POLARITY[w.relation_type] != +1:
        fails.append(f"手机↔打电话 应 part-of(+1)，实际 {w.relation_type}")

    # ── 猫↔植物 = opposite-of(-1) ──
    w = wire_rel("猫", "植物")
    if w is None:
        fails.append("缺 猫↔植物 线")
    elif w.relation_type != "opposite-of" or RELATION_POLARITY[w.relation_type] != -1:
        fails.append(f"猫↔植物 应 opposite-of(-1)，实际 {w.relation_type}")

    # ── 猫↔动物 = part-of(+1)（猫是一种动物）──
    w = wire_rel("猫", "动物")
    if w is None:
        fails.append("缺 猫↔动物 线")
    elif w.relation_type != "part-of" or RELATION_POLARITY[w.relation_type] != +1:
        fails.append(f"猫↔动物 应 part-of(+1)，实际 {w.relation_type}")

    # ── 火灾↔烧伤 = cause-of(+1) ──
    w = wire_rel("火灾", "烧伤")
    if w is None:
        fails.append("缺 火灾↔烧伤 线")
    elif w.relation_type != "cause-of" or RELATION_POLARITY[w.relation_type] != +1:
        fails.append(f"火灾↔烧伤 应 cause-of(+1)，实际 {w.relation_type}")

    # ── 词键共享去重：猫 只建一次 ──
    cats = [k for k in net.all_keys() if k.content == "词:猫"]
    if len(cats) != 1:
        fails.append(f"词:猫 应只建一次，实际 {len(cats)} 次")

    return fails


def ask(rt: LuoXiaoheiRuntime, text: str) -> str:
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
    rt, pot_id, new_keys = build_runtime_with_frag()
    print(f"〔花园就绪〕投喂 .frag 后：{rt.garden.total_keys} 键 / "
          f"{rt.garden.total_wires} 线 / {len(rt.garden.pots)} 盆")

    # 结构检查
    fails = check_structure(rt, pot_id)
    if fails:
        print("\n✗ 结构检查失败：")
        for f in fails:
            print(f"   - {f}")
        return 1
    print("✓ 结构检查全过：词键/原语键/槽位/线 relation_type/极性 全部正确入网")

    # 传导问答检查
    reply = ask(rt, "手机能干什么")
    print(f"\n[问答] 输入：手机能干什么")
    print(f"       回复：{reply}")
    if "打电话" in reply:
        print("✓ 传导够到 CAN 键，答出「打电话」")
    else:
        print("✗ 未答出「打电话」")
        return 1

    print("\n✓ 战场二 .frag 投键闭环全过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

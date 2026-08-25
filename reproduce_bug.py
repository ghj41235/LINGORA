# -*- coding: utf-8 -*-
"""
reproduce_bug.py — 复现"已读乱回"，插桩定位根因（诊断脚本，非正式代码）

直接实例化 LuoXiaoheiRuntime（与测试11.txt 同款全新实例），
monkeypatch 5 个关键函数，逐句喂入问题输入，打印：
  种子（含内容）→ 激活签名 → 胜出语法模式 → 槽位值 → 最终回复

用法（项目根目录）：
    python reproduce_bug.py

不碰真实存档，用临时记忆文件 tmp_repro.luoxiaohei，跑完即删。
"""

import os
import sys
import io
import contextlib

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

TMP_FILE = "tmp_repro.luoxiaohei"

# ── 先清理临时文件（保证全新实例）──
for name in os.listdir(PROJECT_ROOT):
    if name.startswith(TMP_FILE):
        try:
            os.remove(os.path.join(PROJECT_ROOT, name))
        except OSError:
            pass

# ── 导入模块（在 monkeypatch 之前，确保拿到原始函数引用）──
import src.core.grammar as grammar_mod
import src.core.garden as garden_mod
import src.inout.language as language_mod

_orig_compute_sig = grammar_mod.compute_activation_signature
_orig_conduct_match = grammar_mod.conduct_grammar_match
_orig_extract_slots = language_mod._extract_slot_values_from_keys
_orig_garden_conduct = garden_mod.Garden.garden_conduct
_orig_generate = language_mod.LanguageModule.generate_from_garden

CAPTURE = {}  # 每轮重置


def _seed_content(garden, kid, pid):
    try:
        pot = garden.pots.get(pid)
        if pot is None:
            return f"<id={kid}>"
        k = pot.network.get_key(kid)
        if k is None:
            return f"<id={kid}?不存>"
        return f"'{k.content}'"
    except Exception:
        return f"<id={kid}>"


def wrapped_garden_conduct(self, *args, **kwargs):
    seeds = kwargs.get("seeds") or (args[0] if args else [])
    CAPTURE["seeds"] = [
        (self.pots.get(pid).network.get_key(kid).content
         if self.pots.get(pid) and self.pots.get(pid).network.get_key(kid)
         else f"<id={kid}>",
         b, pid)
        for kid, b, pid in seeds
    ]
    return _orig_garden_conduct(self, *args, **kwargs)


def wrapped_compute_sig(cr, garden):
    sig = _orig_compute_sig(cr, garden)
    CAPTURE["signature"] = sig
    return sig


def wrapped_conduct_match(garden, sig, tone="normal"):
    p = _orig_conduct_match(garden, sig, tone)
    if p is not None:
        CAPTURE["pattern"] = {
            "id": p.id,
            "sig": p.activation_signature,
            "templates": p.sentence_structures,
            "tone": p.tone,
            "freq": p.frequency,
        }
    else:
        CAPTURE["pattern"] = None
    return p


def wrapped_extract_slots(bright_keys, cr, key_lookup=None):
    s = _orig_extract_slots(bright_keys, cr, key_lookup)
    CAPTURE["slots"] = dict(s)
    return s


def wrapped_generate(self, garden, kb, result, **kwargs):
    resp = _orig_generate(self, garden, kb, result, **kwargs)
    CAPTURE["response"] = resp
    return resp


grammar_mod.compute_activation_signature = wrapped_compute_sig
grammar_mod.conduct_grammar_match = wrapped_conduct_match
language_mod._extract_slot_values_from_keys = wrapped_extract_slots
garden_mod.Garden.garden_conduct = wrapped_garden_conduct
language_mod.LanguageModule.generate_from_garden = wrapped_generate

# ── 问题输入（取自 测试11.txt）──
TEST_INPUTS = [
    "你好",
    "你认识无限吗？",
    "你爱吃什么？",
    "你知道手机吗？",
    "你会英语吗？",
    "你是谁",
    "你卡了？",
    "我不好",
]


def run():
    from src.runtime.main import LuoXiaoheiRuntime

    # 吞掉启动日志
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rt = LuoXiaoheiRuntime(filepath=TMP_FILE, data_dir="data")

    print("=" * 72)
    print("实例已建（全新记忆，与测试11.txt 同款）。开始逐句复现……")
    print("=" * 72)

    for idx, text in enumerate(TEST_INPUTS, 1):
        CAPTURE.clear()
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            rt._cool_residuals()
            rt._handle_conversation(text)

        print(f"\n【{idx}/{len(TEST_INPUTS)}】输入：{text}")
        # 种子
        seeds = CAPTURE.get("seeds", [])
        if seeds:
            print(f"  种子({len(seeds)})：")
            for content, b, pid in seeds:
                print(f"    - [{pid}] {content}  (亮度{b})")
        else:
            print("  种子：无（走了空种子/独处分支）")
        # 签名
        sig = CAPTURE.get("signature", None)
        print(f"  激活签名：{sig}")
        # 胜出模式
        pat = CAPTURE.get("pattern", None)
        if pat:
            print(f"  胜出模式：#{pat['id']} sig={pat['sig']} tone={pat['tone']}")
            print(f"    模板：{pat['templates'][:3]}{'…' if len(pat['templates'])>3 else ''}")
        else:
            print("  胜出模式：None（未命中，降级路径构建）")
        # 槽位
        slots = CAPTURE.get("slots", None)
        if slots:
            print(f"  槽位：{slots}")
        # 最终回复
        resp = CAPTURE.get("response", "")
        print(f"  回复：{resp.strip()}")
        # 中间 stdout 里可能有的"师父> "之后的正文（供对照）
        print("-" * 72)

    # 清理
    rt._handle_shutdown() if False else None
    for name in os.listdir(PROJECT_ROOT):
        if name.startswith(TMP_FILE):
            try:
                os.remove(os.path.join(PROJECT_ROOT, name))
            except OSError:
                pass
    print("\n（临时文件已清理）")


if __name__ == "__main__":
    run()

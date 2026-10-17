# -*- coding: utf-8 -*-
"""
test_evidence.py —— src/core/evidence.py 单元测试

覆盖对象：
    truth_expectation                纯函数
    revise_evidence                  纯函数
    update_evidence_after_conduction 调用方（用轻量 Fake 对象隔离 Garden/Pot/Network，
                                      不依赖尚未提供的 garden.py / pot.py / network.py）

运行方式（任选其一，在 src/core/ 目录下）：
    python3 -m unittest test_evidence -v
    python3 test_evidence.py

──────────────────────────────────────────────────────────────
本文件里的测试用例分两种：
1. test_regression_ 前缀 —— 曾经是隐患、现已修复（隐患 1/2/3/6），这里锁定
   "修复后的正确行为"，一旦 bug 复现这些用例会变红。
2. test_hazard_ 前缀 —— 断言"代码当前的真实行为"，而这个行为本身就是还没修的
   隐患（隐患 4，以及本文件顺带记录的其他观察），保持原样不动。
（隐患 5 是 docstring 措辞错误，已直接在 evidence.py 里更正，对应测试改为普通用例。）
──────────────────────────────────────────────────────────────

English:
Unit tests for src/core/evidence.py: pure functions truth_expectation and
revise_evidence, plus update_evidence_after_conduction tested via lightweight
fakes; regression tests lock fixed bugs, hazard tests document known ones.
"""

from __future__ import annotations

import os
import random
import sys
import unittest

# 让本文件无论从哪个目录被运行（src/core/ 内部、项目根目录、CI …）都能找到 evidence.py / key.py，
# 不依赖 src 包是否已正确安装或 __init__.py 是否可被包管理器识别。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evidence import (  # noqa: E402
    EVIDENCE_BRIDGE_CONF,
    EVIDENCE_BRIGHT1_CONF,
    EVIDENCE_BRIGHT2_CONF,
    EVIDENCE_COACT_CONF,
    EVIDENCE_CONF_MIN,
    EVIDENCE_DIRECTION_NEG,
    EVIDENCE_HESITATION_NEG,
    EVIDENCE_SEED_CONF,
    revise_evidence,
    truth_expectation,
    update_evidence_after_conduction,
)
from key import Key  # noqa: E402


# ═══════════════════════════════════════════════════════════════
# 1. truth_expectation
# ═══════════════════════════════════════════════════════════════

class TestTruthExpectation(unittest.TestCase):
    """expectation = confidence * (frequency - 0.5) + 0.5"""

    def test_documented_examples(self):
        """docstring 里明确写出的 5 组例子，逐一核对。"""
        cases = [
            (0.9, 0.8, 0.82),
            (0.9, 0.1, 0.54),
            (0.1, 0.8, 0.18),
            (0.1, 0.1, 0.46),
            (0.5, 0.37, 0.50),   # frequency=0.5 时 confidence 取任意值
        ]
        for freq, conf, expected in cases:
            with self.subTest(freq=freq, conf=conf):
                # 用 assertAlmostEqual 而不是 assertEqual：
                # 0.8 * (0.9 - 0.5) + 0.5 在浮点下是 0.8200000000000001，不是字面量 0.82，
                # 直接判等会让一个完全正确的实现被误判为失败。
                self.assertAlmostEqual(truth_expectation(freq, conf), expected, places=7)

    def test_zero_confidence_is_always_neutral(self):
        """confidence=0（毫无证据）时，无论 frequency 是什么，期望值必须是 0.5。"""
        for freq in (0.0, 0.1, 0.5, 0.9, 1.0):
            self.assertAlmostEqual(truth_expectation(freq, 0.0), 0.5, places=7)

    def test_full_confidence_reduces_to_frequency(self):
        """confidence=1（证据充分）时，期望值应退化为 frequency 本身。"""
        for freq in (0.0, 0.25, 0.5, 0.75, 1.0):
            self.assertAlmostEqual(truth_expectation(freq, 1.0), freq, places=7)

    def test_half_frequency_is_always_neutral_regardless_of_confidence(self):
        """frequency=0.5（正反各半）时，不管 confidence 多高，期望值都钉在 0.5。"""
        for conf in (0.0, 0.3, 0.6, 1.0):
            self.assertAlmostEqual(truth_expectation(0.5, conf), 0.5, places=7)

    def test_output_within_bounds_for_all_valid_inputs(self):
        """只要 freq、conf 都落在文档承诺的 [0,1] 内，expectation 必然落在 [0,1] 内。"""
        grid = [i / 20 for i in range(21)]
        for freq in grid:
            for conf in grid:
                exp = truth_expectation(freq, conf)
                self.assertGreaterEqual(exp, 0.0)
                self.assertLessEqual(exp, 1.0)

    def test_hazard_no_clamping_lets_out_of_range_inputs_escape_unchecked(self):
        """
        隐患：truth_expectation 对输入不做任何校验/夹紧。
        一旦 frequency 或 confidence 越界（哪怕只是上游某个别的 bug 传进来的脏数据），
        expectation 会原样跟着越界输出，不报错、不警告——调用方很容易被蒙在鼓里。
        """
        # frequency=2.0 是非法输入（应 ∈[0,1]），期望值被直接顶到 2.0
        self.assertEqual(truth_expectation(2.0, 1.0), 2.0)
        self.assertGreater(truth_expectation(2.0, 1.0), 1.0)
        # frequency=-1.0 同理把期望值拽到 -1.0
        self.assertEqual(truth_expectation(-1.0, 1.0), -1.0)
        self.assertLess(truth_expectation(-1.0, 1.0), 0.0)


# ═══════════════════════════════════════════════════════════════
# 2. revise_evidence —— 规范行为
# ═══════════════════════════════════════════════════════════════

class TestReviseEvidence(unittest.TestCase):
    """
    revise_evidence 的证据修正规则。
    new_freq = (old_freq*old_conf + event_freq*event_conf) / (old_conf+event_conf)
    new_conf = old_conf + event_conf*(1-old_conf)，夹到 [0.01, 0.99]
    """

    def test_positive_event_pulls_frequency_up(self):
        new_f, new_c = revise_evidence(0.5, 0.3, 1.0, EVIDENCE_SEED_CONF)
        self.assertGreater(new_f, 0.5)
        self.assertGreater(new_c, 0.3)

    def test_negative_event_pulls_frequency_down(self):
        new_f, new_c = revise_evidence(0.5, 0.3, 0.0, EVIDENCE_HESITATION_NEG)
        self.assertLess(new_f, 0.5)
        # 负面事件（event_freq=0.0）依然是"多一份证据"，confidence 仍然上升，
        # 只有 frequency 被拉低。这是符合 NARS 语义的（confidence 代表证据量，
        # 不代表证据的方向），不是 bug，只是容易被望文生义地误解。
        self.assertGreater(new_c, 0.3)

    def test_neutral_0_5_pulls_frequency_toward_neutral(self):
        """
        event_freq=0.5 是"中性目标值"：它把 frequency 拉向 0.5，而不是保持原值；
        只有 frequency 已经是 0.5 时才不变。之前 docstring 误写成"不改变 frequency"（隐患 5），
        现已更正，这里锁定真实行为。
        """
        new_f, new_c = revise_evidence(0.9, 0.5, 0.5, 0.5)
        self.assertNotAlmostEqual(new_f, 0.9, places=7)
        self.assertAlmostEqual(new_f, 0.7, places=7)
        # frequency 已经是 0.5 时，0.5 事件确实不改变它
        f_mid, _ = revise_evidence(0.5, 0.5, 0.5, 0.5)
        self.assertAlmostEqual(f_mid, 0.5, places=7)

    def test_confidence_monotonically_nondecreasing_under_realistic_constants(self):
        """项目里实际用到的 EVIDENCE_*_CONF 常量全部 >0，此时 confidence 只会上升或不变。"""
        f, c = 1.0, 0.0
        for _ in range(30):
            new_f, new_c = revise_evidence(f, c, 1.0, EVIDENCE_SEED_CONF)
            self.assertGreaterEqual(new_c, c - 1e-12)
            f, c = new_f, new_c

    def test_confidence_capped_at_0_99_even_after_many_updates(self):
        f, c = 1.0, 0.0
        for _ in range(500):
            f, c = revise_evidence(f, c, 1.0, EVIDENCE_SEED_CONF)
        self.assertLessEqual(c, 0.99)
        self.assertAlmostEqual(c, 0.99, places=6)

    def test_zero_total_weight_short_circuits_and_returns_inputs_unchanged(self):
        """old_conf + event_conf == 0 时不做除法（短路）；frequency 若已在 [0,1] 内原样保留，
        但 confidence 仍要统一夹到下限（修复隐患 3 后，0.0 会被抬到 EVIDENCE_CONF_MIN）。"""
        new_f, new_c = revise_evidence(0.42, 0.0, 0.9, 0.0)
        self.assertEqual(new_f, 0.42)
        self.assertEqual(new_c, EVIDENCE_CONF_MIN)

    def test_result_stays_within_documented_bounds_for_valid_inputs(self):
        """
        对合法输入域（old_freq/old_conf/event_freq/event_conf 均 ∈[0,1]）做随机抽样，
        验证 frequency ∈[0,1]、confidence ∈[0.01,0.99] 始终成立——
        说明公式本身在"正常使用"下是稳的，问题都出在"没人校验输入"上（见下面 hazard 用例）。
        """
        rng = random.Random(20240823)
        for _ in range(3000):
            old_f = rng.random()
            old_c = rng.random()
            ev_f = rng.choice([0.0, 0.5, 1.0, rng.random()])
            ev_c = rng.random()
            if old_c + ev_c == 0.0:
                continue
            new_f, new_c = revise_evidence(old_f, old_c, ev_f, ev_c)
            self.assertGreaterEqual(new_f, 0.0)
            self.assertLessEqual(new_f, 1.0)
            self.assertGreaterEqual(new_c, EVIDENCE_CONF_MIN)
            self.assertLessEqual(new_c, 0.99)


# ═══════════════════════════════════════════════════════════════
# 3. revise_evidence —— 隐患：边界 bug
# ═══════════════════════════════════════════════════════════════

class TestReviseEvidenceHazards(unittest.TestCase):
    """
    对应手册里明确要求检查的三点：
      - frequency 有没有被显式夹在 [0,1]？
      - 有没有除零 / 浮点 / 参数未校验的问题？
      - "增量趋近 1.0"的简化在什么情况下失真？（顺序依赖，见下一节）
    """

    def test_regression_frequency_is_clamped_against_negative_event_conf(self):
        """
        回归测试（原隐患 2）：frequency 现在有显式 clamp。
        负的 event_conf 把 frequency 算成负数后，会被夹回 0.0，不再越界、也不再传播脏值。
        （输入校验仍不在这里做——只夹输出，不校验输入。）
        """
        new_f, new_c = revise_evidence(0.0, 0.1, 1.0, -0.05)
        self.assertEqual(new_f, 0.0)
        self.assertGreaterEqual(new_f, 0.0, "frequency 被夹回 [0,1]")
        self.assertGreaterEqual(new_c, EVIDENCE_CONF_MIN)
        self.assertLessEqual(new_c, 0.99)

    def test_hazard_negative_event_conf_can_make_confidence_decrease(self):
        """
        隐患：手册/文档都说 confidence 应该"单调上升"，但这只是因为目前所有调用点
        传的 event_conf 常量都是正数——函数本身并不禁止负的 event_conf，一旦传入，
        confidence 反而会下降，破坏"revise_evidence 只会增加证据量"的隐含假设。
        """
        _, new_c = revise_evidence(0.9, 0.9, 1.0, -0.5)
        self.assertLess(new_c, 0.9)

    def test_regression_out_of_range_old_freq_is_clamped_back(self):
        """
        回归测试（原隐患 2）：即便 old_freq 因为上游 bug 已经越界（这里 1.5），
        加权平均后得到的 new_freq 仍会被夹回 [0,1]（这里夹到 1.0），
        不再把越界状态继续传播下去。
        """
        new_f, _ = revise_evidence(1.5, 0.5, 1.0, 0.1)
        self.assertEqual(new_f, 1.0)
        self.assertLessEqual(new_f, 1.0, "越界的 old_freq 加权后仍被夹回 [0,1]")

    def test_regression_short_circuit_still_clamps_confidence_to_floor(self):
        """
        回归测试（原隐患 3）：old_conf+event_conf==0.0 的短路分支现在也会统一夹紧，
        返回的 confidence 不会低于 EVIDENCE_CONF_MIN。
        """
        _, new_c = revise_evidence(0.77, 0.0, 0.5, 0.0)
        self.assertEqual(new_c, EVIDENCE_CONF_MIN)
        self.assertGreaterEqual(new_c, EVIDENCE_CONF_MIN, "短路分支不再绕开下限")

    def test_regression_short_circuit_clamps_negative_confidence_to_floor(self):
        """回归测试（原隐患 3）：短路分支里即使 old_conf 是负的（-0.05），
        也会被夹回 EVIDENCE_CONF_MIN，不再把非法负值原样放行。"""
        _, new_c = revise_evidence(0.6, -0.05, 1.0, 0.05)
        self.assertEqual(new_c, EVIDENCE_CONF_MIN)
        self.assertGreaterEqual(new_c, EVIDENCE_CONF_MIN)


# ═══════════════════════════════════════════════════════════════
# 4. revise_evidence —— 顺序依赖（"增量趋近 1.0"简化的失真场景）
# ═══════════════════════════════════════════════════════════════

class TestReviseEvidenceOrderDependence(unittest.TestCase):
    """
    手册问："置信度用'增量趋近 1.0'是简化模型——这个简化在什么情况下失真？"

    答案（本节验证）：当同一个键在一次调用里被多条证据规则依次命中时
    （update_evidence_after_conduction 里，一个键完全可能同时是种子 + 亮度激活
    + 共激活，三条规则依次各调用一次 revise_evidence），confidence 的最终结果
    与命中顺序无关（下面证明恒成立），但 frequency 的最终结果 **依赖于命中顺序**——
    而这个顺序在代码里只是"循环写的先后"，NARS 理论并不要求这个顺序有意义。
    """

    @staticmethod
    def _apply_sequence(f0, c0, events):
        f, c = f0, c0
        for ev_f, ev_c in events:
            f, c = revise_evidence(f, c, ev_f, ev_c)
        return f, c

    def test_confidence_is_order_independent(self):
        """
        数学上可证：1 - new_conf = (1 - old_conf) * (1 - event_conf)，
        对一串事件连乘后就是 (1-c0) * Π(1-ec_i)，乘法可交换，所以无论
        事件按什么顺序应用，最终 confidence 都相同。这里用具体数值验证一次。
        """
        start = (0.3, 0.05)
        events = [(1.0, 0.2), (0.0, 0.3)]
        _, c_forward = self._apply_sequence(*start, events)
        _, c_backward = self._apply_sequence(*start, list(reversed(events)))
        self.assertAlmostEqual(c_forward, c_backward, places=10)

    def test_hazard_frequency_is_order_dependent_unlike_confidence(self):
        """
        隐患：同样两个事件，只是应用顺序反过来，最终 frequency 就不一样了。
        这意味着 update_evidence_after_conduction 里"先处理种子、再处理亮度、
        再处理共激活、再处理桥接、最后处理犹豫/方向切换"这个固定顺序，会实际影响
        一个键最终的 evidence_freq——但这个顺序只是代码组织方式，不是 NARS
        证据合并理论要求的顺序。理论上"合并证据"应该和顺序无关（就像真正的
        NARS 修正公式那样），这里的简化模型破坏了这个性质。
        """
        start = (0.3, 0.05)
        events = [(1.0, 0.2), (0.0, 0.3)]
        f_forward, _ = self._apply_sequence(*start, events)
        f_backward, _ = self._apply_sequence(*start, list(reversed(events)))
        self.assertNotAlmostEqual(f_forward, f_backward, places=6)


# ═══════════════════════════════════════════════════════════════
# 5. update_evidence_after_conduction —— 轻量 Fake 依赖
# ═══════════════════════════════════════════════════════════════
#
# garden.py / pot.py / network.py 不在本次被测文件范围内（README 明确说
# "被测文件最小子集"），所以这里只搭建满足 update_evidence_after_conduction
# 所需接口的最小 Fake：
#   - FakeNetwork.get_key(id) / get_wire(id1, id2)
#   - FakePot.network / is_dormant
#   - FakeGarden.pots
#   - FakeConductionRound.final_pattern / is_hesitating / direction_changed
#   - FakeGardenConductionResult.round1 / round3
# Key 本身用的是真实的 src/core/key.py，不是 Fake。

class _FakeNetwork:
    def __init__(self, keys, wires=()):
        self._keys = {k.id: k for k in keys}
        self._wires = set(wires)  # frozenset({id1, id2}) 的集合

    def get_key(self, key_id):
        return self._keys.get(key_id)

    def get_wire(self, id1, id2):
        return object() if frozenset((id1, id2)) in self._wires else None

    def all_keys(self):
        return list(self._keys.values())


class _FakePot:
    def __init__(self, network, dormant=False):
        self.network = network
        self.is_dormant = dormant


class _FakeGarden:
    def __init__(self, pots):
        self.pots = pots


class _FakeConductionRound:
    def __init__(self, final_pattern, is_hesitating=False, direction_changed=False):
        self.final_pattern = final_pattern
        self.is_hesitating = is_hesitating
        self.direction_changed = direction_changed


class _FakeGardenConductionResult:
    def __init__(self, round1=None, round3=None):
        self.round1 = round1 or {}
        self.round3 = round3 or {}


class TestUpdateEvidenceAfterConduction(unittest.TestCase):

    def test_seed_key_gets_strong_positive_evidence(self):
        k = Key(id=1, content="种子键")
        garden = _FakeGarden({"p1": _FakePot(_FakeNetwork([k]))})
        result = _FakeGardenConductionResult()
        update_evidence_after_conduction(garden, {"p1": [(1, 2)]}, result)
        self.assertGreater(k.evidence_freq, 0.5)  # 种子键默认 freq=1.0，正面证据后仍应显著偏正
        self.assertAlmostEqual(k.evidence_conf, EVIDENCE_SEED_CONF, places=7)

    def test_brightness_2_gives_more_confidence_than_brightness_1(self):
        k_bright = Key(id=1, content="亮度2")
        k_dim = Key(id=2, content="亮度1")
        garden = _FakeGarden({"p1": _FakePot(_FakeNetwork([k_bright, k_dim]))})
        cr = _FakeConductionRound(final_pattern={1: 2, 2: 1})
        result = _FakeGardenConductionResult(round1={"p1": cr})
        update_evidence_after_conduction(garden, {}, result)
        self.assertAlmostEqual(k_bright.evidence_conf, EVIDENCE_BRIGHT2_CONF, places=7)
        self.assertAlmostEqual(k_dim.evidence_conf, EVIDENCE_BRIGHT1_CONF, places=7)
        self.assertGreater(k_bright.evidence_conf, k_dim.evidence_conf)

    def test_coactivated_keys_sharing_a_wire_get_extra_associative_evidence(self):
        kA = Key(id=1, content="A")
        kB = Key(id=2, content="B")
        net = _FakeNetwork([kA, kB], wires=[frozenset((1, 2))])
        garden = _FakeGarden({"p1": _FakePot(net)})
        cr = _FakeConductionRound(final_pattern={1: 1, 2: 1})
        result = _FakeGardenConductionResult(round1={"p1": cr})
        update_evidence_after_conduction(garden, {}, result)
        # 亮度=1 的 BRIGHT1 证据 + 共激活的 COACT 证据，两次 revise_evidence 叠加，
        # 所以最终 confidence 应该明显高于"只有 BRIGHT1"的单独结果。
        self.assertGreater(kA.evidence_conf, EVIDENCE_BRIGHT1_CONF)

    def test_no_wire_between_active_keys_means_no_coactivation_bonus(self):
        kA = Key(id=1, content="A")
        kB = Key(id=2, content="B")
        net = _FakeNetwork([kA, kB], wires=())  # 没有电线
        garden = _FakeGarden({"p1": _FakePot(net)})
        cr = _FakeConductionRound(final_pattern={1: 1, 2: 1})
        result = _FakeGardenConductionResult(round1={"p1": cr})
        update_evidence_after_conduction(garden, {}, result)
        self.assertAlmostEqual(kA.evidence_conf, EVIDENCE_BRIGHT1_CONF, places=7)

    def test_bridge_triggered_key_gets_cross_domain_positive_evidence(self):
        k = Key(id=1, content="桥接键")
        garden = _FakeGarden({"p1": _FakePot(_FakeNetwork([k]))})
        cr3 = _FakeConductionRound(final_pattern={1: 1})
        result = _FakeGardenConductionResult(round3={"p1": cr3})
        update_evidence_after_conduction(garden, {}, result)
        self.assertAlmostEqual(k.evidence_conf, EVIDENCE_BRIDGE_CONF, places=7)
        self.assertGreater(k.evidence_freq, 0.5)  # event_freq=0.7，仍是正面

    def test_hesitation_and_direction_change_both_push_frequency_down(self):
        k1 = Key(id=1, content="犹豫键")
        k2 = Key(id=2, content="换向键")
        net = _FakeNetwork([k1, k2])
        garden = _FakeGarden({"p1": _FakePot(net)})
        cr = _FakeConductionRound(
            final_pattern={1: 1, 2: 1}, is_hesitating=True, direction_changed=True,
        )
        result = _FakeGardenConductionResult(round1={"p1": cr})
        update_evidence_after_conduction(garden, {}, result)
        # 两个键都在同一个 round1 里，既有亮度证据(正面)又有犹豫+换向证据(负面)
        # 三次调用叠加，但两次负面事件应该能把 frequency 压到 0.5 以下
        self.assertLess(k1.evidence_freq, 0.5)
        self.assertLess(k2.evidence_freq, 0.5)

    def test_unknown_pot_id_and_key_id_are_skipped_without_raising(self):
        """参数没有被显式校验，但至少要确认未知 id 会被安全跳过而不是抛异常。"""
        garden = _FakeGarden({"p1": _FakePot(_FakeNetwork([Key(id=1, content="A")]))})
        result = _FakeGardenConductionResult()
        seeds = {"p_not_exist": [(99, 2)], "p1": [(999, 2)]}
        try:
            deltas = update_evidence_after_conduction(garden, seeds, result)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"未知 pot_id/key_id 不应该导致异常，但抛出了: {exc!r}")
        self.assertEqual(deltas, {})

    # ── 隐患 ──────────────────────────────────────────────

    def test_regression_deltas_records_bridge_only_key(self):
        """
        回归测试（原隐患 1）：一个键只被"桥接触发"命中时，deltas 里现在也有它的记录。
        之前第 4 步（桥接）漏写 deltas，导致调试日志完全看不到这类更新。
        """
        k = Key(id=1, content="只被桥接触发")
        garden = _FakeGarden({"p1": _FakePot(_FakeNetwork([k]))})
        cr3 = _FakeConductionRound(final_pattern={1: 1})
        result = _FakeGardenConductionResult(round3={"p1": cr3})

        deltas = update_evidence_after_conduction(garden, {}, result)

        self.assertIn(1, deltas, "桥接命中的键现在也会被记进 deltas")
        self.assertGreater(deltas[1], 0.0)

    def test_regression_deltas_full_sum_when_key_hit_by_multiple_categories(self):
        """
        回归测试（原隐患 1）：一个键在同一次调用里被"亮度激活"+"共激活"两条规则命中时，
        deltas 现在是这两步增量的总和（不再是只记第一步）。deltas = 这个键本轮总共变了多少。
        """
        kA = Key(id=1, content="A")
        kB = Key(id=2, content="B")
        net = _FakeNetwork([kA, kB], wires=[frozenset((1, 2))])
        garden = _FakeGarden({"p1": _FakePot(net)})
        cr = _FakeConductionRound(final_pattern={1: 1, 2: 1})
        result = _FakeGardenConductionResult(round1={"p1": cr})

        # 手工复算：deltas 应该等于"亮度那步的增量" + "共激活那步的增量"
        f0, c0 = 1.0, 0.0
        f1, c1 = revise_evidence(f0, c0, 1.0, EVIDENCE_BRIGHT1_CONF)
        step2_delta = abs(f0 - f1) + abs(c0 - c1)
        f2, c2 = revise_evidence(f1, c1, 0.8, EVIDENCE_COACT_CONF)
        step3b_delta = abs(f1 - f2) + abs(c1 - c2)
        full_expected_delta = step2_delta + step3b_delta

        deltas = update_evidence_after_conduction(garden, {}, result)

        self.assertAlmostEqual(deltas[1], full_expected_delta, places=7)


# ═══════════════════════════════════════════════════════════════
# 6. 附加发现（不在本次核心测试范围内，顺手记录）
# ═══════════════════════════════════════════════════════════════

class TestDecayAllEvidenceBonusObservation(unittest.TestCase):
    """
    decay_all_evidence 直接操作 evidence_conf 边界。这里验证"长期不活跃 → confidence 缓慢下降，
    但处女键（从未被赋予证据）不被反向抬升"的修复后行为。
    """

    def test_regression_decay_never_raises_a_virgin_keys_confidence(self):
        """
        回归测试（原隐患 6）：长期未激活的键 confidence 缓慢下降，但"处女键"
        （从未被赋予证据，conf=0.0）不该被衰减反向抬升到 EVIDENCE_CONF_MIN。
        修复后 decay 对处女键跳过，confidence 保持 0.0。
        """
        import time

        from evidence import EVIDENCE_DECAY_INTERVAL, decay_all_evidence

        virgin = Key(id=1, content="从未被赋予证据的键")
        virgin.created_at = time.time() - EVIDENCE_DECAY_INTERVAL * 5
        virgin.last_activated_at = 0.0
        self.assertEqual(virgin.evidence_conf, 0.0)

        garden = _FakeGarden({"p1": _FakePot(_FakeNetwork([virgin]))})
        decay_all_evidence(garden)

        self.assertEqual(virgin.evidence_conf, 0.0, "衰减不该把处女键的 confidence 抬高")


if __name__ == "__main__":
    unittest.main(verbosity=2)

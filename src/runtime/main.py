"""
键盘模型 AI — 主循环（v0.3 花盆模型）

整个系统的大脑。把输入、传导、消化、语言、存储串起来，
让罗小黑在终端里活过来。

v0.3 升级：Garden（花园）替代单一 KeyNetwork。
花盆隔离噪音，桥接管理跨盆联想。

用法：
    python -m src.runtime.main
    或从项目根目录：
    python main.py
"""

from __future__ import annotations

import sys
import time
import os
import random

# 确保项目根目录在 path 中
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.network import KeyNetwork
from src.core.garden import Garden, GardenConductionResult
from src.core.conduction import ConductionRequest
from src.core.digestion import digest_after_garden_conduction, background_maintenance
from src.core.hesitate import detect_hesitation
from src.core.temperature import (
    calculate_temperature,
    is_followup_question,
    update_workspace,
    decay_workspace,
    get_workspace_preheat,
    temperature_report,
)
from src.core.evidence import update_evidence_after_conduction
from src.core.inference import (
    run_inference_cycle,
    INFERENCE_RECHARGE_PER_CONVO,
)
from src.core.temporal import (
    run_prediction_cycle,
    run_solitude_cycle,
    SOLITUDE_THRESHOLD,
)
from src.core.case import (
    match_case,
    apply_case,
    create_case_from_conduction,
    record_case_outcome,
    get_case_stats,
    CBR_TEMPERATURE_THRESHOLD,
)
from src.core.production import (
    evaluate_situation,
    match_meta_rules,
    update_situation_after_conversation,
    activate_situation_keys,
    get_situation_stats,
    OutputRegulation,
)
from src.knowledge.kb import KnowledgeBase, _infer_drawer_from_content
from src.knowledge.desk import Desk
from src.inout.input import parse_user_input, cleanup_short_term, get_short_term_count, _tokenize
from src.inout.ingest import (
    IngestQueue,
    ingest_file,
    ingest_file_to_garden,
    ingest_document_to_garden,
    # v1.0 CD 拆解路径
    ingest_file_to_garden_v1,
    ingest_document_to_garden_v1,
)
from src.inout.language import (
    LanguageModule,
    startup_message,
    shutdown_message,
    ingest_done_message,
    tired_expression,
    TEMPLATE_PROACTIVE,
)
from src.storage.format import CheckpointManager, create_backup, get_file_info
from src.personality.luoxiaohei import get_personality


# ═══════════════════════════════════════════════
# 运行时参数（来自规格书 05-§7）
# ═══════════════════════════════════════════════

CHECKPOINT_INTERVAL = 300      # 自动保存间隔（秒）
DECAY_SCAN_INTERVAL = 100      # 衰减扫描间隔（迭代数）
DORMANCY_CHECK_INTERVAL = 3600 # 休眠检查间隔（秒）

DEFAULT_FILE = "luoxiaohei.luoxiaohei"
DATA_DIR = "data"

# ── 体感时间（昼夜节律） ──────────────────────

TIME_PERIODS = ["早晨", "下午", "傍晚", "深夜"]
TIME_PERIOD_HOURS: dict[str, tuple[int, int]] = {
    "早晨": (5, 10),
    "下午": (10, 17),
    "傍晚": (17, 21),
    "深夜": (21, 5),   # 跨夜：21:00 ~ 次日 5:00
}


# ═══════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════

def _split_response_to_sentences(response: str) -> list[str]:
    """
    将生成的回复拆解为独立句子，用于语法模式存储。

    按句号、感叹号、问号、波浪号切分。
    过滤空白和过短的片段。
    """
    import re
    parts = re.split(r'[。！？~]+', response)
    sentences = [p.strip() for p in parts if p.strip()]
    # 过滤过短片段（< 3 字）和纯标点
    sentences = [s for s in sentences if len(s) >= 3]
    # 去除"师父，"前缀（保留核心句式）
    cleaned = []
    for s in sentences:
        s = re.sub(r'^(师父|小黑)[，,]\s*', '', s)
        if s:
            cleaned.append(s)
    return cleaned


# ═══════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════

class LuoXiaoheiRuntime:
    """罗小黑运行时。管理花园（盆集合）、知识库、主循环。"""

    def __init__(
        self,
        filepath: str = DEFAULT_FILE,
        data_dir: str = DATA_DIR,
    ):
        self.filepath = filepath
        self.data_dir = data_dir

        # 性格
        self.personality = get_personality()

        # 存储
        self.checkpoint = CheckpointManager(filepath, interval=CHECKPOINT_INTERVAL)

        # 模块
        self.language = LanguageModule()
        self.ingest_queue = IngestQueue(data_dir)

        # 体感时间：4 个固定昼夜键的 ID（必须在 _init_garden 之前）
        self._time_key_ids: dict[str, int] = {}

        # 加载或创建
        self.garden: Garden
        self.knowledge_base: KnowledgeBase
        self.desk: Desk
        self._init_garden()

        # 运行时状态
        self.running = False
        self.last_dormancy_check: float = 0.0
        self.last_conversation_at: float = 0.0
        self.total_conversations: int = 0

        # ── v1.0: Copycat 温度追踪 ──
        self.prev_response_len: int = 0        # 上一轮回答长度（用于追问检测）
        self.prev_hesitated: bool = False      # 上一轮是否犹豫
        self.last_conversation_start: float = 0.0  # 本轮对话开始时间

        # ── v1.0: CBR 追踪 ──
        self.last_cbr_case_id: int | None = None  # 上次使用的 CBR 案例 ID
        self.cbr_applied_this_turn: bool = False  # 本轮是否使用了 CBR

        # ── v1.0: 产生式系统追踪 ──
        self.last_output_regulation: OutputRegulation | None = None
        self.prev_master_input: str = ""
        self._last_situation_strengths: dict[str, float] = {}

        # 疲劳与睡眠
        self.fatigue: float = 0.0
        self.last_saved_at: float = 0.0
        self.sleep_duration: float = 0.0
        self.fatigue_before_sleep: float = 0.0

    # ── 初始化 ──────────────────────────────────

    def _init_garden(self):
        """加载已有花园或创建全新的。"""
        self.garden, self.knowledge_base, self.desk, meta = self.checkpoint.load()

        # 记录上次保存时间（用于计算睡眠时长和疲劳恢复）
        self.last_saved_at = meta.get("saved_at", 0.0)

        if meta["is_new"]:
            print("〔新生命〕创建了全新的花园。")
            self.garden.init_default_garden()
            self._seed_knowledge_base()
            self._ingest_data_dir()
        else:
            saved_at_str = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(meta.get("saved_at", time.time())),
            )
            pot_count = meta.get("pot_count", 1)
            print(
                f"〔加载〕从检查点恢复：{meta['key_count']} 键，"
                f"{meta['wire_count']} 线，{pot_count} 盆，"
                f"保存于 {saved_at_str}"
            )

        # 确保 4 个体感时间键存在（兼容旧网络升级）
        self._ensure_time_keys()

        # ── v1.0: 语法键网络初始化 ──
        from src.core.grammar import init_grammar_pot
        grammar_pot_id = init_grammar_pot(self.garden)
        if self.garden.grammar_pot_id != grammar_pot_id:
            self.garden.grammar_pot_id = grammar_pot_id

    def _seed_knowledge_base(self):
        """
        首次启动：将种子知识放入文件柜。
        包括中文语法基础等硬参考资料。

        只在 --new 或首次创建时执行一次，
        之后正常启动从检查点加载，不会重复导入。
        """
        import os as _os

        # 语法文件在项目根目录
        grammar_file = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
            "中文语法基础.txt",
        )
        grammar_file = _os.path.normpath(grammar_file)

        if not _os.path.exists(grammar_file):
            return

        try:
            with open(grammar_file, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                return

            # 按大标题切分（一、二、三……）
            import re
            chunks = re.split(r'\n(?=[一二三四五六七八九十]、)', content)
            total = 0
            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk or len(chunk) < 10:
                    continue
                if chunk.startswith("═") or chunk.startswith("┅"):
                    continue
                lines = chunk.split("\n", 1)
                title = lines[0].strip().lstrip("一二三四五六七八九十、").strip()
                if not title or len(title) < 2:
                    continue
                body = lines[1].strip() if len(lines) > 1 else chunk

                drawer = self.knowledge_base.find_drawer(title)
                if drawer is None:
                    drawer = self.knowledge_base.find_drawer(
                        _infer_drawer_from_content(title)
                    )

                entry = self.knowledge_base.file_entry(
                    content=body,
                    drawer_title=drawer.title if drawer else title,
                    source_type="document",
                )
                if entry is not None:
                    total += 1

            if total > 0:
                print(f"〔文件柜〕种子知识入库：{total} 条（中文语法基础）")

        except Exception as e:
            print(f"〔文件柜〕种子导入跳过：{e}")

    def _ingest_data_dir(self):
        """启动时扫描并投喂 data/ 目录下的种子文件（v1.0 CD 拆解）。"""
        self.ingest_queue.scan()
        total_new = 0
        while self.ingest_queue.has_pending():
            filepath = self.ingest_queue.next()
            if filepath is None:
                break
            try:
                n = ingest_file_to_garden_v1(self.garden, filepath)
                total_new += n
                fname = os.path.basename(filepath)
                print(f"〔投喂/CD〕{fname} → {n} 个概念键")
            except Exception as e:
                print(f"〔投喂失败〕{filepath}: {e}")

        if total_new > 0:
            print(f"〔投喂完成〕共 {total_new} 个概念键（已 CD 拆解）")
            self.checkpoint.save(self.garden, self.knowledge_base, self.desk)

    # ── 体感时间 ─────────────────────────────────

    def _ensure_time_keys(self):
        """
        确保 4 个体感时间键存在于内部盆。
        """
        internal_pot = self.garden.pots.get(self.garden.internal_pot_id)
        if internal_pot is None:
            return
        network = internal_pot.network

        for period_name in TIME_PERIODS:
            existing = None
            for k in network.all_keys():
                if k.content == period_name and k.source == "internal":
                    existing = k
                    break

            if existing:
                self._time_key_ids[period_name] = existing.id
            else:
                key = network.create_key(
                    content=period_name,
                    source="internal",
                )
                key.activation_count = 9999
                key.peak_brightness = 2
                self._time_key_ids[period_name] = key.id

    def _get_current_period(self) -> str:
        """根据当前时间判断属于哪个时段。"""
        hour = time.localtime().tm_hour
        if 5 <= hour < 10:
            return "早晨"
        elif 10 <= hour < 17:
            return "下午"
        elif 17 <= hour < 21:
            return "傍晚"
        else:
            return "深夜"

    def _activate_time_period(self) -> tuple[int | None, str | None]:
        """
        点亮当前时段的时间键（亮度=1）。
        返回 (时间键 ID, 内部盆 ID)。
        """
        period = self._get_current_period()
        period_id = self._time_key_ids.get(period)
        internal_pot_id = self.garden.internal_pot_id

        if period_id is not None and internal_pot_id:
            internal_pot = self.garden.pots.get(internal_pot_id)
            if internal_pot is not None:
                key = internal_pot.network.get_key(period_id)
                if key is not None and key.brightness == 0:
                    key.activate(1)

        return period_id, internal_pot_id

    # ── 主循环 ──────────────────────────────────

    def run(self):
        """启动主循环。"""
        self.running = True

        # 计算睡了多久
        now = time.time()
        if self.last_saved_at > 0:
            offline_seconds = now - self.last_saved_at
            self.sleep_duration = offline_seconds / 3600.0  # 转为小时

            # 离线超过 30 分钟 → 真的睡了一觉
            if offline_seconds > 1800:
                # 计算疲劳恢复
                recovery = self.sleep_duration * self.personality.fatigue_recovery_per_hour
                self.fatigue_before_sleep = self.fatigue
                self.fatigue = max(0.0, self.fatigue - recovery)
        else:
            self.sleep_duration = 0.0
            self.fatigue_before_sleep = 0.0
            self.fatigue = 0.0

        # 启动问候
        print()
        msg = startup_message(
            sleep_hours=self.sleep_duration,
            fatigue_before=self.fatigue_before_sleep,
            fatigue_after=self.fatigue,
        )
        print(msg)
        print()
        print("（输入 /help 查看指令，/sleep 保存并退出）")
        print()

        try:
            while self.running:
                try:
                    user_input = input("师父> ").strip()
                except (EOFError, KeyboardInterrupt):
                    self._handle_shutdown()
                    break

                if not user_input:
                    continue

                # 冷却上轮残留亮度
                self._cool_residuals()

                # 处理输入
                self._process_input(user_input)

                # 后台维护
                self._background_tasks()

        except KeyboardInterrupt:
            self._handle_shutdown()

    # ── 输入处理 ────────────────────────────────

    def _process_input(self, text: str):
        """处理一行用户输入。"""
        if text.startswith("/"):
            self._handle_command(text)
            return

        if text.startswith("[投喂]") or text.startswith("[ingest]"):
            content = text[4:] if text.startswith("[投喂]") else text[8:]
            self._handle_ingest_text(content.strip())
            return

        self._handle_conversation(text)

    # ── 对话处理 ────────────────────────────────

    def _handle_conversation(self, text: str):
        """处理一次对话（花园模式跨盆传导 + v1.0 Copycat 温度）。"""
        start_time = time.time()
        self.last_conversation_start = start_time

        # 0. 点亮体感时间键
        period_id, internal_pot_id = self._activate_time_period()

        # ── v1.0 HTM: 师父说话了（无论是否匹配到种子）──
        self.garden.last_master_input_at = start_time

        # 1. 在所有活跃盆中匹配种子键
        is_master = True
        all_seeds: list[tuple[int, int, str]] = []  # [(key_id, brightness, pot_id)]
        seed_counts: dict[str, int] = {}             # 每个盆匹配到的种子数
        seeds_by_pot: dict[str, list[tuple[int, int]]] = {}  # v1.0: 用于温度计算

        global_total = self.garden.total_keys
        # 预计算每个 token 的全局匹配数（跨所有盆统计）
        tokens = _tokenize(text)
        global_matching: dict[str, int] = {}
        for token in tokens:
            count = 0
            for pot in self.garden.pots.values():
                for k in pot.network.all_keys():
                    if not k.is_dormant and token in k.content:
                        count += 1
            global_matching[token] = count

        for pot in self.garden.get_active_pots():
            seeds = parse_user_input(pot.network, text, is_master=is_master, global_total=global_total, global_matching=global_matching)
            if seeds:
                seed_counts[pot.id] = len(seeds)
                seeds_by_pot[pot.id] = [(kid, b) for kid, b in seeds]
            for kid, brightness in seeds:
                all_seeds.append((kid, brightness, pot.id))

        # 时间键加入种子
        if period_id is not None and internal_pot_id:
            time_key = None
            internal_pot = self.garden.pots.get(internal_pot_id)
            if internal_pot is not None:
                time_key = internal_pot.network.get_key(period_id)
            if time_key is not None and time_key.brightness >= 1:
                all_seeds.append((period_id, time_key.brightness, internal_pot_id))
                if internal_pot_id not in seeds_by_pot:
                    seeds_by_pot[internal_pot_id] = []
                seeds_by_pot[internal_pot_id].append((period_id, time_key.brightness))

        if not all_seeds:
            now = time.time()

            # ── v1.0 Phase 6: 主动搭话检测 ──
            hour = time.localtime().tm_hour
            pa_hours = self.personality.proactive_active_hours
            in_active_hours = pa_hours[0] <= hour < pa_hours[1]
            idle_minutes = (now - self.garden.last_master_input_at) / 60.0
            time_since_proactive = (
                (now - self.garden.last_proactive_at) / 60.0
                if self.garden.last_proactive_at > 0 else 999
            )

            if (self.personality.proactive_enabled
                    and in_active_hours
                    and idle_minutes >= self.personality.proactive_idle_minutes
                    and time_since_proactive >= self.personality.proactive_min_interval
                    and self.fatigue < self.personality.fatigue_tired_threshold):
                # 生成主动搭话
                proactive_msg = random.choice(TEMPLATE_PROACTIVE)
                proactive_msg = self.language._add_personality(proactive_msg)
                print(proactive_msg)
                print()
                self.garden.last_proactive_at = now
                self._post_conversation_update(proactive_msg, False, start_time)
                self.garden.inference_energy.recharge(INFERENCE_RECHARGE_PER_CONVO)
                return

            # ── v1.0 HTM: 独处模式检测 ──
            solitude_seeds = run_solitude_cycle(self.garden)
            if solitude_seeds:
                # 有自发想法 → 用它们当种子走传导流程
                for kid, pid in solitude_seeds:
                    all_seeds.append((kid, 1, pid))
                    if pid not in seeds_by_pot:
                        seeds_by_pot[pid] = []
                    seeds_by_pot[pid].append((kid, 1))
                # 独处模式温度升高
                self.garden.global_temperature = min(
                    0.95, self.garden.global_temperature + 0.15,
                )
            else:
                import random
                response = random.choice(self.personality.confused_responses)
                print(response)
                print()
                self._post_conversation_update("", False, start_time)
                # 即使空种子也要恢复推理能量
                self.garden.inference_energy.recharge(INFERENCE_RECHARGE_PER_CONVO)
                return

        # ── v1.0: 计算温度 ──
        time_since_last = (start_time - self.last_conversation_at) if self.last_conversation_at > 0 else 999
        is_followup = is_followup_question(text, self.prev_response_len, time_since_last)
        temperature = calculate_temperature(
            self.garden,
            seeds_by_pot,
            is_followup=is_followup,
            prev_hesitated=self.prev_hesitated,
        )

        # ── v1.0: 工作空间预热 ──
        preheat = get_workspace_preheat(self.garden)

        # ── v1.0 Phase 6: 情境评估 + 元规则匹配 ──
        prev_master_short = (
            len(self.prev_master_input) < 4 and self.prev_master_input != ""
        )
        situation_strengths = evaluate_situation(
            garden=self.garden,
            input_text=text,
            input_len=len(text),
            time_since_last=time_since_last,
            current_period=self._get_current_period(),
            fatigue=self.fatigue,
            prev_response_len=self.prev_response_len,
            prev_master_short=prev_master_short,
        )
        # 点亮对应情境键
        activate_situation_keys(self.garden, situation_strengths)
        # 保存本轮情境强度（Phase 6: 对话后 evidence 更新用）
        self._last_situation_strengths = situation_strengths
        output_regulation = match_meta_rules(
            garden=self.garden,
            situation_strengths=situation_strengths,
            prev_response_len=self.prev_response_len,
            prev_master_short=prev_master_short,
            is_hesitating=self.prev_hesitated,
        )
        self.last_output_regulation = output_regulation

        # 2. 判断问题复杂度
        is_complex = len(text) > 20
        max_hops = (
            self.personality.complex_hop_limit
            if is_complex
            else self.personality.daily_hop_limit
        )
        max_hops_original = max_hops  # 保存原始值（CBR 可能缩减）

        # ── v1.0 CBR: 案例推理快速路径 ──
        seed_ids_for_cbr = [kid for kid, _, _ in all_seeds]
        cbr_case = match_case(self.garden, seed_ids_for_cbr, temperature)
        self.cbr_applied_this_turn = False
        self.last_cbr_case_id = None

        if cbr_case and temperature < CBR_TEMPERATURE_THRESHOLD:
            # CBR 快速路径：加载案例快照作为强预热，减少传导跳数
            cbr_preheat = apply_case(self.garden, cbr_case, seed_ids_for_cbr)
            for kid, strength in cbr_preheat.items():
                preheat[kid] = max(preheat.get(kid, 0.0), strength)
            max_hops = min(max_hops, 2)  # 大幅减少跳数（节能）
            self.cbr_applied_this_turn = True
            self.last_cbr_case_id = cbr_case.id

        # 3. 执行花园跨盆传导（v1.0：传入温度 + 工作空间预热）
        result = self.garden.garden_conduct(
            seeds=all_seeds,
            max_hops=max_hops,
            source="master_input" if is_master else "user_input",
            temperature=temperature,
            workspace_preheat=preheat,
        )

        # ── v1.0 HTM: 预测周期（时序记忆 + 预测状态机 + 注意力模式）──
        pred_result = run_prediction_cycle(self.garden, result)

        # ── v1.0 NARS: 证据更新 ──
        update_evidence_after_conduction(self.garden, seeds_by_pot, result)

        # ── v1.0 NARS: 推理周期 ──
        inference_result = run_inference_cycle(
            self.garden, result, self.garden.inference_energy,
        )
        self.garden.inference_energy.recharge(INFERENCE_RECHARGE_PER_CONVO)

        # 4. 犹豫检测（检查各盆的第一轮结果）
        is_hesitating = False
        for pot_id, cr in result.round1.items():
            if not cr.is_stable:
                h, detail = detect_hesitation(
                    self.garden.pots[pot_id].network,
                    cr.amplitude_history,
                    cr.elapsed_ms,
                )
                if h:
                    is_hesitating = True
                    cr.is_hesitating = True
                    cr.hesitation_detail = detail

        # 5. 生成回复（传入种子计数 + 产生式调控）
        response = self.language.generate_from_garden(
            self.garden,
            self.knowledge_base,
            result,
            seed_counts=seed_counts,
            output_regulation=output_regulation,
        )

        # ── v1.0: 语法键存储（记录本次格局→输出的映射）──
        self._store_grammar_pattern(result, response, output_regulation)

        # 6. 消化（花园级）
        if period_id is not None and internal_pot_id:
            internal_pot = self.garden.pots.get(internal_pot_id)
            if internal_pot is not None:
                pk = internal_pot.network.get_key(period_id)
                if pk is not None and pk.brightness == 0:
                    pk.activate(1)
        digest_after_garden_conduction(self.garden)

        # ── v1.0: 更新工作空间 ──
        bright_by_pot: dict[str, list[tuple[int, int]]] = {}
        for pot_id, cr in result.round1.items():
            bright_by_pot[pot_id] = list(cr.final_pattern.items())
        for pot_id, cr in result.round3.items():
            if pot_id not in bright_by_pot:
                bright_by_pot[pot_id] = []
            bright_by_pot[pot_id].extend(cr.final_pattern.items())
        update_workspace(self.garden, bright_by_pot)

        # 7. 标记残留
        self._set_residuals(result)

        # ── v1.0 CBR: 案例创建/更新 ──
        if self.cbr_applied_this_turn and self.last_cbr_case_id is not None:
            # 记录 CBR 复用结果
            cbr_case = self.garden.case_library.get(self.last_cbr_case_id)
            if cbr_case is not None:
                total_iters = sum(r.iterations for r in result.round1.values())
                record_case_outcome(cbr_case, success=not is_hesitating,
                                    iterations_saved=max(0, max_hops_original - total_iters))
        elif not self.cbr_applied_this_turn:
            # 正常传导 → 尝试创建新案例
            seed_pot_ids = list(set(pid for _, _, pid in all_seeds))
            # 收集响应中涉及的键 ID（通过匹配响应文本中的子串）
            response_key_ids = self._extract_response_key_ids(response, result)
            create_case_from_conduction(
                self.garden,
                seed_ids_for_cbr,
                seed_pot_ids,
                result,
                response_key_ids,
            )

        # 8. 输出
        print(response)

        # 疲劳表达
        if self.fatigue > self.personality.fatigue_very_tired_threshold:
            print(f"  〔小黑打了个哈欠——他真的很累了〕")
        elif self.fatigue > self.personality.fatigue_tired_threshold:
            if random.random() < 0.3:
                print(f"  〔小黑揉了揉眼睛，有点累了〕")

        # 9. 主动建议放入文件柜
        candidate = self._check_filing_suggestion()
        if candidate:
            print(f"  〔小黑心想：'{candidate.content[:30]}…' 这个用了好多次了，"
                  f"要不要放进柜子里？师父说 /remember 就行〕")

        elapsed = time.time() - start_time
        if elapsed > 5.0:
            total_iters = sum(r.iterations for r in result.round1.values())
            temp_label = "🔥" if temperature > 0.7 else ("🌡" if temperature > 0.3 else "❄")
            cbr_label = " ⚡CBR" if self.cbr_applied_this_turn else ""
            print(f"  〔想了 {elapsed:.1f} 秒，{total_iters} 轮传导，"
                  f"{len(result.round1)} 盆活跃"
                  + (f"，{len(result.bridges_triggered)} 桥接触发" if result.bridges_triggered else "")
                  + f"  {temp_label}T={temperature:.2f}{cbr_label}"
                  + "〕")
        print()

        # 储存当前输入（Phase 6：下一轮情境评估用）
        self.prev_master_input = text

        # 追踪状态
        self._post_conversation_update(response, is_hesitating, start_time, is_complex)

    def _post_conversation_update(self, response: str, is_hesitating: bool, start_time: float,
                                   is_complex: bool = False):
        """对话后状态更新（疲劳、统计、工作空间衰减）。"""
        self.prev_response_len = len(response)
        self.prev_hesitated = is_hesitating
        self.total_conversations += 1
        self.last_conversation_at = time.time()

        # ── v1.0: 衰减工作空间 ──
        decay_workspace(self.garden)

        # 疲劳累积
        if is_hesitating:
            fatigue_delta = self.personality.fatigue_per_complex + self.personality.fatigue_per_hesitation
        elif is_complex:
            fatigue_delta = self.personality.fatigue_per_complex
        else:
            fatigue_delta = self.personality.fatigue_per_simple
        self.fatigue = min(self.personality.fatigue_max,
                          self.fatigue + fatigue_delta)

        # ── v1.0 Phase 6: 更新情境键 evidence ──
        if self.last_output_regulation is not None:
            update_situation_after_conversation(
                garden=self.garden,
                situation_strengths=self._last_situation_strengths,
                response_len=len(response),
                master_response_short=(
                    len(self.prev_master_input) < 4 and self.prev_master_input != ""
                ),
                is_proactive=False,
            )

        # 自动保存
        saved = self.checkpoint.maybe_save(self.garden, self.knowledge_base, self.desk)
        if saved:
            print("  〔已自动保存〕")
            print()

    # ── 指令处理 ────────────────────────────────

    def _handle_command(self, text: str):
        """处理 / 开头的系统指令。"""
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/help":
            self._cmd_help()
        elif cmd == "/status":
            self._cmd_status()
        elif cmd == "/save":
            self._cmd_save()
        elif cmd == "/sleep":
            self._cmd_sleep()
        elif cmd == "/memory":
            self._cmd_memory()
        elif cmd == "/find":
            self._cmd_find(arg)
        elif cmd == "/ingest":
            self._cmd_ingest()
        elif cmd == "/backup":
            self._cmd_backup()
        elif cmd == "/keys":
            self._cmd_keys()
        elif cmd == "/pots":
            self._cmd_pots()
        elif cmd == "/bridges":
            self._cmd_bridges()
        elif cmd == "/cases":
            self._cmd_cases()
        elif cmd == "/remember":
            self._cmd_remember(arg)
        elif cmd == "/kb":
            self._cmd_kb(arg)
        elif cmd == "/write":
            self._cmd_write(arg)
        elif cmd == "/desk":
            self._cmd_desk()
        elif cmd == "/clear":
            self._cmd_clear(arg)
        elif cmd == "/split":
            self._cmd_split(arg)
        elif cmd == "/temp":
            self._cmd_temp()
        elif cmd == "/evidence":
            self._cmd_evidence()
        else:
            print(f"小黑不懂 '{cmd}' 这个指令。输入 /help 看看能做什么。")
            print()

    def _cmd_help(self):
        print("""
┌─ 罗小黑 指令 ──────────────────────┐
│ /help      显示此帮助                │
│ /status    键、线、盆、迭代统计      │
│ /save      立即保存检查点            │
│ /sleep     保存并休眠                │
│ /memory    最近亮过的键              │
│ /find xxx  搜索包含关键词的键        │
│ /ingest    扫描 data/ 投喂新文件     │
│ /backup    备份当前记忆文件          │
│ /keys      显示所有键的内容列表      │
│ /pots      显示花盆列表              │
│ /bridges   显示跨盆桥接统计          │
│ /cases     查看CBR案例库（节能模式）  │
│ /remember  小黑把这句放进文件柜      │
│ /kb        查看文件柜里的知识        │
│ /write     在桌上放一张便条          │
│ /desk      看桌上摊着什么            │
│ /clear     把没用的草稿扔掉          │
│ /split     按主题把当前盆拆分成多个    │
│ /temp      查看温度和搜索状态          │
│ /evidence  查看键的 NARS 证据量       │
└────────────────────────────────────┘
""")

    def _cmd_status(self):
        g = self.garden
        kb = self.knowledge_base
        info = get_file_info(self.filepath)

        bright_count = sum(
            len(pot.network.get_bright_keys())
            for pot in g.pots.values()
        )
        dormant_key_count = sum(
            sum(1 for k in pot.network.all_keys() if k.is_dormant)
            for pot in g.pots.values()
        )

        print(f"""
┌─ 小黑的状态 ────────────────────────┐
│ 盆：{len(g.pots)}（活跃 {g.active_pot_count} / 休眠 {g.dormant_pot_count}）│
│ 键：{g.total_keys}（亮 {bright_count} / 休眠 {dormant_key_count}）│
│ 电线：{g.total_wires}              │
│ 桥接：{g.total_bridges}              │
│ 对话次数：{self.total_conversations}               │
│ 知识库：{kb.drawer_count} 抽屉 / {kb.total_entries} 条目  │
│ 工作桌：{self.desk.active_count} 张草稿（共 {self.desk.total_count}）        │
│ 短期记忆：{get_short_term_count()} 条                 │
│ 文件：{info['size_kb'] if info else 0} KB             │
│ 保存次数：{self.checkpoint.save_count}              │
│ 推理能量：{g.inference_energy.current:.0f}/{g.inference_energy.max_energy} │
│ 推理次数：{g.inference_energy.total_fires}               │
└────────────────────────────────────┘""")
        # 情境键状态
        sit_stats = get_situation_stats(g)
        if sit_stats:
            lines = []
            for name, s in sit_stats.items():
                icon = {2: "●", 1: "○", 0: "·"}[s["brightness"]]
                lines.append(f"│ {icon} {name}")
            if lines:
                print("┌─ 情境感知 ──────────────────────────┐")
                for l in lines:
                    print(l)
                print("└────────────────────────────────────┘")
        print()

    def _cmd_save(self):
        print("保存中……")
        self.checkpoint.save(self.garden, self.knowledge_base, self.desk)
        info = get_file_info(self.filepath)
        size = info["size_kb"] if info else 0
        print(f"保存好了师父。（{size} KB）")
        print()

    def _cmd_sleep(self):
        self._handle_shutdown()

    def _cmd_memory(self):
        """显示最近激活的键（跨所有盆）。"""
        all_keys = []
        for pot in self.garden.pots.values():
            for k in pot.network.all_keys():
                if k.last_activated_at > 0:
                    all_keys.append((k, pot.id, pot.topic_label))
        all_keys.sort(key=lambda x: x[0].last_activated_at, reverse=True)

        if not all_keys:
            print("小黑还什么都没记住呢。")
            print()
            return

        print("最近亮过的记忆：")
        for i, (k, pid, label) in enumerate(all_keys[:10]):
            ago = time.time() - k.last_activated_at
            if ago < 60:
                ago_str = f"{int(ago)}秒前"
            elif ago < 3600:
                ago_str = f"{int(ago / 60)}分钟前"
            elif ago < 86400:
                ago_str = f"{int(ago / 3600)}小时前"
            else:
                ago_str = f"{int(ago / 86400)}天前"

            brightness_icon = "●" * k.brightness + "○" * (2 - k.brightness)
            content = k.content[:50]
            if len(k.content) > 50:
                content += "…"
            print(f"  {i + 1}. [{brightness_icon}] {content}  "
                  f"[{label}] ({ago_str})")
        print()

    def _cmd_find(self, keyword: str):
        """搜索包含关键词的键（跨所有盆）。"""
        if not keyword:
            print("师父，要找什么？/find 后面加上关键词。")
            print()
            return

        results = []
        for pot in self.garden.pots.values():
            for k in pot.network.all_keys():
                if keyword in k.content:
                    results.append((k, pot.topic_label))

        if not results:
            print(f"没找到跟'{keyword}'有关的记忆。")
            print()
            return

        print(f"找到 {len(results)} 个跟'{keyword}'有关的记忆：")
        for i, (k, label) in enumerate(results[:20]):
            status = "💤" if k.is_dormant else "  "
            content = k.content[:60]
            if len(k.content) > 60:
                content += "…"
            print(f"  {status} [{k.id}] [{label}] {content}")
        if len(results) > 20:
            print(f"  ...还有 {len(results) - 20} 条")
        print()

    def _cmd_ingest(self):
        """扫描 data/ 目录并投喂新文件（花园模式）。"""
        self.ingest_queue.scan()
        if not self.ingest_queue.has_pending():
            print("师父，data/ 里没有新文件可以看。")
            print()
            return

        total = 0
        while self.ingest_queue.has_pending():
            filepath = self.ingest_queue.next()
            if filepath is None:
                break
            try:
                n = ingest_file_to_garden_v1(self.garden, filepath)
                total += n
                fname = os.path.basename(filepath)
                print(f"〔投喂〕{fname} → {n} 个新键")
            except Exception as e:
                print(f"〔投喂失败〕{filepath}: {e}")

        print(ingest_done_message(total))
        print()
        self.checkpoint.save(self.garden, self.knowledge_base, self.desk)

    def _cmd_backup(self):
        path = create_backup(self.filepath)
        if path:
            print(f"备份好了师父：{path}")
        else:
            print("还没有记忆文件，不需要备份。")
        print()

    def _cmd_keys(self):
        """列出所有键（按盆分组）。"""
        for pot in self.garden.pots.values():
            keys = pot.network.all_keys()
            if not keys:
                continue

            status_icon = "💤" if pot.is_dormant else "🌱"
            print(f"{status_icon} 盆 [{pot.topic_label}] ({pot.source_type}) "
                  f"共 {len(keys)} 键：")
            for k in keys:
                status = "💤" if k.is_dormant else "○"
                brightness = "●" * k.brightness + "○" * (2 - k.brightness)
                neighbors = len(pot.network.get_neighbors(k.id))
                content = k.content[:60]
                if len(k.content) > 60:
                    content += "…"
                print(f"  {status} [{k.id}] {brightness} 邻{neighbors:>3} | {content}")
        print()

    def _cmd_pots(self):
        """显示花盆列表。"""
        print(f"共 {len(self.garden.pots)} 个花盆：")
        for pot in self.garden.pots.values():
            status = "💤" if pot.is_dormant else "🌱"
            bright = len(pot.network.get_bright_keys())
            print(f"  {status} {pot.id} [{pot.topic_label}] "
                  f"来源={pot.source_type} "
                  f"键={pot.key_count} 亮={bright} "
                  f"线={pot.network.wire_count}")
        print(f"桥接总数：{self.garden.total_bridges}")
        print()

    def _cmd_bridges(self):
        """显示跨盆桥接。"""
        seen: set[tuple] = set()
        bridges = []
        for bridge_list in self.garden.bridge_table.values():
            for b in bridge_list:
                sig = (b.src_pot_id, b.dst_pot_id, b.src_key_id, b.dst_key_id)
                if sig not in seen:
                    seen.add(sig)
                    bridges.append(b)

        if not bridges:
            print("还没有跨盆桥接线。")
            print()
            return

        bridges.sort(key=lambda b: b.thickness, reverse=True)
        print(f"共 {len(bridges)} 条桥接线：")
        for b in bridges[:20]:
            src_pot = self.garden.pots.get(b.src_pot_id)
            dst_pot = self.garden.pots.get(b.dst_pot_id)
            src_label = src_pot.topic_label if src_pot else "?"
            dst_label = dst_pot.topic_label if dst_pot else "?"
            bar = "█" * min(int(b.thickness * 5), 5)
            print(f"  [{src_label}] → [{dst_label}] "
                  f"{bar} {b.thickness:.3f} ×{b.trigger_count}")
        if len(bridges) > 20:
            print(f"  ...还有 {len(bridges) - 20} 条")
        print()

    def _cmd_cases(self):
        """显示 CBR 案例库统计。"""
        stats = get_case_stats(self.garden)

        print(f"案例库：{stats['total']} 个案例 "
              f"（活跃 {stats['active']}，休眠 {stats['dormant']}，"
              f"待修正 {stats['needs_revision']}）")
        if stats['total'] > 0:
            print(f"  平均可靠性：{stats['avg_reliability']:.2f}")
            print(f"  累计节能：{stats['total_energy_saved']:.0f} 迭代")
        print()

        # 列出活跃案例
        active_cases = [
            c for c in self.garden.case_library.values()
            if not c.is_dormant
        ]
        if active_cases:
            active_cases.sort(key=lambda c: c.reliability, reverse=True)
            print("活跃案例：")
            for c in active_cases[:10]:
                trigger_preview = ", ".join(
                    str(t) for t in c.trigger_pattern[:3]
                )
                if len(c.trigger_pattern) > 3:
                    trigger_preview += f"… (+{len(c.trigger_pattern) - 3})"
                print(f"  Case#{c.id} triggers=[{trigger_preview}] "
                      f"snap={len(c.activation_snapshot)}keys "
                      f"rel={c.reliability:.2f} ×{c.frequency}")
            if len(active_cases) > 10:
                print(f"  ...还有 {len(active_cases) - 10} 个")
        print()

    # ── 文件柜命令 ──────────────────────────────

    def _cmd_remember(self, arg: str):
        """小黑主动把知识放进文件柜。"""
        if not arg:
            print("师父，要记住什么？用 /remember <内容>")
            print()
            return

        # 智能拆解：如果内容包含"｜"则前面是抽屉名
        if "｜" in arg:
            drawer, content = arg.split("｜", 1)
            drawer = drawer.strip()
            content = content.strip()
        else:
            drawer = ""
            content = arg.strip()

        entry = self.knowledge_base.file_entry(
            content=content,
            drawer_title=drawer,
            source_type="master_teaching",
        )

        if entry is not None:
            print(f"喵，记住了。放进 [{entry.drawer_title}] 抽屉里了。")
        else:
            print("师父，这个柜子里已经有差不多的了，我帮它记得更牢了。")
        print()

    def _cmd_kb(self, arg: str):
        """查看文件柜。"""
        kb = self.knowledge_base
        if kb.drawer_count == 0:
            print("文件柜还空着呢。师父教点什么吧。")
            print()
            return

        if arg:
            # 查看指定抽屉
            drawer = kb.find_drawer(arg)
            if drawer is None:
                drawer = kb.find_drawer(_infer_drawer_from_content(arg))
            if drawer is None:
                print(f"没有叫 '{arg}' 的抽屉。")
                all_titles = ", ".join(d.title for d in kb.drawers)
                print(f"柜子里有这些抽屉：{all_titles}")
                print()
                return

            print(f"📁 抽屉 [{drawer.title}]：{len(drawer.entries)} 条")
            for e in drawer.entries:
                source_icon = {"master_teaching": "👤师", "self_filed": "🐱黑",
                               "document": "📄档"}.get(e.source_type, "  ")
                accessed = ""
                if e.access_count > 0:
                    accessed = f" 查{e.access_count}次"
                print(f"  {source_icon} {e.content[:70]}{accessed}")
            print()
        else:
            # 总览
            print(f"📚 文件柜：{kb.drawer_count} 个抽屉，{kb.total_entries} 条知识")
            for d in kb.drawers:
                print(f"  📁 [{d.title}] {len(d.entries)} 条")
            print(f"\n用 /kb <抽屉名> 查看具体内容")
            print()

    # ── 工作桌命令 ──────────────────────────────

    def _cmd_write(self, arg: str):
        """师父在桌上放一张便条。"""
        if not arg:
            print("师父，要在桌上写什么？/write <内容>")
            print()
            return

        note = self.desk.add_note(arg, source="master")
        print(f"喵，放桌上了。便条 [{note.id}]。")
        print()

    def _cmd_desk(self):
        """看桌上摊着什么。"""
        active = self.desk.get_active_notes()
        all_notes = self.desk.get_all_notes()

        if not all_notes:
            print("桌子干干净净的，什么都没有。")
            print()
            return

        print(f"🪵 工作桌上{len(all_notes)}张草稿纸（{len(active)}张还在写）：")
        for note in all_notes:
            source_icon = {
                "master": "👤师",
                "xiaohei": "🐱黑",
                "ingest": "📄",
                "system": "⚙",
            }.get(note.source, "  ")
            done_mark = " ✓好了" if note.status == "done" else ""
            pin_mark = " 📌" if note.pinned else ""
            print(f"  {source_icon} {note}{done_mark}{pin_mark}")
        print()

    def _cmd_clear(self, arg: str):
        """扔掉草稿。不带参数 = 收走已完成的。"""
        if arg:
            # 指定 ID
            try:
                note_id = int(arg)
            except ValueError:
                print("师父，后面跟数字（草稿纸的编号），或者直接 /clear 清理已完成的。")
                print()
                return

            if self.desk.remove_note(note_id):
                print(f"扔掉了，便条 [{note_id}]。")
            else:
                print(f"桌上没有编号 [{note_id}] 的便条。")
        else:
            # 清理已完成的
            before = self.desk.total_count
            self.desk.clear_completed()
            removed = before - self.desk.total_count
            if removed > 0:
                print(f"收走了 {removed} 张打完勾的草稿，桌子清爽了一些。")
            else:
                print("桌上没有打完勾的草稿。用 /clear <编号> 来扔特定的一张。")
        print()

    # ── 盆拆分命令 ──────────────────────────────

    def _cmd_split(self, arg: str):
        """手动拆分花盆（按内容聚类）。"""
        # 默认拆最大的盆
        target_pot_id = arg.strip() if arg else ""

        if target_pot_id:
            if target_pot_id not in self.garden.pots:
                print(f"没有叫 '{target_pot_id}' 的盆。用 /pots 看看有哪些盆。")
                print()
                return
        else:
            # 找键数最多的盆
            best = None
            best_count = 0
            for pot in self.garden.pots.values():
                if pot.is_dormant or pot.source_type == "internal":
                    continue
                if pot.key_count > best_count:
                    best_count = pot.key_count
                    best = pot
            if best is None:
                print("没有可以拆分的盆。")
                print()
                return
            target_pot_id = best.id

        pot = self.garden.pots.get(target_pot_id)
        if pot is None:
            return

        if pot.key_count < 10:
            print(f"盆 [{pot.topic_label}] 只有 {pot.key_count} 个键，太少了不需要拆。")
            print()
            return

        # 决定拆成几个盆
        if pot.key_count > 200:
            num = 4
        elif pot.key_count > 100:
            num = 3
        else:
            num = 2

        print(f"正在把 [{pot.topic_label}]（{pot.key_count} 键）按内容拆分成 {num} 个盆……")
        sub_pots = self.garden.split_by_content(target_pot_id, num_clusters=num)

        if not sub_pots:
            print("拆分失败——键之间的内容太相似，分不开。")
        else:
            print(f"拆分完成：{len(sub_pots)} 个新盆：")
            for sp in sub_pots:
                print(f"  🌱 {sp.id} [{sp.topic_label}] {sp.key_count} 键")
        print()
        self.checkpoint.save(self.garden, self.knowledge_base, self.desk)

    # ── v1.0: 温度命令 ──────────────────────────

    def _cmd_temp(self):
        """显示 Copycat 温度和工作空间状态。"""
        from src.core.temperature import temperature_report
        print(temperature_report(self.garden))

        # 工作空间详情
        ws = self.garden.workspace
        if ws:
            print("工作空间中的键：")
            # 取痕迹最强的 10 个
            sorted_ws = sorted(ws.items(), key=lambda x: x[1], reverse=True)[:10]
            for key_id, strength in sorted_ws:
                # 找到键所属的盆
                found = None
                for pot in self.garden.pots.values():
                    k = pot.network.get_key(key_id)
                    if k is not None:
                        found = k
                        break
                if found:
                    bar = "#" * int(strength * 10) + "." * (10 - int(strength * 10))
                    preview = found.content[:40] + ("..." if len(found.content) > 40 else "")
                    prim = "[%s]" % found.primitive if found.primitive else ""
                    print("  [%s] %s %s %s" % (bar, prim, preview, "(休眠)" if found.is_dormant else ""))
        print()

    def _cmd_evidence(self):
        """显示 NARS 证据量和推理统计。"""
        from src.core.evidence import truth_expectation, evidence_label

        g = self.garden
        inf = g.inference_energy

        print(f"""
┌─ NARS 推理状态 ───────────────────────┐
│ 推理能量：{inf.current:.1f} / {inf.max_energy}           │
│ 累计消耗：{inf.total_spent:.1f}           │
│ 推理次数：{inf.total_fires}               │
│ 推理规则键：{len(g.rule_key_ids)} 个              │
└────────────────────────────────────┘
""")
        # 显示推理规则键
        internal = g.pots.get(g.internal_pot_id)
        if internal:
            print("推理规则键：")
            for rule_type, kid in g.rule_key_ids.items():
                rk = internal.network.get_key(kid)
                if rk:
                    icon = "●" if rk.brightness > 0 else "·"
                    exp = truth_expectation(rk.evidence_freq, rk.evidence_conf)
                    print(f"  {icon} [{rule_type}] {rk.content} exp={exp:.3f}")

        # 显示高 evidence 键（evidence_conf >= 0.15）
        high_ev_keys = []
        for pot in g.pots.values():
            if pot.is_dormant:
                continue
            for key in pot.network.all_keys():
                if key.evidence_conf >= 0.15 and not key.is_dormant:
                    exp = truth_expectation(key.evidence_freq, key.evidence_conf)
                    high_ev_keys.append((key, exp, pot.id))

        if high_ev_keys:
            high_ev_keys.sort(key=lambda x: x[1], reverse=True)
            print(f"\n高证据键（conf≥0.15，前 15 个）：")
            for key, exp, pid in high_ev_keys[:15]:
                label = evidence_label(key.evidence_freq, key.evidence_conf)
                prim = f"[{key.primitive}]" if key.primitive else ""
                preview = key.content[:35] + ("..." if len(key.content) > 35 else "")
                print(f"  ({key.evidence_freq:.2f},{key.evidence_conf:.2f}) exp={exp:.3f} {label} {prim} {preview}")
        print()

    # ── 主动询问（文件柜） ──────────────────────────

    def _check_filing_suggestion(self):
        """
        检查是否有键值得建议放入文件柜。
        条件：激活次数 > 20 且连接的键 ≥ 3。
        """
        SUGGEST_ACTIVATION_MIN = 20
        SUGGEST_NEIGHBORS_MIN = 3

        candidates = []
        for pot in self.garden.get_active_pots():
            for key in pot.network.all_keys():
                if key.is_dormant:
                    continue
                if key.source == "internal":
                    continue
                if key.activation_count < SUGGEST_ACTIVATION_MIN:
                    continue
                neighbors = pot.network.get_neighbors(key.id)
                if len(neighbors) < SUGGEST_NEIGHBORS_MIN:
                    continue
                # 检查柜子里是否已有相似内容
                existing = self.knowledge_base.find_similar(key.content, threshold=0.8)
                if existing is not None:
                    continue
                candidates.append((key, len(neighbors)))

        if not candidates:
            return None

        # 取激活次数最高且连接最多的
        candidates.sort(key=lambda x: (x[0].activation_count, x[1]), reverse=True)
        return candidates[0][0]

    # ── 文本投喂处理 ──────────────────────────────

    def _classify_ingest_content(self, text: str) -> str:
        """
        判断投喂内容是叙事/对话（→记忆）还是参考资料（→文件柜）。

        参考资料特征：
        - 结构化列表（数字编号、破折号列表）
        - 定义/公式密集
        - 短行多（每行 < 30 字且换行频繁）
        - 无第一人称叙事标记

        返回 "memory" 或 "knowledge"。
        """
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return "memory"

        total_chars = len(text)
        line_count = len(lines)
        avg_line_len = total_chars / max(line_count, 1)

        # 特征检测
        numbered_lines = sum(1 for l in lines if l and (l[0].isdigit() or l.startswith("【")))
        bullet_lines = sum(1 for l in lines if l.startswith("  ") or l.startswith("  -"))
        short_lines = sum(1 for l in lines if len(l) < 30)
        has_first_person = any(
            word in text for word in ["我", "师父", "小黑", "他对我", "我叫"]
        )
        has_definition_pattern = any(
            marker in text for marker in [
                "是指", "定义", "规则", "用法", "公式", "类型", "步骤",
                "声母", "韵母", "笔画", "修辞", "要素", "体裁",
            ]
        )

        # 判定
        ref_score = 0
        if numbered_lines > line_count * 0.3:
            ref_score += 3
        if bullet_lines > 2:
            ref_score += 2
        if avg_line_len < 30 and line_count > 10:
            ref_score += 2
        if has_definition_pattern:
            ref_score += 3
        if short_lines > line_count * 0.6:
            ref_score += 2

        narr_score = 0
        if has_first_person:
            narr_score += 3
        if avg_line_len > 50:
            narr_score += 2

        if ref_score > narr_score:
            return "knowledge"
        return "memory"

    def _handle_ingest_text(self, text: str):
        """处理 [投喂] 标记的内联文本（花园模式 + 智能路由）。"""
        if not text:
            print("师父，投喂什么内容？")
            print()
            return

        # 判断去向
        route = self._classify_ingest_content(text)

        if route == "knowledge":
            # ── 投喂到文件柜 ──
            # 尝试智能切分为多个条目
            entries_added = 0
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            # 如果行很多且短 → 按主题段落分组
            if len(lines) > 5:
                # 整段作为一个知识条目
                entry = self.knowledge_base.file_entry(
                    content=text,
                    source_type="document",
                )
                if entry is not None:
                    entries_added = 1
                    print(f"〔文件柜〕放进 [{entry.drawer_title}] 抽屉了。")
                else:
                    print("师父，柜子里已经有差不多的内容了，我帮它记得更牢了。")
            else:
                for line in lines:
                    if len(line) < 3:
                        continue
                    entry = self.knowledge_base.file_entry(
                        content=line,
                        source_type="document",
                    )
                    if entry is not None:
                        entries_added += 1

            if entries_added > 0:
                print(f"〔文件柜〕共 {entries_added} 条知识入库。")
            print()
            self.checkpoint.save(self.garden, self.knowledge_base, self.desk)
            return

        # ── v1.0 CD 拆解投喂 ──
        new_keys, pot_id, is_new_pot, clusters = ingest_document_to_garden_v1(
            self.garden, text
        )

        if is_new_pot:
            print(f"〔新盆〕这个话题小黑还没学过，建了一个新盆。")
        print(ingest_done_message(len(new_keys)))
        if new_keys:
            # 显示概念键概览
            previews = []
            for k in new_keys[:5]:
                prim_tag = f"[{k.primitive}]" if k.primitive else ""
                preview = (k.content[:20] + "…") if len(k.content) > 20 else k.content
                previews.append(f"{prim_tag}{preview}")
            print(f"〔概念键〕{', '.join(previews)}")
            if len(new_keys) > 5:
                print(f"  ...还有 {len(new_keys) - 5} 个")
        print()
        self.checkpoint.save(self.garden, self.knowledge_base, self.desk)

    # ── 后台任务 ────────────────────────────────

    def _background_tasks(self):
        """对话间隙执行后台维护。含夜间自动入睡检测。"""
        cleanup_short_term()

        now = time.time()

        # ── 夜间自动入睡检测 ──
        self._check_auto_sleep(now)

        if now - self.last_dormancy_check > DORMANCY_CHECK_INTERVAL:
            # 花园级维护（含盆拆分/合并/休眠 + 桥接衰减）
            self.garden.maintenance()
            # 各盆内网络维护
            for pot in self.garden.pots.values():
                if not pot.is_dormant:
                    background_maintenance(pot.network)
            self.knowledge_base.maintenance()
            self.desk.auto_cleanup()
            self.last_dormancy_check = now

    # ── 自动入睡检测 ──────────────────────────

    def _check_auto_sleep(self, now: float):
        """
        检查是否应该自动入睡。

        条件：
        - 在睡眠窗口内（22:00 ~ 7:00）
        - 师父超过 N 分钟没说话
        - 疲劳值超过最低自动入睡阈值

        这不是"定时器强制睡觉"——是小黑真的累了，天也黑了，
        他自己决定去床上。
        """
        if self.last_conversation_at == 0:
            return

        from datetime import datetime
        hour = datetime.now().hour
        idle_minutes = (now - self.last_conversation_at) / 60.0

        # 检查是否在睡眠时间窗口
        p = self.personality
        in_sleep_window = (
            hour >= p.auto_sleep_start_hour
            or hour < p.auto_sleep_end_hour
        )

        if not in_sleep_window:
            return

        # 空闲时间够长
        if idle_minutes < p.auto_sleep_idle_minutes:
            return

        # 疲劳值够高
        if self.fatigue < p.auto_sleep_min_fatigue:
            return

        # 小黑决定去睡觉
        print()
        if self.fatigue > self.personality.fatigue_very_tired_threshold:
            print("小黑非常累了，决定去床上睡觉。")
        else:
            print("夜深了，小黑自己去床上了。")
        self._handle_shutdown()
        # _handle_shutdown 会设置 self.running = False，主循环退出

    # ── 残留亮度管理 ──────────────────────────

    RESIDUAL_TTL = 120
    RESIDUAL_COOL_PERIOD = 30

    def _extract_concept_key_pot_map(
        self,
        result: GardenConductionResult,
    ) -> dict[int, str]:
        """
        从传导结果中提取非 OUTPUT 原语的亮键 → 所在盆 ID 的映射。

        用于语法桥接：找出哪些概念键参与了当前传导，
        以便在概念键和 OUTPUT 键之间建立/加粗桥接线。
        """
        concept_map: dict[int, str] = {}

        for round_name in ('round1', 'round3'):
            round_data = getattr(result, round_name, {})
            for pot_id, cr in round_data.items():
                pot = self.garden.pots.get(pot_id)
                if pot is None:
                    continue
                for kid, brightness in cr.final_pattern.items():
                    if brightness < 1:
                        continue
                    key = pot.network.get_key(kid)
                    if key is None:
                        continue
                    if key.primitive == "OUTPUT":
                        continue
                    concept_map[kid] = pot_id

        return concept_map

    def _store_grammar_pattern(
        self,
        result: GardenConductionResult,
        response: str,
        output_regulation=None,
    ):
        """
        v1.0: 从本轮对话中提取激活签名并存储语法模式。

        仅在语法键未命中（降级到路径生成）时创建新模式。
        已命中的模式由 _try_grammar_generation 内部调用 record_success。
        """
        from src.core.grammar import (
            compute_activation_signature,
            store_output_pattern,
        )

        signature = compute_activation_signature(result, self.garden)
        if not signature or len(signature) < 2:
            return

        tone = "normal"
        if output_regulation is not None:
            tone = getattr(output_regulation, 'tone', 'normal')

        # 拆解响应为句子（用于模板提取）
        raw_sentences = _split_response_to_sentences(response)
        if not raw_sentences:
            return

        # 提取参与的概念键（用于创建桥接线）
        concept_key_to_pot = self._extract_concept_key_pot_map(result)

        # store_output_pattern 内部会做去重检查（Jaccard ≥ 0.8 则合并）
        store_output_pattern(
            self.garden,
            signature,
            raw_sentences,
            tone=tone,
            concept_key_to_pot=concept_key_to_pot,
        )

    def _extract_response_key_ids(
        self,
        response: str,
        result: GardenConductionResult,
    ) -> list[int]:
        """
        从响应文本中提取涉及的键 ID。

        通过匹配响应中的子串与传导结果中激活键的 content，
        找出哪些键参与了输出。
        """
        matched: list[tuple[int, int]] = []  # [(key_id, match_length)]

        for pot_id, cr in result.round1.items():
            for kid in cr.final_pattern:
                pot = self.garden.pots.get(pot_id)
                if pot is None:
                    continue
                key = pot.network.get_key(kid)
                if key is None:
                    continue
                # 检查键内容是否在响应中出现
                if len(key.content) >= 2 and key.content in response:
                    matched.append((kid, len(key.content)))

        for pot_id, cr in result.round3.items():
            for kid in cr.final_pattern:
                pot = self.garden.pots.get(pot_id)
                if pot is None:
                    continue
                key = pot.network.get_key(kid)
                if key is None:
                    continue
                if len(key.content) >= 2 and key.content in response:
                    matched.append((kid, len(key.content)))

        # 去重，按匹配长度降序排列（更长的匹配更相关）
        seen: set[int] = set()
        unique: list[int] = []
        for kid, _ in sorted(matched, key=lambda x: x[1], reverse=True):
            if kid not in seen:
                seen.add(kid)
                unique.append(kid)

        return unique

    def _set_residuals(self, result: GardenConductionResult):
        """跨盆传导结束后，给所有盆的亮键打残留标记。"""
        now = time.time()
        deadline = now + self.RESIDUAL_TTL

        for pot_id, cr in result.round1.items():
            for k in cr.top_keys:
                if k.brightness >= 2:
                    k.residual_until = deadline
                elif k.brightness == 1 and k.activation_count >= 2:
                    k.residual_until = deadline

    def _cool_residuals(self):
        """每回合开始前，清除上一轮所有键的亮度。上下文维持交给工作空间预热。"""
        for pot in self.garden.pots.values():
            if pot.is_dormant:
                continue
            for key in pot.network.get_bright_keys():
                key.brightness = 0

    # ── 关闭 ────────────────────────────────────

    def _handle_shutdown(self):
        """优雅关闭。"""
        self.running = False
        print()
        print("保存记忆中……")
        self.checkpoint.save(self.garden, self.knowledge_base, self.desk)
        print(shutdown_message())


# ═══════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════

def main():
    """程序入口。"""
    filepath = DEFAULT_FILE
    data_dir = DATA_DIR

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--file" and i + 1 < len(args):
            filepath = args[i + 1]
            i += 2
        elif args[i] == "--data" and i + 1 < len(args):
            data_dir = args[i + 1]
            i += 2
        elif args[i] == "--new":
            if os.path.exists(filepath):
                backup = create_backup(filepath)
                print(f"〔备份旧文件〕{backup}")
                os.remove(filepath)
            i += 1
        elif args[i] == "--info":
            info = get_file_info(filepath)
            if info:
                print(f"文件：{filepath}")
                print(f"版本：{info['version']}")
                print(f"键：{info['key_count']}")
                print(f"电线：{info['wire_count']}")
                if info.get('pot_count', 1) > 1:
                    print(f"花盆：{info['pot_count']}")
                    print(f"桥接：{info.get('bridge_count', 0)}")
                print(f"迭代：{info['total_iterations']}")
                print(f"知识库：{info['kb_drawers']} 抽屉 / {info['kb_entries']} 条目")
                print(f"大小：{info['size_kb']} KB")
                saved = time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(info['saved_at']) if info['saved_at'] else time.localtime(),
                )
                print(f"保存时间：{saved}")
            else:
                print(f"文件不存在：{filepath}")
            return
        elif args[i] == "--help" or args[i] == "-h":
            print("罗小黑 键盘模型 AI  v0.3 花盆模型")
            print()
            print("用法：python main.py [选项]")
            print()
            print("选项：")
            print("  --file PATH   指定记忆文件（默认 luoxiaohei.luoxiaohei）")
            print("  --data DIR    指定投喂目录（默认 data/）")
            print("  --new         忽略已有记忆，重新开始")
            print("  --info        显示记忆文件信息后退出")
            print("  --help, -h    显示此帮助")
            return
        else:
            i += 1

    runtime = LuoXiaoheiRuntime(filepath=filepath, data_dir=data_dir)
    runtime.run()


if __name__ == "__main__":
    main()

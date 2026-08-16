"""
键盘模型 AI — 知识库模块（文件柜）

小黑的文件柜。存放硬知识——公式、代码、数据、规格、参考资料。
不存记忆、对话、情感、人物。那些是记忆网络（大脑）的事。

小黑主动决定往柜子里放什么，不是自动的。
师父也可以说"记住这个"，小黑就把它整理好放进去。

文件柜 = 工具（在房间里，不在小黑脑子里）
记忆网络 = 大脑
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.key import Key


# ═══════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════

@dataclass
class Entry:
    """文件柜里的一条知识。"""

    id: int
    content: str              # 知识内容（公式、代码、数据、规格等）
    drawer_title: str = ""    # 归属抽屉名
    source_type: str = "master_teaching"  # "master_teaching" | "self_filed" | "document"
    source_keys: list[int] = field(default_factory=list)  # 关联记忆键的 ID（可为空）
    tags: list[str] = field(default_factory=list)         # e.g. ["formula", "python"]
    confidence: float = 1.0   # 文件柜条目默认为确信知识（不是推测，是记录）
    created_at: float = field(default_factory=time.time)
    access_count: int = 0     # 被查询命中的次数
    last_accessed_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "drawer_title": self.drawer_title,
            "source_type": self.source_type,
            "source_keys": self.source_keys,
            "tags": self.tags,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "last_accessed_at": self.last_accessed_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Entry":
        return cls(
            id=d["id"],
            content=d["content"],
            drawer_title=d.get("drawer_title", ""),
            source_type=d.get("source_type", "master_teaching"),
            source_keys=d.get("source_keys", []),
            tags=d.get("tags", []),
            confidence=d.get("confidence", 1.0),
            created_at=d.get("created_at", time.time()),
            access_count=d.get("access_count", 0),
            last_accessed_at=d.get("last_accessed_at", 0.0),
        )


@dataclass
class Drawer:
    """知识库中的一个抽屉（主题分类）。"""
    id: int
    title: str                # 抽屉标题，如"关于师父"、"关于无限"
    entries: list[Entry] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "entries": [e.to_dict() for e in self.entries],
            "created_at": self.created_at,
            "last_updated_at": self.last_updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Drawer":
        drawer = cls(
            id=d["id"],
            title=d["title"],
            created_at=d.get("created_at", time.time()),
            last_updated_at=d.get("last_updated_at", time.time()),
        )
        drawer.entries = [Entry.from_dict(e) for e in d.get("entries", [])]
        return drawer


# ═══════════════════════════════════════════════
# 知识库
# ═══════════════════════════════════════════════

class KnowledgeBase:
    """抽屉式知识库。"""

    def __init__(self):
        self.drawers: list[Drawer] = []
        self.next_entry_id: int = 1
        self.next_drawer_id: int = 1

    # ── 抽屉操作 ──────────────────────────────

    def create_drawer(self, title: str) -> Drawer:
        """新建抽屉。同名抽屉会返回已有抽屉。"""
        existing = self.find_drawer(title)
        if existing:
            return existing
        drawer = Drawer(id=self.next_drawer_id, title=title)
        self.drawers.append(drawer)
        self.next_drawer_id += 1
        return drawer

    def find_drawer(self, title: str) -> Drawer | None:
        """按标题查找抽屉（精确匹配）。"""
        for d in self.drawers:
            if d.title == title:
                return d
        return None

    def find_or_create_drawer(self, title: str) -> Drawer:
        """查找抽屉，找不到就创建。"""
        return self.find_drawer(title) or self.create_drawer(title)

    # ── 条目操作 ──────────────────────────────

    def add_entry(
        self,
        drawer_title: str,
        content: str,
        source_type: str = "master_teaching",
        source_keys: list[int] | None = None,
        tags: list[str] | None = None,
        confidence: float = 1.0,
    ) -> Entry:
        """向指定抽屉添加一条知识。默认来源为师父教学。"""
        drawer = self.find_or_create_drawer(drawer_title)
        entry = Entry(
            id=self.next_entry_id,
            content=content,
            drawer_title=drawer_title,
            source_type=source_type,
            source_keys=source_keys or [],
            tags=tags or [],
            confidence=confidence,
        )
        drawer.entries.append(entry)
        drawer.last_updated_at = time.time()
        self.next_entry_id += 1
        return entry

    def file_entry(
        self,
        content: str,
        drawer_title: str = "",
        tags: list[str] | None = None,
        source_type: str = "self_filed",
        key_refs: list[int] | None = None,
    ) -> Entry | None:
        """
        小黑主动往文件柜里放东西。

        先去重——如果柜子里已经有相似度 > 0.85 的内容，不重复放，
        改为增加原条目的确信度。

        返回新条目（或 None 表示已存在不重复放）。
        """
        # 智能推断抽屉
        if not drawer_title:
            drawer_title = _infer_drawer_from_content(content)

        # 去重
        existing = self.find_similar(content, threshold=0.85)
        if existing is not None:
            existing.confidence = min(1.0, existing.confidence + 0.05)
            existing.access_count += 1
            existing.last_accessed_at = time.time()
            return None  # 不重复，但更新了已有条目

        return self.add_entry(
            drawer_title=drawer_title,
            content=content,
            source_type=source_type,
            source_keys=key_refs or [],
            tags=tags or _infer_tags(content),
        )

    def remove_entry(self, entry_id: int):
        """删除一条结论。"""
        for drawer in self.drawers:
            for entry in drawer.entries:
                if entry.id == entry_id:
                    drawer.entries.remove(entry)
                    return

    # ── 查询 ──────────────────────────────────

    def query(
        self,
        bright_keys: list = None,
        query_text: str = "",
        max_results: int = 5,
    ) -> list[Entry]:
        """
        查询知识库中相关的内容。

        两种查询方式：
        1. 通过 bright_keys 的 source_keys 反查（直接关联，权重高）
        2. 通过 query_text 与条目内容的文本重叠（间接关联，权重低）

        返回去重并按权重排序的结果。
        """
        bright_keys = bright_keys or []
        key_ids = set(k.id for k in bright_keys)
        scored: dict[int, tuple[Entry, float]] = {}

        for drawer in self.drawers:
            for entry in drawer.entries:
                score = 0.0

                # 方法 1：source_keys 关联
                overlap = key_ids & set(entry.source_keys)
                if overlap:
                    score = 0.8 * (len(overlap) / len(entry.source_keys))

                # 方法 2：文本重叠
                if query_text:
                    text_score = _text_overlap_score(query_text, entry.content)
                    score = max(score, text_score * 0.5)

                if score > 0.2:
                    if entry.id not in scored or score > scored[entry.id][1]:
                        scored[entry.id] = (entry, score)

        # 排序 + 更新访问计数
        results = sorted(scored.values(), key=lambda x: x[1], reverse=True)
        now = time.time()
        for entry, _ in results[:max_results]:
            entry.access_count += 1
            entry.last_accessed_at = now

        return [entry for entry, _ in results[:max_results]]

    def find_similar(self, text: str, threshold: float = 0.7) -> Entry | None:
        """查找与给定文本最相似的已有条目。用于去重。"""
        best = None
        best_score = 0.0
        for drawer in self.drawers:
            for entry in drawer.entries:
                score = _text_overlap_score(text, entry.content)
                if score > best_score:
                    best_score = score
                    best = entry
        if best_score > threshold and best is not None:
            return best
        return None

    # ── 后台维护 ──────────────────────────────

    def maintenance(self):
        """空闲时调用：合并重复、清理低质量条目。"""
        self._merge_duplicate_entries()
        self._clean_low_confidence()

    def _merge_duplicate_entries(self):
        """合并内容高度相似的条目。"""
        for drawer in self.drawers:
            merged = []
            skip = set()
            for i, e1 in enumerate(drawer.entries):
                if i in skip:
                    continue
                for j, e2 in enumerate(drawer.entries):
                    if j <= i or j in skip:
                        continue
                    if _text_overlap_score(e1.content, e2.content) > 0.9:
                        # 保留确信度更高的
                        if e1.confidence >= e2.confidence:
                            e1.access_count += e2.access_count
                            skip.add(j)
                        else:
                            e2.access_count += e1.access_count
                            skip.add(i)
                            break
                if i not in skip:
                    merged.append(e1)
            drawer.entries = merged

    def _clean_low_confidence(self):
        """清理确信度极低且从未被查询的旧条目。"""
        now = time.time()
        for drawer in self.drawers:
            drawer.entries = [
                e for e in drawer.entries
                if not (
                    now - e.created_at > 30 * 86400
                    and e.confidence < 0.2
                    and e.access_count == 0
                )
            ]

    # ── 统计 ──────────────────────────────────

    @property
    def total_entries(self) -> int:
        return sum(len(d.entries) for d in self.drawers)

    @property
    def drawer_count(self) -> int:
        return len(self.drawers)

    # ── 持久化 ────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "drawers": [d.to_dict() for d in self.drawers],
            "next_entry_id": self.next_entry_id,
            "next_drawer_id": self.next_drawer_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeBase":
        kb = cls()
        kb.drawers = [Drawer.from_dict(dd) for dd in d.get("drawers", [])]
        kb.next_entry_id = d.get("next_entry_id", 1)
        kb.next_drawer_id = d.get("next_drawer_id", 1)
        return kb

    def __repr__(self) -> str:
        return f"KnowledgeBase(drawers={self.drawer_count}, entries={self.total_entries})"


# ═══════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════

def _text_overlap_score(a: str, b: str) -> float:
    """两个文本的重叠度（max(Jaccard, containment)，基于字符 2-gram）。"""
    if not a or not b:
        return 0.0
    grams_a = set(a[i:i + 2] for i in range(len(a) - 1))
    grams_b = set(b[i:i + 2] for i in range(len(b) - 1))
    if not grams_a or not grams_b:
        return 0.0

    intersection = len(grams_a & grams_b)
    jaccard = intersection / len(grams_a | grams_b)
    containment = intersection / min(len(grams_a), len(grams_b))

    return max(jaccard, containment)


def _infer_drawer_from_content(content: str) -> str:
    """从内容自动推断抽屉名。"""
    # 关键词 → 抽屉映射
    KEYWORD_MAP: dict[str, str] = {
        "Python": "编程",
        "python": "编程",
        "代码": "编程",
        "函数": "编程",
        "类": "编程",
        "import": "编程",
        "def ": "编程",
        "公式": "数学",
        "定理": "数学",
        "方程": "数学",
        "证明": "数学",
        "数据": "数据",
        "表格": "数据",
        "统计": "数据",
        "SQL": "数据",
        "sql": "数据",
        "算法": "算法",
        "排序": "算法",
        "搜索": "算法",
        "http": "网络",
        "HTTP": "网络",
        "API": "网络",
        "api": "网络",
        "协议": "网络",
        "Linux": "系统",
        "linux": "系统",
        "命令": "系统",
        "配置": "系统",
        "安装": "系统",
    }

    for keyword, drawer in KEYWORD_MAP.items():
        if keyword in content:
            return drawer

    # 没有匹配的关键词 → 取内容前几个字
    short = content[:12].rstrip("，。！？、")
    return f"关于{short}"


def _infer_tags(content: str) -> list[str]:
    """从内容推断标签。"""
    tags: list[str] = []

    TAG_HINTS: dict[str, str] = {
        "公式": "formula",
        "定理": "formula",
        "代码": "code",
        "范例": "example",
        "例子": "example",
        "定义": "definition",
        "规格": "spec",
        "参考": "reference",
        "命令": "command",
        "配置": "config",
    }

    for keyword, tag in TAG_HINTS.items():
        if keyword in content:
            tags.append(tag)

    return list(set(tags))  # 去重


def infer_drawer_title_from_keys(keys: list) -> str:
    """
    根据键集群推断抽屉标题（保留旧接口兼容）。
    """
    if not keys:
        return "未分类"
    content = keys[0].content if hasattr(keys[0], 'content') else str(keys[0])
    return _infer_drawer_from_content(content)

"""
键盘模型 AI — 工作桌（草稿纸）

小黑的工作桌。桌子上放着草稿纸——正在写的东西、
琢磨到一半的想法、读文档时的笔记、想不通时的涂鸦。

和文件柜（隔壁）的区别：
- 工作桌 = 临时的、在写的、没归档的。草稿纸，可以扔。
- 文件柜 = 整理好的、查资料的。写好了放进去的。

小黑自己往桌上写东西不需要师父管。
师父可以用指令看一眼桌上有什么、帮小黑把没用的扔掉。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════
# 草稿纸
# ═══════════════════════════════════════════════

@dataclass
class ScratchNote:
    """桌上的一张草稿纸。"""

    id: int
    content: str
    source: str = "master"        # "master" | "xiaohei" | "ingest" | "system"
    status: str = "drafting"      # "drafting" | "done"
    created_at: float = field(default_factory=time.time)
    pinned: bool = False           # 钉在桌上，自动清理时不会扔

    def mark_done(self):
        """标注为完成（下次清理时会收走）。"""
        self.status = "done"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "status": self.status,
            "created_at": self.created_at,
            "pinned": self.pinned,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScratchNote":
        return cls(
            id=d["id"],
            content=d["content"],
            source=d.get("source", "master"),
            status=d.get("status", "drafting"),
            created_at=d.get("created_at", time.time()),
            pinned=d.get("pinned", False),
        )

    def __repr__(self) -> str:
        pin = "📌" if self.pinned else "📝"
        done = " ✓" if self.status == "done" else ""
        preview = self.content[:40] + ("…" if len(self.content) > 40 else "")
        return f"{pin} [{self.id}]{done} {preview}"


# ═══════════════════════════════════════════════
# 工作桌
# ═══════════════════════════════════════════════

class Desk:
    """小黑的工作桌。管理桌上的草稿纸。"""

    def __init__(self):
        self.notes: list[ScratchNote] = []
        self.next_id: int = 1

    # ── 放草稿纸 ──────────────────────────────

    def add_note(
        self,
        content: str,
        source: str = "master",
        pinned: bool = False,
    ) -> ScratchNote:
        """放一张草稿纸到桌上。"""
        note = ScratchNote(
            id=self.next_id,
            content=content,
            source=source,
            pinned=pinned,
        )
        self.notes.append(note)
        self.next_id += 1
        return note

    # ── 扔掉 ──────────────────────────────────

    def remove_note(self, note_id: int) -> bool:
        """扔掉桌上一张草稿纸。"""
        for i, note in enumerate(self.notes):
            if note.id == note_id:
                self.notes.pop(i)
                return True
        return False

    # ── 查看 ──────────────────────────────────

    def get_active_notes(self) -> list[ScratchNote]:
        """桌上还在写的东西（未完成的草稿）。"""
        return [n for n in self.notes if n.status == "drafting"]

    def get_all_notes(self) -> list[ScratchNote]:
        """桌上所有草稿纸（含已完成的）。"""
        return list(self.notes)

    def get_note(self, note_id: int) -> ScratchNote | None:
        """按 ID 找一张草稿纸。"""
        for note in self.notes:
            if note.id == note_id:
                return note
        return None

    # ── 清理 ──────────────────────────────────

    def clear_completed(self):
        """把打完勾的草稿收走（不钉住的）。已完成的钉住草稿保留。"""
        self.notes = [
            n for n in self.notes
            if not (n.status == "done" and not n.pinned)
        ]

    def clear_all_completed(self):
        """全部已完成的都收走（含钉住的）。"""
        self.notes = [n for n in self.notes if n.status != "done"]

    def auto_cleanup(self, max_age_days: int = 7):
        """
        自动清理：超过 N 天的非钉住草稿，默默收掉。
        小黑不主动扔桌上的东西，但太久远的会自己消失。
        """
        now = time.time()
        cutoff = now - max_age_days * 86400
        self.notes = [
            n for n in self.notes
            if n.pinned or n.created_at > cutoff
        ]

    # ── 小黑自己写 ────────────────────────────

    def xiaohei_jot(
        self,
        content: str,
        pinned: bool = False,
    ) -> ScratchNote:
        """小黑自己在桌上记点东西。"""
        return self.add_note(content, source="xiaohei", pinned=pinned)

    # ── 统计 ──────────────────────────────────

    @property
    def active_count(self) -> int:
        return len(self.get_active_notes())

    @property
    def total_count(self) -> int:
        return len(self.notes)

    # ── 持久化 ────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "notes": [n.to_dict() for n in self.notes],
            "next_id": self.next_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Desk":
        desk = cls()
        desk.notes = [ScratchNote.from_dict(nd) for nd in d.get("notes", [])]
        desk.next_id = d.get("next_id", 1)
        return desk

    def __repr__(self) -> str:
        return f"Desk(active={self.active_count}, total={self.total_count})"

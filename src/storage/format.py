"""
键盘模型 AI — 存储格式模块

.luoxiaohei 文件的读写。原型阶段使用 JSON 格式（规格书 §08-九 允许），
后期 Unikernel 阶段迁移为二进制格式。

一个 .luoxiaohei 文件 = 一个完整的 AI 生命记录。
包含：键网络 + 知识库 + 统计 + 时间戳。

English:
Storage format: reads and writes .luoxiaohei files (JSON during the
prototype phase, binary in the later Unikernel stage). One file is a
complete AI life record: key network + knowledge base + stats + timestamps.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.network import KeyNetwork
    from src.core.garden import Garden
    from src.knowledge.kb import KnowledgeBase


# ── 魔数（JSON 模式下用版本字段替代） ──────────

FILE_VERSION = 3  # v1.0: 新增 CD 原语/NARS证据量/HTM预测/Copycat滑移
FILE_EXTENSION = ".luoxiaohei"



# ═══════════════════════════════════════════════
# 加载
# ═══════════════════════════════════════════════

def load_checkpoint(
    filepath: str = "luoxiaohei.luoxiaohei",
) -> tuple:
    """
    从 .luoxiaohei 文件加载检查点。兼容 v0.2 旧格式。

    返回: (Garden, KnowledgeBase, Desk, metadata)
    文件不存在 → 返回全新的空花园。
    """
    from src.core.network import KeyNetwork
    from src.core.garden import Garden
    from src.knowledge.kb import KnowledgeBase
    from src.knowledge.desk import Desk

    if not os.path.exists(filepath):
        garden = Garden()
        kb = KnowledgeBase()
        desk = Desk()
        return garden, kb, desk, {"is_new": True, "filepath": filepath}

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    version = data.get("version", 0)
    if version > FILE_VERSION:
        raise ValueError(
            f"文件版本 {version} 高于当前支持版本 {FILE_VERSION}，"
            f"请升级程序。"
        )

    # ── v0.3+: 从 Garden 快照恢复 ─────────────────
    if "garden" in data:
        garden = Garden.from_snapshot(data["garden"])
    elif "network" in data:
        # ── v0.2 迁移：将旧网络包装为花园 ─────────
        network = KeyNetwork.from_snapshot(data["network"])
        garden = _migrate_network_to_garden(network)
    else:
        garden = Garden()

    # 恢复知识库
    kb_data = data.get("knowledge_base")
    if kb_data:
        knowledge_base = KnowledgeBase.from_dict(kb_data)
    else:
        knowledge_base = KnowledgeBase()

    # 恢复工作桌
    from src.knowledge.desk import Desk
    desk_data = data.get("desk")
    if desk_data:
        desk = Desk.from_dict(desk_data)
    else:
        desk = Desk()

    metadata = {
        "is_new": False,
        "filepath": filepath,
        "version": version,
        "saved_at": data.get("saved_at", 0),
        "key_count": garden.total_keys,
        "wire_count": garden.total_wires,
        "pot_count": len(garden.pots),
    }

    return garden, knowledge_base, desk, metadata


def _migrate_network_to_garden(network: "KeyNetwork") -> "Garden":
    """
    将 v0.2 的旧网络迁移为花园格式。

    旧网络的所有键 → 一个"默认"盆（dialogue 类型）。
    内部键（source="internal"的）→ 移到内部盆。
    """
    from src.core.garden import Garden
    from src.core.network import KeyNetwork

    garden = Garden()

    # 创建内部盆
    internal = garden.create_pot(
        source_type="internal",
        topic_label="系统",
        is_internal=True,
    )
    garden.internal_pot_id = internal.id

    # 创建默认盆（对话）
    default = garden.create_pot(
        source_type="dialogue",
        topic_label="日常对话（迁移）",
    )

    # 分配键：internal 键 → 内部盆，其余 → 默认盆
    internal_net = internal.network
    default_net = default.network

    internal_count = 0
    default_count = 0

    for key in network.all_keys():
        if key.source == "internal":
            target_net = internal_net
            internal_count += 1
        else:
            target_net = default_net
            default_count += 1

        new_key = target_net.create_key(
            content=key.content,
            source=key.source,
            brightness=0,
        )
        new_key.activation_count = key.activation_count
        new_key.activation_history = key.activation_history.copy()
        new_key.last_activated_at = key.last_activated_at
        new_key.peak_brightness = key.peak_brightness
        # 存储映射
        key._migrated_id = new_key.id
        key._migrated_pot_id = target_net is internal_net and internal.id or default.id

    # 迁移电线
    for (a, b), wire in network.wires.items():
        key_a = network.get_key(a)
        key_b = network.get_key(b)
        if key_a is None or key_b is None:
            continue
        ma = getattr(key_a, '_migrated_id', None)
        mb = getattr(key_b, '_migrated_id', None)
        pa = getattr(key_a, '_migrated_pot_id', None)
        pb = getattr(key_b, '_migrated_pot_id', None)
        if ma is None or mb is None:
            continue

        if pa == pb:
            # 同盆 → 盆内电线
            target_net = garden.pots[pa].network
            target_net.set_wire(ma, mb, thickness=wire.thickness)
        else:
            # 跨盆 → 桥接线
            garden.create_bridge(
                src_key_id=ma, src_pot_id=pa,
                dst_key_id=mb, dst_pot_id=pb,
                thickness=wire.thickness,
            )

    default.calculate_signature()
    default.topic_label = "日常对话"

    # 迁移后立即保存一次（下次加载就是 v0.3 格式）
    print(f"〔迁移完成〕{default_count} 键 → 对话盆，{internal_count} 键 → 系统盆")
    return garden


def save_garden_checkpoint(
    garden: "Garden",
    knowledge_base: "KnowledgeBase | None" = None,
    desk = None,
    filepath: str = "luoxiaohei.luoxiaohei",
) -> str:
    """
    保存花园检查点到 .luoxiaohei 文件（v0.3 格式）。

    返回保存的文件路径。
    """
    checkpoint: dict = {
        "version": FILE_VERSION,
        "saved_at": time.time(),
        "garden": garden.snapshot(),
    }

    if knowledge_base is not None:
        checkpoint["knowledge_base"] = knowledge_base.to_dict()

    if desk is not None:
        checkpoint["desk"] = desk.to_dict()

    # 原子写入
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    if os.path.exists(filepath):
        os.replace(tmp_path, filepath)
    else:
        os.rename(tmp_path, filepath)

    return filepath


# ═══════════════════════════════════════════════
# 文件信息
# ═══════════════════════════════════════════════

def get_file_info(filepath: str = "luoxiaohei.luoxiaohei") -> dict | None:
    """获取 .luoxiaohei 文件的基本信息（不完整加载）。兼容 v0.2/v0.3。"""
    if not os.path.exists(filepath):
        return None

    size = os.path.getsize(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    kb_data = data.get("knowledge_base", {})

    # v0.3: Garden 格式
    if "garden" in data:
        garden = data["garden"]
        pots_data = garden.get("pots", {})
        key_count = sum(
            len(pd.get("network", {}).get("keys", {}))
            for pd in pots_data.values()
        )
        wire_count = sum(
            len(pd.get("network", {}).get("wires", []))
            for pd in pots_data.values()
        )
        bridge_count = sum(
            len(bridges) for bridges in garden.get("bridge_table", {}).values()
        ) // 2  # 双向索引，除以 2
        total_iters = sum(
            pd.get("network", {}).get("total_iterations", 0)
            for pd in pots_data.values()
        )
        return {
            "filepath": filepath,
            "size_bytes": size,
            "size_kb": round(size / 1024, 1),
            "version": data.get("version", 0),
            "saved_at": data.get("saved_at", 0),
            "key_count": key_count,
            "wire_count": wire_count,
            "pot_count": len(pots_data),
            "bridge_count": bridge_count,
            "total_iterations": total_iters,
            "kb_drawers": len(kb_data.get("drawers", [])),
            "kb_entries": sum(
                len(d.get("entries", [])) for d in kb_data.get("drawers", [])
            ),
        }

    # v0.2: 旧格式
    net = data.get("network", {})
    return {
        "filepath": filepath,
        "size_bytes": size,
        "size_kb": round(size / 1024, 1),
        "version": data.get("version", 0),
        "saved_at": data.get("saved_at", 0),
        "key_count": len(net.get("keys", {})),
        "wire_count": len(net.get("wires", [])),
        "pot_count": 1,
        "bridge_count": 0,
        "total_iterations": net.get("total_iterations", 0),
        "kb_drawers": len(kb_data.get("drawers", [])),
        "kb_entries": sum(
            len(d.get("entries", [])) for d in kb_data.get("drawers", [])
        ),
    }


# ═══════════════════════════════════════════════
# 自动检查点
# ═══════════════════════════════════════════════

class CheckpointManager:
    """
    自动检查点管理（v0.3 升级：支持 Garden）。

    用法：
        ckpt = CheckpointManager("luoxiaohei.luoxiaohei", interval=300)
        # 每次对话后：
        ckpt.maybe_save(garden, knowledge_base)
    """

    def __init__(self, filepath: str = "luoxiaohei.luoxiaohei", interval: float = 300):
        self.filepath = filepath
        self.interval = interval  # 自动保存间隔（秒）
        self.last_saved_at: float = 0.0
        self.save_count: int = 0

    def should_save(self, garden: "Garden | None" = None) -> bool:
        """是否到了自动保存的时间。"""
        if self.last_saved_at == 0:
            return True
        elapsed = time.time() - self.last_saved_at
        return elapsed >= self.interval

    def maybe_save(
        self,
        garden: "Garden",
        knowledge_base: "KnowledgeBase | None" = None,
        desk = None,
    ) -> str | None:
        """如果需要保存，则保存。返回文件路径或 None。"""
        if not self.should_save(garden):
            return None
        return self.save(garden, knowledge_base, desk)

    def save(
        self,
        garden: "Garden",
        knowledge_base: "KnowledgeBase | None" = None,
        desk = None,
    ) -> str:
        """立即保存。"""
        path = save_garden_checkpoint(garden, knowledge_base, desk, self.filepath)
        self.last_saved_at = time.time()
        self.save_count += 1
        return path

    def load(self) -> tuple:
        """加载检查点（返回 Garden, KnowledgeBase, Desk, metadata）。"""
        garden, kb, desk, meta = load_checkpoint(self.filepath)
        if not meta["is_new"]:
            self.last_saved_at = meta.get("saved_at", time.time())
        return garden, kb, desk, meta


# ═══════════════════════════════════════════════
# 备份
# ═══════════════════════════════════════════════

def create_backup(filepath: str = "luoxiaohei.luoxiaohei") -> str | None:
    """创建当前文件的备份。"""
    if not os.path.exists(filepath):
        return None

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.backup_{timestamp}"

    import shutil
    shutil.copy2(filepath, backup_path)
    return backup_path

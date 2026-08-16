"""
键盘模型 AI — 键网络（KeyNetwork）

管理所有键和电线的容器。提供增删改查、邻居查询、快照等功能。
网络本身不负责传导——传导由 conduction.py 执行。
"""

from __future__ import annotations

import time
from collections import defaultdict

from .key import Key
from .wire import Wire, MIN_THICKNESS, INITIAL_THICKNESS


class KeyNetwork:
    """键 + 电线的网络容器。"""

    def __init__(self):
        self.keys: dict[int, Key] = {}
        self.wires: dict[tuple[int, int], Wire] = {}
        self.next_key_id: int = 1

        # 统计
        self.total_iterations: int = 0
        self.created_at: float = time.time()
        self.last_checkpoint_at: float = 0.0

    # ── 键操作 ────────────────────────────────────

    def create_key(
        self,
        content: str,
        source: str = "external",
        brightness: int = 0,
    ) -> Key:
        """创建新键，返回它。"""
        key = Key(
            id=self.next_key_id,
            content=content,
            brightness=brightness,
            source=source,
        )
        self.keys[key.id] = key
        self.next_key_id += 1
        return key

    def get_key(self, key_id: int) -> Key | None:
        """获取键，不存在返回 None。"""
        return self.keys.get(key_id)

    def remove_key(self, key_id: int):
        """删除键及其所有关联电线。"""
        if key_id in self.keys:
            del self.keys[key_id]
        # 删除所有涉及此键的电线
        to_remove = [
            (a, b) for (a, b) in self.wires
            if a == key_id or b == key_id
        ]
        for pair in to_remove:
            del self.wires[pair]

    def all_keys(self) -> list[Key]:
        """所有键（含休眠键）。"""
        return list(self.keys.values())

    def get_bright_keys(self) -> list[Key]:
        """亮度 > 0 的所有键（当前在意识中的）。"""
        return [k for k in self.keys.values() if k.is_active]

    def get_top_k_keys(self, k: int = 10) -> list[Key]:
        """亮度最高的 k 个键。"""
        active = self.get_bright_keys()
        active.sort(key=lambda x: x.brightness, reverse=True)
        return active[:k]

    # ── 电线操作 ──────────────────────────────────

    @staticmethod
    def _wire_key(a: int, b: int) -> tuple[int, int]:
        """规范化电线键对（小 ID 在前）。"""
        return (a, b) if a < b else (b, a)

    def get_wire(self, a: int, b: int) -> Wire | None:
        """获取两键之间的电线，不存在返回 None。"""
        return self.wires.get(self._wire_key(a, b))

    def set_wire(self, a: int, b: int, thickness: float = INITIAL_THICKNESS):
        """创建或覆盖两键之间的电线。已存在且更粗 → 不降级。返回 Wire 对象。"""
        pair = self._wire_key(a, b)
        if pair in self.wires:
            # 只升级，不降级
            if thickness > self.wires[pair].thickness:
                self.wires[pair].thickness = thickness
        else:
            self.wires[pair] = Wire(key_from=pair[0], key_to=pair[1], thickness=thickness)
        return self.wires[pair]

    def strengthen_wire(self, a: int, b: int, delta: float):
        """加粗两键之间的电线。若不存在则创建。"""
        wire = self.get_wire(a, b)
        if wire is None:
            self.set_wire(a, b, thickness=min(delta, INITIAL_THICKNESS))
        else:
            wire.strengthen(delta)

    def weaken_wire(self, a: int, b: int, decay: float):
        """变细两键之间的电线。"""
        wire = self.get_wire(a, b)
        if wire is not None:
            wire.weaken(decay)

    def get_neighbors(self, key_id: int) -> list[tuple[int, float]]:
        """
        获取键的所有邻居及电线粗细。
        返回 [(neighbor_id, thickness), ...]，按 thickness 降序排列。
        """
        neighbors = []
        for (a, b), wire in self.wires.items():
            if a == key_id:
                neighbors.append((b, wire.thickness))
            elif b == key_id:
                neighbors.append((a, wire.thickness))
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors

    def all_wires(self) -> list[Wire]:
        """所有电线。"""
        return list(self.wires.values())

    # ── 持久化 ────────────────────────────────────

    def snapshot(self) -> dict:
        """生成网络的完整可序列化快照。"""
        return {
            "keys": {str(kid): k.to_dict() for kid, k in self.keys.items()},
            "wires": [
                {"pair": list(pair), **w.to_dict()}
                for pair, w in self.wires.items()
            ],
            "next_key_id": self.next_key_id,
            "total_iterations": self.total_iterations,
            "created_at": self.created_at,
            "last_checkpoint_at": self.last_checkpoint_at,
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "KeyNetwork":
        """从快照恢复网络。"""
        network = cls()
        network.next_key_id = data.get("next_key_id", 1)
        network.total_iterations = data.get("total_iterations", 0)
        network.created_at = data.get("created_at", time.time())
        network.last_checkpoint_at = data.get("last_checkpoint_at", 0.0)

        for kid, kd in data.get("keys", {}).items():
            network.keys[int(kid)] = Key.from_dict(kd)

        for wd in data.get("wires", []):
            pair = tuple(wd["pair"])
            network.wires[pair] = Wire.from_dict({k: v for k, v in wd.items() if k != "pair"})

        return network

    # ── 统计 ────────────────────────────────────

    @property
    def key_count(self) -> int:
        return len(self.keys)

    @property
    def wire_count(self) -> int:
        return len(self.wires)

    @property
    def dormant_count(self) -> int:
        return sum(1 for k in self.keys.values() if k.is_dormant)

    @property
    def avg_wire_thickness(self) -> float:
        if not self.wires:
            return 0.0
        return sum(w.thickness for w in self.wires.values()) / len(self.wires)

    def __repr__(self) -> str:
        bright = len(self.get_bright_keys())
        return (
            f"KeyNetwork(keys={self.key_count}, wires={self.wire_count}, "
            f"bright={bright}, dormant={self.dormant_count})"
        )

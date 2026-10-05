#!/usr/bin/env python3
"""
frag_check —— 备课查重 CLI（脑外，纯符号）

用法：
    python tools/frag_check.py --new 待入库.frag --existing 已有1.frag [已有2.frag ...]

    --new       待入库的 .frag（文件或目录）
    --existing  已入网的 .frag（文件或目录，可重复传）
    --threshold 近重复阈值（默认 0.85）

退出码：0 = 无重复/冗余；1 = 有精确重复 / 近重复 / 冗余提示。

判定口径（与 src/inout/frag.py 的 deduplicate_triples 一致）：
    - 精确重复：主体 / 关系词 / 客体 三项全同
    - 近重复：主体、客体相似度都 ≥ 阈值，且关系词同原语（同义）
    - 冗余上限：同一 (主体, 客体) 出现 ≥3 次（重点概念允许 2~3 次强化）
"""

from __future__ import annotations

import argparse
import os
import sys

# 项目根目录入 path（脚本在 tools/ 下）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 控制台默认 GBK，print 中文需 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.inout.frag import deduplicate_triples, load_triples_from_files  # noqa: E402


def _expand(path: str) -> list[str]:
    """路径 → .frag 文件列表（文件或目录）。"""
    if os.path.isdir(path):
        return sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".frag")
        )
    if os.path.isfile(path):
        return [path]
    return []


def _show(triple, indent="  ") -> str:
    return f"{indent}{triple.subject} {triple.relation} {triple.object}"


def main() -> int:
    ap = argparse.ArgumentParser(description="备课查重（脑外纯符号）")
    ap.add_argument("--new", required=True, help="待入库 .frag（文件或目录）")
    ap.add_argument("--existing", nargs="+", required=True, help="已有 .frag（文件或目录）")
    ap.add_argument("--threshold", type=float, default=0.85, help="近重复阈值")
    args = ap.parse_args()

    new_files = _expand(args.new)
    existing_files: list[str] = []
    for p in args.existing:
        existing_files.extend(_expand(p))

    new_triples = load_triples_from_files(new_files)
    existing_triples = load_triples_from_files(existing_files)

    if not new_triples:
        print(f"〔无新三元组〕{args.new} 里没解析出任何三元组")
        return 1

    report = deduplicate_triples(
        new_triples, existing_triples, near_threshold=args.threshold,
    )

    print(f"〔查重报告〕")
    print(f"  新三元组: {len(new_triples)} 条")
    print(f"  保留: {len(report.kept)} 条")
    print(f"  精确重复: {len(report.exact_dups)} 条")
    print(f"  近重复: {len(report.near_dups)} 条")
    print(f"  冗余(≥3 次强化): {len(report.redundant)} 条")
    print()

    if report.exact_dups:
        print("── 精确重复（已存在于网，勿再投）──")
        for new, _existing in report.exact_dups:
            print(_show(new))
        print()

    if report.near_dups:
        print("── 近重复（建议合并，避免碎片化）──")
        for new, existing, score in report.near_dups:
            print(f"  新: {new.subject} {new.relation} {new.object}")
            print(f"  已有: {existing.subject} {existing.relation} {existing.object}"
                  f"（相似度 {score:.2f}）")
        print()

    if report.redundant:
        print("── 冗余提示（同一对主体/客体强化 ≥3 次）──")
        for subj, obj, count in report.redundant:
            print(f"  {subj} ↔ {obj}: 已出现 {count} 次")
        print()

    if report.is_clean:
        print("✓ 干净，可全部投喂")
        return 0

    print(f"✗ 有 {len(report.exact_dups)} 精确重复 / "
          f"{len(report.near_dups)} 近重复 / "
          f"{len(report.redundant)} 冗余，请处理后重试")
    return 1


if __name__ == "__main__":
    sys.exit(main())

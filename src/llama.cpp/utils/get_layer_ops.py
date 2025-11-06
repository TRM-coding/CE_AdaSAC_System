#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count_ops_by_layer.py
从 .dot 计算图统计第 i 层使用到的算子及次数。

修正点：
- 解析边并做“层归属传播”：以带层号 i 的节点为种子，沿邻接节点 BFS，把相邻但未命名为层号的中间节点一并归到第 i 层；
  遇到“其它层号”的节点停止扩散；常量节点不作为扩散桥。
- 仍忽略常量/权重（label 含 'CONST'）。
- 算子名仍取 label 最后一段的 '<port>OP(...)'，去端口、去参数。
"""

import re
import argparse
import json
from collections import Counter, deque

# 节点与边
NODE_RE = re.compile(r'^\s*"([^"]+)"\s*\[\s*[^]]*label="([^"]*)"')
EDGE_RE = re.compile(r'^\s*"([^"]+)"\s*->\s*"([^"]+)"')

# 任意层号标记（用于“其它层号”的边界识别）
ANY_LAYER_PAT = re.compile(r'blk\.\d+\.|-\d+(?=[^\d])|_l\d+\b')

def is_const_label(label: str) -> bool:
    return "CONST" in label

def label_hits_layer(label: str, layer: int) -> bool:
    if f"blk.{layer}." in label:
        return True
    if re.search(rf"_l{layer}\b", label):
        return True
    if re.search(rf"-(?:{layer})(?=[^\d])", label):
        return True
    return False

def extract_op_name(label: str) -> str | None:
    if is_const_label(label):
        return None
    # 取 label 最后一段（算子段）
    last = label.rsplit('|', 1)[-1].strip()
    if not last.startswith('<'):
        return None
    # 去端口
    last = re.sub(r'^<[^>]+>', '', last).strip()
    if not last:
        return None
    # 去参数括号（若存在）
    op = re.sub(r'\(.*\)$', '', last).strip()
    return op or None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dot_file")
    ap.add_argument("--layer", type=int, required=True, help="要统计的层号 i")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    args = ap.parse_args()

    labels: dict[str, str] = {}        # id -> label
    ops: dict[str, str | None] = {}    # id -> op or None
    neighbors: dict[str, set[str]] = {}  # 无向邻接：id -> 邻接集

    seeds_this_layer: set[str] = set()  # 本层种子
    blockers_other_layers: set[str] = set()  # 其它层的阻断节点

    # 一遍扫描收集节点与边
    with open(args.dot_file, "r", encoding="utf-8") as f:
        for line in f:
            m = NODE_RE.match(line)
            if m:
                nid, label = m.group(1), m.group(2)
                labels[nid] = label
                ops[nid] = extract_op_name(label)

                if label_hits_layer(label, args.layer):
                    seeds_this_layer.add(nid)
                else:
                    # 任意“带层号但非本层”的节点，记为传播边界
                    if ANY_LAYER_PAT.search(label):
                        blockers_other_layers.add(nid)
                continue

            m = EDGE_RE.match(line)
            if m:
                u, v = m.group(1), m.group(2)
                neighbors.setdefault(u, set()).add(v)
                neighbors.setdefault(v, set()).add(u)

    # 若没有任何本层种子，直接空结果
    if not seeds_this_layer:
        out = {"layer": args.layer, "ops": {}}
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print(f"Layer {args.layer} operator usage:")
        return

    # BFS 传播归属
    belong: set[str] = set(seeds_this_layer)
    seen: set[str] = set(seeds_this_layer)
    q = deque(seeds_this_layer)

    while q:
        u = q.popleft()
        for v in neighbors.get(u, ()):
            if v in seen:
                continue
            # 其它层的节点作为阻断，不跨越
            if v in blockers_other_layers:
                seen.add(v)
                continue
            # 常量节点不作为桥，不加入归属
            lbl_v = labels.get(v, "")
            if lbl_v and is_const_label(lbl_v):
                seen.add(v)
                continue
            seen.add(v)
            belong.add(v)
            q.append(v)

    # 计数：仅统计归属集合中的算子节点
    counts = Counter()
    for nid in belong:
        op = ops.get(nid)
        if op:
            counts[op] += 1

    if args.json:
        print(json.dumps({"layer": args.layer, "ops": dict(counts)}, ensure_ascii=False, indent=2))
    else:
        print(f"Layer {args.layer} operator usage:")
        for op, c in counts.most_common():
            print(f"{op}\t{c}")

if __name__ == "__main__":
    main()

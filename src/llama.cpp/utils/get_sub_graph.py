#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_dot_by_layers.py
从计算图 .dot 文件中过滤出指定层（如 4,6,7）的节点与边，输出子图 .dot。

识别规则（尽量覆盖常见命名）：
- 常量与权重：  "blk.{i}.*"        例：blk.4.ffn_up.weight
- 算子后缀层号："*-{i}"            例：ffn_out-6, __fattn__-6
- 下划线层号：  "*_l{i}"           例：cache_k_l6, cache_v_l6

可选：--neighbors N 额外包含与目标集合相距 N 跳的邻居节点，便于保留上下文。
"""

import re
import argparse
from collections import defaultdict, deque

NODE_RE = re.compile(r'^\s*"([^"]+)"\s*\[\s*[^]]*label="([^"]*)"')
EDGE_RE = re.compile(r'^\s*"([^"]+)"\s*->\s*"([^"]+)"')

def parse_layers(arg: str) -> set[int]:
    """
    将"6,7,9-12"解析为{6,7,9,10,11,12}
    """
    out = set()
    for part in arg.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out

def label_hits_layer(label: str, layers: set[int]) -> bool:
    """
    判断节点 label 是否属于指定层集合。
    命中条件任一成立即返回 True。
    """
    # blk.{i}.
    for i in layers:
        if f'blk.{i}.' in label:
            return True

    # *_l{i}（如 cache_k_l6）
    m_iter = re.finditer(r'_l(\d+)\b', label)
    for m in m_iter:
        if int(m.group(1)) in layers:
            return True

    # *-{i}（如 ffn_out-6, __fattn__-12）
    # 允许后面跟空格、竖线、右引号或非数字字符
    m_iter = re.finditer(r'-(\d+)(?=[^\d])', label)
    for m in m_iter:
        if int(m.group(1)) in layers:
            return True

    return False

def main():
    ap = argparse.ArgumentParser(description="Filter a DOT compute-graph by layers.")
    ap.add_argument("input_dot", help="输入 .dot 文件路径")
    ap.add_argument("output_dot", help="输出 .dot 文件路径")
    ap.add_argument("--layers", required=True,
                    help='层号集合，如 "6" 或 "4,6,7" 或 "3-5,8"')
    ap.add_argument("--neighbors", type=int, default=0,
                    help="额外包含与目标集合相距 N 跳的邻居节点，默认 0")
    args = ap.parse_args()

    layers = parse_layers(args.layers)

    # 读入并解析
    node_label: dict[str, str] = {}
    node_line: dict[str, str] = {}
    edges: list[tuple[str, str, str]] = []

    with open(args.input_dot, "r", encoding="utf-8") as f:
        for line in f:
            # 记录节点
            nm = NODE_RE.match(line)
            if nm:
                nid, label = nm.group(1), nm.group(2)
                node_label[nid] = label
                node_line[nid] = line.rstrip('\n')
                continue
            # 记录边
            em = EDGE_RE.match(line)
            if em:
                src, dst = em.group(1), em.group(2)
                edges.append((src, dst, line.rstrip('\n')))

    # 初始保留集合：命中指定层的节点
    keep: set[str] = set(
        nid for nid, lbl in node_label.items()
        if label_hits_layer(lbl, layers)
    )

    # 可选：扩邻 N 跳
    if args.neighbors > 0 and keep:
        adj = defaultdict(set)
        radj = defaultdict(set)
        for s, d, _ in edges:
            adj[s].add(d)
            radj[d].add(s)

        frontier = deque((nid, 0) for nid in keep)
        seen = set(keep)
        while frontier:
            nid, dist = frontier.popleft()
            if dist == args.neighbors:
                continue
            for nxt in adj[nid] | radj[nid]:
                if nxt not in seen and nxt in node_label:
                    seen.add(nxt)
                    keep.add(nxt)
                    frontier.append((nxt, dist + 1))

    # 过滤边：两端都在 keep
    kept_edges = [el for el in edges if el[0] in keep and el[1] in keep]

    # 输出 .dot
    with open(args.output_dot, "w", encoding="utf-8") as w:
        w.write("digraph G {\n")
        # 可选：你也可以在这里加全局样式，如 rankdir=LR;
        # w.write('  rankdir=LR;\n')
        for nid in keep:
            w.write(f"  {node_line[nid]}\n")
        for _, _, eline in kept_edges:
            w.write(f"  {eline}\n")
        w.write("}\n")

if __name__ == "__main__":
    main()

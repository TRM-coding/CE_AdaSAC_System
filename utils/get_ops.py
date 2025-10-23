#!/usr/bin/env python3
# usage:
#   python extract_ops_unique.py ggml.log
#   cat ggml.log | python extract_ops_unique.py

import sys, re

pat = re.compile(r'=\s*\([^)]+\)\s*([A-Z][A-Z0-9_]+)\s*\(')
ops = []

def iter_lines(paths):
    if paths:
        for p in paths:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    yield line
    else:
        for line in sys.stdin:
            yield line

for line in iter_lines(sys.argv[1:]):
    m = pat.search(line)
    if m:
        ops.append(m.group(1))

for op in dict.fromkeys(ops):  # 去重且保序
    print(op)

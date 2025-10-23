#!/usr/bin/env python3
# 用法:
#   python explain_ops_unique.py ggml.log
#   cat ggml.log | python explain_ops_unique.py
import sys, re

pat_full = re.compile(r'=\s*\([^)]+\)\s*([A-Z][A-Z0-9_]+)\s*\(')
pat_shapes = re.compile(r'\{([^}]*)\}')
pat_simple = re.compile(r'^\s*([A-Z][A-Z0-9_]+)\s+(.+?)\s+([0-9x]+)\s*$')

def norm_shape(s):
    s = s.replace(",", "x")
    xs = [t.strip() for t in s.split("x") if t.strip()]
    ne = []
    for t in xs:
        try: ne.append(int(t))
        except: ne.append(t)
    while len(ne) < 4: ne.append(1)
    return ne[:4]

def parse_line(line):
    if ") = {" in line:
        m_op = pat_full.search(line)
        if not m_op: return None
        op = m_op.group(1)
        ss = pat_shapes.findall(line)
        if not ss: return None
        ins = [norm_shape(s) for s in ss[:-1]]
        out = norm_shape(ss[-1])
        return op, ins, out
    m = pat_simple.match(line)
    if not m: return None
    op = m.group(1)
    ins = [norm_shape(s) for s in m.group(2).split(";")]
    out = norm_shape(m.group(3))
    return op, ins, out

def shape_str(ne): return "x".join(str(v) for v in ne)

class Ctx:
    def __init__(self):
        self.H=self.D=self.NH=self.NKV=self.I=self.T=self.Tk=None
    def learn_flash(self,q,kv):
        self.D=self.D or q[0]; self.T=self.T or q[1]; self.NH=self.NH or q[2]
        self.Tk=self.Tk or kv[1]; self.NKV=self.NKV or kv[2]
    def learn_reshape(self,src,dst):
        if dst[0]==src[0]*src[1]:
            self.D=self.D or src[0]; self.NH=self.NH or src[1]; self.T=self.T or src[2]; self.H=self.H or dst[0]
        if self.D and self.NH and src[0]==self.D and src[1]==self.NH: self.H=self.H or dst[0]
    def learn_ffn(self,W,X,Y):
        M,K = W[0],W[1]
        if self.H and K==self.H and M>K: self.I=self.I or M
    def role_ffn(self,W,X,Y):
        M,K = W[0],W[1]
        if self.H and self.I:
            if M==self.I and K==self.H: return "FFN up/gate"
            if M==self.H and K==self.I: return "FFN down"
        if M==K: return "H→H 投影"
        return "线性层"

ctx = Ctx()

def explain(op,ins,out):
    if op=="GET_ROWS":
        W,ids = ins; H,V = W[0],W[1]; T = ids[0]
        return f"嵌入查表：W{{H={H},V={V}}}, ids{{T={T}}}"
    if op in ("RMS_NORM","NORM"):
        return "归一化：逐通道"
    if op=="MUL":
        a,b = ins
        return "逐元素乘" + ("（通道缩放）" if b[1]==1 and a[0]==b[0] else "")
    if op=="ADD": return "逐元素加（残差）"
    if op=="SCALE": return "乘标量"
    if op=="MUL_MAT":
        W,X = ins; M,K = W[0],W[1]; K2,N = X[0],X[1]
        ctx.learn_ffn(W,X,out)
        return f"{ctx.role_ffn(W,X,out)}：W[{M}×{K}]·X[{K2}×{N}]"
    if op=="SWIGLU": return "SwiGLU：SiLU(gate)⊙up"
    if op=="GELU": return "GELU"
    if op=="RESHAPE":
        src = ins[0]; ctx.learn_reshape(src,out)
        extra = ""
        if ctx.D and ctx.NH and ctx.H and src[0]==ctx.D and src[1]==ctx.NH and out[0]==ctx.H:
            extra = f"（合并头 D×NH={ctx.D}×{ctx.NH}→H={ctx.H}）"
        return "重塑"+extra
    if op=="ROPE": return "RoPE 旋转位置编码"
    if op=="VIEW": return "VIEW 视图"
    if op=="SET_ROWS": return "按 ids 写入行（KV 缓存）"
    if op=="PERMUTE": return "轴交换"
    if op=="CPY": return "拷贝/类型转换"
    if op=="FLASH_ATTN_EXT":
        q=ins[0]; kv = ins[1] if len(ins)>1 else None
        if kv: ctx.learn_flash(q,kv); return f"融合注意力：Q{{D={q[0]},T={q[1]},NH={q[2]}}}, KV{{D={kv[0]},Tk={kv[1]},NKV={kv[2]}}}"
        return "融合注意力"
    return op

def iter_lines(paths):
    if paths:
        for p in paths:
            with open(p,"r",encoding="utf-8",errors="ignore") as f:
                for line in f: yield line.rstrip("\n")
    else:
        for line in sys.stdin: yield line.rstrip("\n")

def main():
    rows=[]
    for line in iter_lines(sys.argv[1:]):
        r = parse_line(line)
        if r: rows.append(r)

    # 先学全局维度
    for op,ins,out in rows:
        if op=="FLASH_ATTN_EXT" and len(ins)>1: ctx.learn_flash(ins[0],ins[1])
        if op=="RESHAPE": ctx.learn_reshape(ins[0],out)
        if op=="MUL_MAT": ctx.learn_ffn(ins[0],ins[1],out)

    hdr=[]
    if ctx.H: hdr.append(f"H={ctx.H}")
    if ctx.D: hdr.append(f"D={ctx.D}")
    if ctx.NH: hdr.append(f"NH={ctx.NH}")
    if ctx.NKV: hdr.append(f"NKV={ctx.NKV}")
    if ctx.T: hdr.append(f"T={ctx.T}")
    if ctx.Tk: hdr.append(f"Tk={ctx.Tk}")
    if ctx.I: hdr.append(f"I={ctx.I}")
    if hdr: print("# dims:", ", ".join(hdr))

    seen=set()
    for op,ins,out in rows:
        if op in seen: continue
        seen.add(op)
        ins_s = "; ".join(shape_str(s) for s in ins)
        print(f"{op:14s} {ins_s:>24s}  -> {shape_str(out):<12s} | {explain(op,ins,out)}")

if __name__=="__main__":
    main()

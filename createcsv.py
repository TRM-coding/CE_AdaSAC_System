#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 生成“核心字段”CSV示例：op、各维度、backend、device、avg_ms
# 每个算子单独文件夹，文件名 results.csv
import os, csv, random, itertools as it
from datetime import datetime

random.seed(42)

# 维度集合（含你指定的 T）
H_SET  = [1024, 2048, 4096]
V_SET  = [32768, 65536, 131072]
T_SET  = [4, 64, 256, 1024]
NH_SET = [8, 16, 32]
I_SET  = [4096, 6144, 8192, 12288]
TK_SET = [256, 2048, 8192]   # 注意力/缓存写入

CSV_HEADER = [
    "op","in_shapes","out_shape",
    "H","D","NH","NKV","T","Tk","I","V","M","K","N",
    "backend","device","avg_ms"
]

BACKEND = "cpu-ggml"
DEVICE  = "host"

def shape(*ne): return "{" + ",".join(str(x) for x in ne) + "}"

def ms(work, scale, noise=0.15):
    base = work / max(scale, 1.0)
    return max(0.001, base * random.gauss(1.0, noise) * 1e3)

def row(op, in_shapes, out_shape, dims, work, scale):
    d = {k:str(v) for k,v in dims.items()}
    return [
        op, in_shapes, out_shape,
        d.get("H",""), d.get("D",""), d.get("NH",""), d.get("NKV",""),
        d.get("T",""), d.get("Tk",""), d.get("I",""), d.get("V",""),
        d.get("M",""), d.get("K",""), d.get("N",""),
        BACKEND, DEVICE, f"{ms(work,scale):.3f}"
    ]

def write(op, rows):
    os.makedirs(op, exist_ok=True)
    with open(os.path.join(op, "results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(CSV_HEADER); w.writerows(rows)

# ---------------- 各算子示例生成 ----------------
def gen_GET_ROWS():
    rows=[]
    for H,V,T in it.product(H_SET,V_SET,T_SET):
        in1=shape(H,V,1,1); in2=shape(T,1,1,1); out=shape(H,T,1,1)
        rows.append(row("GET_ROWS", f"W{in1}; ids{in2}", out,
                        {"H":H,"V":V,"T":T}, work=H*T, scale=3e8))
    write("GET_ROWS", rows)

def gen_RMS_NORM():
    rows=[]
    for H,T in it.product(H_SET,T_SET):
        x=shape(H,T,1,1)
        rows.append(row("RMS_NORM", f"x{x}", x,
                        {"H":H,"T":T}, work=H*T, scale=5e8))
    write("RMS_NORM", rows)

def gen_MUL():
    rows=[]
    for H,T in it.product(H_SET,T_SET):
        a=shape(H,T,1,1); gamma=shape(H,1,1,1)
        rows.append(row("MUL", f"a{a}; gamma{gamma}", a,
                        {"H":H,"T":T}, work=H*T, scale=8e8))
        b=shape(H,T,1,1)
        rows.append(row("MUL", f"a{a}; b{b}", a,
                        {"H":H,"T":T}, work=H*T, scale=8e8))
    write("MUL", rows)

def gen_ADD():
    rows=[]
    for H,T in it.product(H_SET,T_SET):
        a=shape(H,T,1,1); b=a
        rows.append(row("ADD", f"a{a}; b{b}", a,
                        {"H":H,"T":T}, work=H*T, scale=8e8))
        b2=shape(H,1,1,1)
        rows.append(row("ADD", f"a{a}; b{b2}", a,
                        {"H":H,"T":T}, work=H*T, scale=8e8))
    write("ADD", rows)

def gen_SCALE():
    rows=[]
    for H,T in it.product(H_SET,T_SET):
        x=shape(H,T,1,1); alpha=shape(1,1,1,1)
        rows.append(row("SCALE", f"x{x}; alpha{alpha}", x,
                        {"H":H,"T":T}, work=H*T, scale=1e9))
    write("SCALE", rows)

def gen_MUL_MAT():
    rows=[]
    # H->H
    for H,T in it.product(H_SET,T_SET):
        W=shape(H,H,1,1); X=shape(H,T,1,1); Y=shape(H,T,1,1)
        rows.append(row("MUL_MAT", f"W{W}; X{X}", Y,
                        {"H":H,"T":T,"M":H,"K":H,"N":T}, work=2*H*H*T, scale=2e11))
    # FFN up/down
    for H,I,T in it.product(H_SET,I_SET,T_SET):
        W=shape(I,H,1,1); X=shape(H,T,1,1); Y=shape(I,T,1,1)
        rows.append(row("MUL_MAT", f"W{W}; X{X}", Y,
                        {"H":H,"I":I,"T":T,"M":I,"K":H,"N":T}, work=2*I*H*T, scale=2e11))
        W=shape(H,I,1,1); X=shape(I,T,1,1); Y=shape(H,T,1,1)
        rows.append(row("MUL_MAT", f"W{W}; X{X}", Y,
                        {"H":H,"I":I,"T":T,"M":H,"K":I,"N":T}, work=2*H*I*T, scale=2e11))
    # K/V 投影（GQA）
    rows2=[]
    for H,NH,T in it.product(H_SET,NH_SET,T_SET):
        if H%NH: continue
        D=H//NH
        for NKV in {1, NH, NH//2 if NH%2==0 else None}-{None}:
            W=shape(D*NKV,H,1,1); X=shape(H,T,1,1); Y=shape(D*NKV,T,1,1)
            rows2.append(row("MUL_MAT", f"W{W}; X{X}", Y,
                             {"H":H,"D":D,"NH":NH,"NKV":NKV,"T":T,"M":D*NKV,"K":H,"N":T},
                             work=2*(D*NKV)*H*T, scale=2e11))
    write("MUL_MAT", rows+rows2)

def gen_RESHAPE():
    rows=[]
    for H,NH,T in it.product(H_SET,NH_SET,T_SET):
        if H%NH: continue
        D=H//NH
        src=shape(H,T,1,1); dst=shape(D,NH,T,1)
        rows.append(row("RESHAPE", f"x{src}", dst,
                        {"H":H,"D":D,"NH":NH,"T":T}, work=H*T, scale=2e9))
    write("RESHAPE", rows)

def gen_ROPE():
    rows=[]
    for H,NH,T in it.product(H_SET,NH_SET,T_SET):
        if H%NH: continue
        D=H//NH
        q=shape(D,NH,T,1); pos=shape(T,1,1,1)
        rows.append(row("ROPE", f"q{q}; pos{pos}", q,
                        {"H":H,"D":D,"NH":NH,"T":T}, work=D*NH*T, scale=8e8))
    write("ROPE", rows)

def gen_VIEW():
    rows=[]
    for H,NH,T in it.product(H_SET,NH_SET,T_SET):
        if H%NH: continue
        D=H//NH
        buf=shape(D*T,NH,1,1); view=shape(D,NH,T,1)
        rows.append(row("VIEW", f"buf{buf}", view,
                        {"H":H,"D":D,"NH":NH,"T":T}, work=D*NH*T, scale=1e9))
    write("VIEW", rows)

def gen_SET_ROWS():
    rows=[]
    for H,NH,NKV,T,Tk in it.product(H_SET,NH_SET,[1,4,16],T_SET,TK_SET):
        if H%NH: continue
        D=H//NH; dnkv=D*NKV
        dst=shape(dnkv,Tk,1,1); src=shape(dnkv,T,1,1); ids=shape(T,1,1,1)
        rows.append(row("SET_ROWS", f"dst{dst}; src{src}; ids{ids}", dst,
                        {"H":H,"D":D,"NH":NH,"NKV":NKV,"T":T,"Tk":Tk},
                        work=dnkv*T, scale=8e8))
    write("SET_ROWS", rows)

def gen_PERMUTE():
    rows=[]
    for H,NH,T in it.product(H_SET,NH_SET,T_SET):
        if H%NH: continue
        D=H//NH
        src=shape(D,NH,T,1); dst=shape(D,T,NH,1)
        rows.append(row("PERMUTE", f"x{src}", dst,
                        {"H":H,"D":D,"NH":NH,"T":T}, work=D*NH*T, scale=1.5e9))
    write("PERMUTE", rows)

def gen_CPY():
    rows=[]
    for H,T in it.product(H_SET,T_SET):
        src=shape(H,T,1,1); dst=src
        rows.append(row("CPY", f"src{src}", dst,
                        {"H":H,"T":T}, work=H*T, scale=5e8))
    write("CPY", rows)

def gen_FLASH_ATTN_EXT():
    rows=[]
    for H,NH,T,Tk in it.product(H_SET,NH_SET,T_SET,TK_SET):
        if H%NH: continue
        D=H//NH
        for NKV in {1, NH, NH//2 if NH%2==0 else None}-{None}:
            Q=shape(D,NH,T,1); KV=shape(D,Tk,NKV,1); OUT=shape(D,NH,T,1)
            rows.append(row("FLASH_ATTN_EXT", f"Q{Q}; KV{KV}", OUT,
                            {"H":H,"D":D,"NH":NH,"NKV":NKV,"T":T,"Tk":Tk},
                            work=NH*T*Tk*D, scale=8e10))
    write("FLASH_ATTN_EXT", rows)

def gen_SWIGLU():
    rows=[]
    for I,T in it.product(I_SET,T_SET):
        up=shape(I,T,1,1); gate=shape(I,T,1,1)
        rows.append(row("SWIGLU", f"up{up}; gate{gate}", up,
                        {"I":I,"T":T}, work=2*I*T, scale=5e9))
    write("SWIGLU", rows)

def gen_GELU():
    rows=[]
    for I,T in it.product(I_SET,T_SET):
        x=shape(I,T,1,1)
        rows.append(row("GELU", f"x{x}", x,
                        {"I":I,"T":T}, work=I*T, scale=6e9))
    write("GELU", rows)

def main():
    gens = [
        gen_GET_ROWS, gen_RMS_NORM, gen_MUL, gen_ADD, gen_SCALE,
        gen_MUL_MAT, gen_RESHAPE, gen_ROPE, gen_VIEW, gen_SET_ROWS,
        gen_PERMUTE, gen_CPY, gen_FLASH_ATTN_EXT, gen_SWIGLU, gen_GELU
    ]
    for g in gens: g()

if __name__ == "__main__":
    main()

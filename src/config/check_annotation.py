#!/usr/bin/env python3
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#
# Usage: config/check_annotation.py (from top-level `src` folder)
#
# Check that annoted regions are opened and closed properly. For each target `.c` file:
#   1. Parse file function ranges with `ctags` (function name + start/end line).
#   2. Lex the file into a light token stream.
#   3. Detect lambda-body block starts (`{` that start lambda bodies).
#   4. Produce a stream of *events* inside ordinary blocks only (exclude lambda internals):
#   5. For each function’s line range, walk those events and apply stack-based checks.
#

import glob, pathlib, re, shutil, subprocess, sys

P='HYPRE_ANNOTATE_'; FB=f'{P}FUNC_BEGIN'; FE=f'{P}FUNC_END'
OPEN={f'{P}REGION_BEGIN':'REGION',f'{P}MGLEVEL_BEGIN':'MGLEVEL',f'{P}ITER_BEGIN':'ITER',f'{P}LOOP_BEGIN':'LOOP'}
CLOSE={f'{P}REGION_END':'REGION',f'{P}MGLEVEL_END':'MGLEVEL',f'{P}ITER_END':'ITER',f'{P}LOOP_END':'LOOP'}
TOK={FB,FE,'return','break','continue',*OPEN,*CLOSE}
TOK_RE=re.compile(r'''(?P<nl>\n)|(?P<ws>[\t \r\f\v]+)|(?P<line_comment>//[^\n]*\n?)|(?P<block_comment>/\*.*?\*/)|(?P<str>(?:"(?:\\.|[^"\\\n])*")|(?:\'(?:\\.|[^\'\\\n])*\'))|(?P<id>[A-Za-z_][A-Za-z0-9_]*)|(?P<op>->|::|&&)|(?P<ch>.)''',re.S)

def funcs(path):
    out=subprocess.run(["ctags","-f","-","--sort=no","--fields=+ne","--kinds-c=f",path],capture_output=True,text=True,check=False).stdout
    for row in out.splitlines():
        d={p.split(':',1)[0]:int(p.split(':',1)[1]) for p in row.split('\t') if p.startswith(('line:','end:'))}
        if 'line' in d and 'end' in d: yield row.split('\t',1)[0],d['line'],d['end']

def lex(src):
    out=[]; line=1
    for m in TOK_RE.finditer(src):
        tok,kind=m.group(0),m.lastgroup
        if kind in {'ws','nl'}: line += kind=='nl'; continue
        line += tok.count('\n')
        if kind not in {'line_comment','block_comment','str'}: out.append((tok,line))
    return out

def match(t,i,a,b,step):
    d=0
    for j in range(i, len(t) if step>0 else -1, step):
        v=t[j][0]; d += v==(a if step>0 else b); d -= v==(b if step>0 else a)
        if d==0: return j

def lambda_blocks(t):
    s=set('mutable constexpr const noexcept & && *'.split()); out=set(); i=0
    while i<len(t):
        if t[i][0]!='[': i+=1; continue
        j=match(t,i,'[',']',1)
        if j is None: i+=1; continue
        k=j+1
        if k<len(t) and t[k][0]=='(':
            k=match(t,k,'(',')',1)
            if k is None: i+=1; continue
            k+=1
        while k<len(t) and t[k][0] in s: k+=1
        if k<len(t) and t[k][0]=='->':
            k+=1
            while k<len(t) and t[k][0] not in {'{',';',',',')',']','}'}: k+=1
        if k<len(t) and t[k][0]=='{': out.add(k); i=k+1; continue
        i+=1
    return out

def block_kind(t,i):
    if i and t[i-1][0] in {'else','do'}: return t[i-1][0]
    if i and t[i-1][0]==')':
        j=match(t,i-1,'(',')',-1)
        if j and t[j-1][0] in {'if','for','while','switch'}: return t[j-1][0]
    return 'block'

def events(src):
    t=lex(src); lambdas=lambda_blocks(t); out=[]; kinds={}; path=[]; blocks=[]; nid=1; lam=0
    for i,(v,ln) in enumerate(t):
        if not lam and v in TOK: out.append((v,ln,tuple(path)))
        if v=='{':
            is_l=i in lambdas; blocks.append((nid,is_l)); lam += is_l
            if not is_l: kinds[nid]=block_kind(t,i); path.append(nid)
            nid+=1
        elif v=='}' and blocks:
            b,is_l=blocks.pop()
            if is_l: lam-=1
            elif path and path[-1]==b: path.pop()
    return out,kinds

def pre(a,b): return len(a)<=len(b) and a==b[:len(a)]

if not shutil.which('ctags'): sys.exit(print('check_annotation: requires Universal Ctags (ctags in PATH)',file=sys.stderr) or 2)
SCRIPT_DIR=pathlib.Path(__file__).resolve().parent
SRC_ROOT=(SCRIPT_DIR/"src" if (SCRIPT_DIR/"src").is_dir() else SCRIPT_DIR.parent/"src" if (SCRIPT_DIR.parent/"src").is_dir() else SCRIPT_DIR.parent.parent/"src" if (SCRIPT_DIR.parent.parent/"src").is_dir() else SCRIPT_DIR)

err=nf=nr=nm=ne=0
for path in [p for p in (sys.argv[1:] or glob.glob(str(SRC_ROOT/"**"/"*.c"), recursive=True)) if p]:
    ev,kinds=events(open(path,encoding='utf-8',errors='replace').read())
    for name,s,e in funcs(path):
        st=[]; fb=fe=0; ends=[]; prev=(); entry={}; skip=None
        for v,ln,p in ev:
            if ln<s: continue
            if ln>e: break
            if skip:
                block,keep=skip
                if block in p: continue
                st=keep[:]; skip=None
            common=0
            while common<len(prev) and common<len(p) and prev[common]==p[common]: common+=1
            for b in p[common:]: entry[b]=st[:]
            prev=p
            if v==FB: fb+=1; fe=0; ends=[]; continue
            if v==FE: fe+=1; ends.append(p)
            if fb==fe==0: continue
            if v==FE:
                if st: print(f"{path}:{ln}: {name}: unclosed before FUNC_END ({' > '.join(st)})",file=sys.stderr); err=1; nm+=1
                st=[]; continue
            if v=='return':
                if not any(pre(end,p) for end in ends):
                    print(f"{path}:{ln}: {name}: return before FUNC_END",file=sys.stderr); err=1; nr+=1
                cond=next((b for b in p if kinds.get(b) in {'if','else'}), None)
                if cond is not None: skip=(cond, entry.get(cond, st[:]))
                continue
            if v in OPEN: st.append(OPEN[v]); continue
            if v in CLOSE:
                m=CLOSE[v]
                if not st: print(f"{path}:{ln}: {name}: unmatched {m}_END",file=sys.stderr); err=1; nm+=1
                elif st[-1]!=m: print(f"{path}:{ln}: {name}: stack mismatch ({st[-1]} vs {m})",file=sys.stderr); err=1; nm+=1
                else: st.pop()
                continue
            if v in {'break','continue'}:
                cond=next((b for b in p if kinds.get(b) in {'if','else'}), None)
                if cond is not None: skip=(cond, entry.get(cond, st[:]))
        if (fb or fe) and (fb!=1 or fe<1): print(f"{path}: {name}: FUNC_BEGIN={fb} FUNC_END={fe} (expected BEGIN=1, END>=1)",file=sys.stderr); err=1; nf+=1
        if st: print(f"{path}:{e}: {name}: unclosed annotation at end ({' > '.join(st)})",file=sys.stderr); err=1; ne+=1

print('Summary:',file=sys.stderr)
print(f'  FUNC: {nf} function(s) with FUNC_BEGIN/FUNC_END count issue',file=sys.stderr)
print(f'  FUNC: {nr} return(s) before FUNC_END',file=sys.stderr)
print(f'  STACK: {nm} stack mismatch/unclosed annotation(s) before FUNC_END',file=sys.stderr)
print(f'  STACK: {ne} unclosed annotation(s) at end of function',file=sys.stderr)
sys.exit(err)

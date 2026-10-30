"""Explicit evidence for over-accumulation in iterative SD (sum rule).
sum score (beta) = (1 - beta*(k-1)) log p + beta * sum_j log q_j.  For k>=3 at beta=1 the base coefficient is negative."""
import os, sys
from pathlib import Path as _P
ROOT = _P(__file__).resolve().parents[1]
os.chdir(ROOT)

import json, glob, statistics as st, collections as C, re
E=str(ROOT/'experiments/alfworld_ace_sd/runs')
R=['kitchen','living_room','bathroom','bedroom']
def load(pat):
    d={}
    for f in glob.glob(pat):
        x=json.load(open(f)); d[(x['target'],x['task_index'],x['repeat'])]=x
    return d
runs={'sum b1':load(f'{E}/20261029_iter/beta_1/*/results/*.json'),
      'sum b0.5':load(f'{E}/20261029_iter/beta_0.5/*/results/*.json'),
      'mean b1':load(f'{E}/20261029_fixedfb/mean/beta_1/*/results/*.json'),
      'ema b1':load(f'{E}/20261029_fixedfb/ema/beta_1/*/results/*.json'),
      'replace b1':load(f'{E}/20261029_fixedfb/replace/beta_1/*/results/*.json')}
am=lambda v:max(range(len(v)),key=lambda i:v[i])
def step_flags(s):
    lp,lq=s['log_p'],s['log_q']; n=len(lp)
    if n<2 or not lq: return None
    poe=am([sum(q[a] for q in lq) for a in range(n)])          # teachers only, no base term
    tops={am(q) for q in lq}
    unanimous = len(tops)==1 and am(lp) in tops                # base and every teacher agree
    ch=s['chosen']
    return dict(k=len(lq)+1, override=unanimous and ch!=am(lp),  # picks against unanimous agreement
                flip=ch!=poe, odd=ch not in tops, argmin=ch==min(range(n),key=lambda i:lp[i]),
                lp_ch=lp[ch], invalid='Nothing happens' in s['next_observation'])
print('1. Unanimity violations: base and all teachers rank the same action first, rule picks another (% of steps)')
print('   coefficient on log p = 1 - beta(k-1)')
for name,d in runs.items():
    beta=1.0 if 'b1' in name else 0.5
    row=[]
    for k in range(2,6):
        fl=[f for x in d.values() if x['repeat']==k for s in x['steps'] if (f:=step_flags(s))]
        un=[f for f in fl if f]  # all
        nu=sum(1 for f in fl if f['override']); nuan=sum(1 for x in d.values() if x['repeat']==k for s in x['steps'] if s.get('log_q') and len(s['log_p'])>1 and len({am(q) for q in s['log_q']})==1 and am(s['log_p'])==am(s['log_q'][0]))
        coef=1-beta*(k-1) if name.startswith('sum') else None
        row.append(f'k={k}'+(f'(c={coef:+.1f})' if coef is not None else '')+f': {100*nu/max(nuan,1):.1f}% of {nuan} unanimous')
    print(f'   {name:10s}',' | '.join(row))

print('\n2. sum b1, k>=3: what drives the deviating picks')
fl=[step_flags(s) for x in runs['sum b1'].values() if x['repeat']>=3 for s in x['steps']]; fl=[f for f in fl if f]
odd=[f for f in fl if f['odd']]; norm=[f for f in fl if not f['odd']]
print(f'   steps {len(fl)}, odd picks (no teacher ranks first) {len(odd)} ({100*len(odd)/len(fl):.1f}%)')
print(f'   odd picks that teachers-only product (sum log q) would NOT pick: {100*st.mean(f["flip"] for f in odd):.1f}%  -> caused by the -(k-2)log p term')
print(f'   odd picks = base argmin: {100*st.mean(f["argmin"] for f in odd):.1f}%;  median log p of chosen: odd {st.median(f["lp_ch"] for f in odd):.2f} vs normal {st.median(f["lp_ch"] for f in norm):.2f}')
print(f'   unanimity violations among odd picks: {sum(f["override"] for f in odd)} ({100*st.mean(f["override"] for f in odd):.1f}%)')

print('\n3. Are deviating picks bad actions? (sum b1, k>=3)')
def revisit_rate(x):
    out=[]; seen=[]
    for s in x['steps']:
        f=step_flags(s)
        if f: out.append((f,s['action'] in seen))
        seen.append(s['action'])
    return out
pairs=[p for x in runs['sum b1'].values() if x['repeat']>=3 for p in revisit_rate(x)]
for lab,sel in [('odd',lambda f:f['odd']),('normal',lambda f:not f['odd'])]:
    g=[(f,rv) for f,rv in pairs if sel(f)]
    print(f'   {lab:6s}: n={len(g)}, "Nothing happens" {100*st.mean(f["invalid"] for f,_ in g):.1f}%, repeats an earlier action {100*st.mean(rv for _,rv in g):.1f}%')

print('\n4. Trial-level: same task, same trial, same feedback text -> mean b1 minus sum b1 success, split by whether sum made odd picks EARLY (first 10 steps)')
S,M=runs['sum b1'],runs['mean b1']
grp=C.defaultdict(list)
for key,x in S.items():
    if x['repeat']<3: continue
    early=[f for s in x['steps'][:10] if (f:=step_flags(s))]
    n_odd=sum(f['odd'] for f in early)
    g='0 odd' if n_odd==0 else ('1 odd' if n_odd==1 else '2+ odd')
    grp[g].append((x['success'],M[key]['success']))
for g in ('0 odd','1 odd','2+ odd'):
    v=grp[g]; s_=100*st.mean(a for a,_ in v); m_=100*st.mean(b for _,b in v)
    print(f'   {g:6s}: n={len(v):3d}  sum SR {s_:5.1f}  mean SR {m_:5.1f}  gap {m_-s_:+5.1f}')
# same split but using the mean run as a difficulty control: are these tasks harder per se? use trial-1 (identical policy) success
print('   difficulty control: trial-1 SR (identical policy in both runs) of the same tasks')
for g in ('0 odd','1 odd','2+ odd'):
    ks=[key for key,x in S.items() if x['repeat']>=3 and (lambda n: ('0 odd' if n==0 else '1 odd' if n==1 else '2+ odd'))(sum(f['odd'] for s in x['steps'][:10] if (f:=step_flags(s))))==g]
    t1=[S[(r,i,1)]['success'] for r,i,_ in ks]; t1m=[M[(r,i,1)]['success'] for r,i,_ in ks]
    print(f'   {g:6s}: sum trial-1 {100*st.mean(t1):5.1f}  mean trial-1 {100*st.mean(t1m):5.1f}')

print('\n5. Within-trajectory: success after the first odd pick vs matched (same task/trial) mean run')
fo=[]
for key,x in S.items():
    if x['repeat']<3: continue
    idx=next((j for j,s in enumerate(x['steps']) if (f:=step_flags(s)) and f['odd']),None)
    if idx is not None: fo.append((idx, x['success'], len(x['steps'])))
print(f'   trials with an odd pick: {len(fo)}; SR {100*st.mean(s for _,s,_ in fo):.1f}%; median first odd step {st.median(i for i,_,_ in fo)}')

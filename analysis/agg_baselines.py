import os, sys
from pathlib import Path as _P
ROOT = _P(__file__).resolve().parents[1]
os.chdir(ROOT)
import json, glob, re, statistics as st
from collections import defaultdict
ROOMS=['kitchen','living_room','bathroom','bedroom']
B='experiments/alfworld_baselines/runs/20261029'
def load_base(m):
    out={}
    for r in ROOMS:
        for f in glob.glob(f'{B}/{m}/{r}/results/*.json'):
            d=json.load(open(f)); out[(r,d['task_index'],d['repeat'])]=d
    return out
def load_simple(pat):
    out={}
    for r in ROOMS:
        for f in glob.glob(pat.format(r=r)):
            d=json.load(open(f)); out[(r,d['task_index'],d['repeat'])]=dict(success=d['success'],task_type=d.get('task_type'))
    return out
arms={m:load_base(m) for m in ('reflexion','evotest','awm')}
arms['mean']=load_simple('experiments/alfworld_ace_sd/runs/20261029_fixedfb/mean/beta_1/{r}/results/*.json')
arms['ema']=load_simple('experiments/alfworld_ace_sd/runs/20261029_fixedfb/ema/beta_1/{r}/results/*.json')
arms['iter05']=load_simple('experiments/alfworld_ace_sd/runs/20261029_iter/beta_0.5/{r}/results/*.json')
arms['nomem']=load_simple('experiments/alfworld_react/runs/20261020/results/jitrl_nomem_{r}/*.json')
arms['react0']=load_simple('experiments/alfworld_react/runs/20261020/results/react0_{r}/*.json')
for a,v in arms.items(): print(a,len(v))
def table(a):
    v=arms[a]; row=[]
    for r in ROOMS:
        ks=[k for k in v if k[0]==r]
        avg=100*st.mean(v[k]['success'] for k in ks); fin=100*st.mean(v[k]['success'] for k in ks if k[2]==5)
        row.append((avg,fin))
    pt=[st.mean(100*st.mean(v[k]['success'] for k in v if k[0]==r and k[2]==t) for r in ROOMS) for t in range(1,6)]
    tf=[];ft=[]
    for (r,i,t),d in v.items():
        if t>1:
            p=v[(r,i,t-1)]['success']
            (tf if p else ft).append(not d['success'] if p else d['success'])
    return row,(st.mean(x[0] for x in row),st.mean(x[1] for x in row)),pt,100*st.mean(tf)
for a in arms:
    row,m,pt,tf=table(a)
    print(a,' | '.join(f'{x:.2f} / {y:.2f}' for x,y in row),'| MEAN %.2f / %.2f'%m,'| trials',[round(x,1) for x in pt],'T->F %.1f'%tf)
def paired(a,b,trials=range(1,6)):
    diffs=[]
    for r in ROOMS:
        tasks=sorted({k[1] for k in arms[a] if k[0]==r})
        dr=[st.mean(arms[a][(r,i,t)]['success']-arms[b][(r,i,t)]['success'] for t in trials) for i in tasks]
        diffs.append((st.mean(dr),st.pstdev(dr)/len(dr)**.5 if len(dr)>1 else 0))
    mu=st.mean(d[0] for d in diffs); se=(sum(d[1]**2 for d in diffs))**.5/4
    return 100*mu,100*se
for a in ('reflexion','evotest','awm'):
    for b in ('nomem','mean','react0'):
        print(a,'vs',b,'all %.1f±%.1f'%paired(a,b),'t2-5 %.1f±%.1f'%paired(a,b,range(2,6)))
print('mean vs nomem', '%.1f±%.1f'%paired('mean','nomem'))
# judge accuracy in-run
for m in ('reflexion','evotest','awm'):
    v=arms[m]; y=[d['success'] for d in v.values()]; j=[d['method_log']['judge']['judged_success'] for d in v.values()]
    tp=sum(a and b for a,b in zip(y,j)); fp=sum((not a) and b for a,b in zip(y,j)); fn=sum(a and not b for a,b in zip(y,j)); tn=sum((not a) and not b for a,b in zip(y,j))
    print(m,'judge acc %.3f prec %.3f rec %.3f failrec %.3f'%((tp+tn)/len(y),tp/(tp+fp),tp/(tp+fn),tn/(tn+fp)), 'FP',fp,'FN',fn, 'missing',sum(bool(d['method_log']['judge']['label_missing']) for d in v.values()))
# method stats
v=arms['reflexion']; print('reflexion reflections',sum(d['method_log']['reflection'] is not None for d in v.values()))
v=arms['evotest']; from collections import Counter
print('evotest actions',Counter(d['guide_log'].get('evo_action') for d in v.values()))
# freeze on FP: frozen node where true success at freeze-time false
fz=[d for d in v.values() if d['guide_log'].get('evo_action')=='freeze']; print('evotest freeze trials success %.1f'%(100*st.mean(d['success'] for d in fz)))
ev=[d for d in v.values() if d['guide_log'].get('evo_action')=='evolve']; print('evotest evolve trials success %.1f'%(100*st.mean(d['success'] for d in ev)))
v=arms['awm']
for r in ROOMS:
    ks=sorted(k for k in v if k[0]==r)
    print('awm',r,'induced',sum(v[k]['method_log']['induced'] is not None for k in ks),'final guide chars',len(v[ks[-1]]['guide_used']), 'n workflows', v[ks[-1]]['method_state_after']['workflows'].count('Workflow'))
# per type
for a in ('reflexion','evotest','awm','mean','nomem'):
    c=defaultdict(list)
    for d in arms[a].values(): c[d['task_type']].append(d['success'])
    print(a,{k:round(100*st.mean(x)) for k,x in sorted(c.items())})
# steps / guide length
for m in ('reflexion','evotest','awm'):
    v=arms[m]; print(m,'mean steps %.1f'%st.mean(len(d['steps']) for d in v.values()),'guide chars %.0f'%st.mean(len(d['guide_used']) for d in v.values()))
print('--- AWM by room position (first half / second half), vs nomem')
for a in ('awm','nomem'):
    h=[[],[]]
    for r in ROOMS:
        tasks=sorted({k[1] for k in arms[a] if k[0]==r}, key=lambda i: i)
        n=len(tasks)
        for j,i in enumerate(tasks):
            h[j>=n//2]+= [arms[a][(r,i,t)]['success'] for t in range(1,6)]
    print(a,'%.1f / %.1f'%(100*st.mean(h[0]),100*st.mean(h[1])))

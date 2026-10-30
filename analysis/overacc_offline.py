import os, sys
from pathlib import Path as _P
ROOT = _P(__file__).resolve().parents[1]
os.chdir(ROOT)
import json, glob, collections as C, statistics as st
import sys; B = sys.argv[1] if len(sys.argv) > 1 else '1'
E = str(ROOT/'experiments')
R = ['kitchen', 'living_room', 'bathroom', 'bedroom']

def seq(pat, key):
    d = {}
    for f in glob.glob(pat):
        x = json.load(open(f)); d[key(x)] = x
    return d

it = seq(f'{E}/alfworld_ace_sd/runs/20261029_iter/beta_'+B+'/*/results/*.json', lambda x: (x['target'], x['task_index'], x['repeat']))
def other(arm):
    d = {}
    for r in R:
        for f in glob.glob(f'{E}/alfworld_react/runs/20261020/results/{arm}_{r}/*.json'):
            x = json.load(open(f)); d[(r, x.get('task_index', x.get('index')), x['repeat'])] = bool(x['success'])
    return d

# A. transitions by trial (k-1 -> k), vs noise baselines
print('A. transitions (k-1->k): T->F / #T(k-1), F->T / #F(k-1)')
for name, d in [('iter', {k: v['success'] for k, v in it.items()}), ('nomem', other('jitrl_nomem')), ('react', other('react'))]:
    row = []
    for k in range(2, 6):
        tf = sum(1 for (r, i, t), s in d.items() if t == k and d[(r, i, k - 1)] and not s)
        nt = sum(1 for (r, i, t), s in d.items() if t == k - 1 and s)
        ft = sum(1 for (r, i, t), s in d.items() if t == k and not d[(r, i, k - 1)] and s)
        nf = sum(1 for (r, i, t), s in d.items() if t == k - 1 and not s)
        row.append(f'k={k}: TF {tf}/{nt} FT {ft}/{nf}')
    tot_tf = sum(1 for (r, i, t), s in d.items() if t > 1 and d[(r, i, t - 1)] and not s)
    tot_t = sum(1 for (r, i, t), s in d.items() if t > 1 and d[(r, i, t - 1)])
    print(f'  {name:6s}', ' | '.join(row), f'| TF rate {100*tot_tf/tot_t:.1f}%')

# B/C/E. per-step decision analysis
def argmax(v): return max(range(len(v)), key=lambda i: v[i])
stats = C.defaultdict(lambda: C.Counter())
mag = C.defaultdict(list)
for (r, i, k), x in it.items():
    for s in x['steps']:
        lp, lqs = s['log_p'], s['log_q']
        n = len(lp)
        if n < 2 or not lqs: continue
        c = stats[k]; c['steps'] += 1
        cum = s['chosen']; base = s['base_choice']
        last = argmax([lp[a] + lqs[-1][a] - lp[a] for a in range(n)])  # = argmax q_last
        mean = argmax([lp[a] + st.mean(q[a] - lp[a] for q in lqs) for a in range(n)])
        singles = {argmax(q) for q in lqs}
        c['cum!=base'] += cum != base
        c['cum!=last'] += cum != last
        c['cum!=mean'] += cum != mean
        c['cum not any q argmax'] += cum not in singles
        c['q argmax disagree'] += len(singles) > 1
        if len(lqs) >= 2:
            c['q1==q2 argmax'] += argmax(lqs[0]) == argmax(lqs[1]); c['pairs'] += 1
        # margin of chosen over runner-up in score vs in q_last
        sc = sorted(s['scores'], reverse=True)
        mag[k].append(sc[0] - sc[1])
print('\nB. per-step choice disagreement (trial k uses k-1 teachers)')
for k in sorted(stats):
    c = stats[k]; n = c['steps']
    print(f'  k={k} steps={n} cum!=base {100*c["cum!=base"]/n:.1f}% | cum!=latest-only {100*c["cum!=last"]/n:.1f}% | cum!=mean {100*c["cum!=mean"]/n:.1f}% '
          f'| cum picks action no single q prefers {100*c["cum not any q argmax"]/n:.1f}% | teachers disagree {100*c["q argmax disagree"]/n:.1f}% '
          f'| q1,q2 same argmax {100*c["q1==q2 argmax"]/max(c["pairs"],1):.1f}% | median score margin {st.median(mag[k]):.2f}')

# D. trials k>=3 whose outcome flipped T->F: did cum differ from latest-only anywhere?
print('\nD. trials k>=3: share with >=1 step where cumulative != latest-only')
for lab, cond in [('T->F', lambda p, s: p and not s), ('T->T', lambda p, s: p and s), ('F->F', lambda p, s: not p and not s), ('F->T', lambda p, s: not p and s)]:
    tot = hit = 0
    for (r, i, k), x in it.items():
        if k < 3 or not cond(it[(r, i, k - 1)]['success'], x['success']): continue
        tot += 1
        hit += any(len(s['log_p']) > 1 and s['chosen'] != argmax(s['log_q'][-1]) for s in x['steps'])
    print(f'  {lab}: {hit}/{tot}')

# F. anti-base pressure: with beta=1, score = sum_j log q_j - (k-2) log p
print('\nF. where cumulative picks an action that no single teacher ranks first')
for k in range(2, 6):
    n = lowp = odd = odd_lowp = 0; pbar = []
    for (r, i, kk), x in it.items():
        if kk != k: continue
        for s in x['steps']:
            lp, lqs = s['log_p'], s['log_q']
            if len(lp) < 2: continue
            n += 1; c = s['chosen']
            is_low = c == min(range(len(lp)), key=lambda a: lp[a])
            lowp += is_low
            if c not in {argmax(q) for q in lqs}:
                odd += 1; odd_lowp += is_low
                pbar.append(sum(2.718281828 ** q[c] for q in lqs) / len(lqs))
    print(f'  k={k}: chosen=base-argmin {100*lowp/n:.1f}% | among "no teacher prefers" steps ({odd}): base-argmin {100*odd_lowp/max(odd,1):.1f}%, mean q_j(chosen) {st.mean(pbar) if pbar else 0:.3f}')

# G. per-step disagreement rate (cum != latest-only) by outcome, first 10 steps only (length control)
print('\nG. per-step rate cum != latest-only, steps 0-9, trials k>=3')
for lab, cond in [('T->F', lambda p, s: p and not s), ('T->T', lambda p, s: p and s), ('F->F', lambda p, s: not p and not s), ('F->T', lambda p, s: not p and s)]:
    n = h = 0
    for (r, i, k), x in it.items():
        if k < 3 or not cond(it[(r, i, k - 1)]['success'], x['success']): continue
        for s in x['steps'][:10]:
            if len(s['log_p']) < 2: continue
            n += 1; h += s['chosen'] != argmax(s['log_q'][-1])
    print(f'  {lab}: {100*h/n:.1f}% of {n} steps')

# H. ceiling: tasks never/always solved in trials 2-5, and how other arms do on them
nm, rc = other('jitrl_nomem'), other('react')
tasks = sorted({(r, i) for (r, i, k) in it})
hard = [t for t in tasks if not any(it[(*t, k)]['success'] for k in range(2, 6))]
part = [t for t in tasks if 0 < sum(it[(*t, k)]['success'] for k in range(2, 6)) < 4]
print(f'\nH. {len(tasks)} tasks: never solved in trials 2-5: {len(hard)}, partially solved: {len(part)}, always solved: {len(tasks)-len(hard)-len(part)}')
for name, d in [('nomem', nm), ('react', rc)]:
    print(f'  on the {len(hard)} never-solved tasks, {name} SR = {100*st.mean(d[(*t, k)] for t in hard for k in range(1, 6)):.0f}%')
print('  never-solved:', [(r[:4], i, it[(r, i, 1)]['task_type'].split('_')[1]) for r, i in hard])
print(f'  trials 2-5 SR ceiling if partially-solved tasks always succeeded: {100*(len(tasks)-len(hard))/len(tasks):.1f}% (task-weighted)')

print('\nI. never-solved tasks: trial-1 success, odd-pick rate by trial')
print('  trial1 success:', [int(it[(*t, 1)]['success']) for t in hard])
for grp, ts in [('never', hard), ('rest', [t for t in tasks if t not in hard])]:
    row = []
    for k in range(2, 6):
        n = odd = 0
        for t in ts:
            for s in it[(*t, k)]['steps']:
                if len(s['log_p']) < 2: continue
                n += 1; odd += s['chosen'] not in {argmax(q) for q in s['log_q']}
        row.append(f'k={k} {100*odd/n:.0f}%')
    print(f'  {grp}: odd-pick', ' '.join(row))
t = hard[0]
print('\n  sample feedback, task', t, 'trial 1 key_insight/correct_approach:')
fb = it[(*t, 1)]['reflection']
import re
for f in ('correct_approach', 'key_insight'):
    m = re.search(rf'"{f}":\s*"(.*?)"', fb, re.S); print('   ', f, ':', m.group(1)[:300] if m else None)

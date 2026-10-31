"""Collect every ALFWorld arm into one per-trial CSV and print the per-room Average / Final table.

  python analysis/export_trials.py   # writes results/alfworld_trials.csv
"""
import csv
import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments'
ROOMS = ('kitchen', 'living_room', 'bathroom', 'bedroom')
ARMS = {
    'react_0shot': 'alfworld_react/runs/20261020/results/react0_{room}/*.json',
    'react_2shot': 'alfworld_react/runs/20261020/results/react_{room}/*.json',
    'jitrl_memory_off': 'alfworld_react/runs/20261020/results/jitrl_nomem_{room}/*.json',
    **{f'ace_sd_beta{b}': f'alfworld_ace_sd/runs/20261022/beta_{b}/{{room}}/results/*.json'
       for b in ('0.25', '0.5', '0.75', '1', '1.5')},
    'iter_sd_sum_beta1': 'alfworld_ace_sd/runs/20261029_iter/beta_1/{room}/results/*.json',
    'iter_sd_sum_beta0.5': 'alfworld_ace_sd/runs/20261029_iter/beta_0.5/{room}/results/*.json',
    **{f'iter_sd_{r}_fixedfb_beta1': f'alfworld_ace_sd/runs/20261029_fixedfb/{r}/beta_1/{{room}}/results/*.json'
       for r in ('replace', 'mean', 'ema')},
    **{m: f'alfworld_baselines/runs/20261029/{m}/{{room}}/results/*.json' for m in ('reflexion', 'evotest', 'awm')},
    'jitrl_eq25_logit_judge': 'alfworld_jitrl/runs/20261022_eq25_logit/results/jitrl_to_{room}/*.json',
}


def main():
    task_type = {r['index']: r['task_type'] for r in json.loads((ROOT / 'data/alfworld_valid_seen_74.json').read_text())['splits']['valid_seen']}
    rows = []
    for arm, pattern in ARMS.items():
        for room in ROOMS:
            for f in glob.glob(str(EXP / pattern.format(room=room))):
                d = json.loads(Path(f).read_text())
                rows.append(dict(arm=arm, room=room, task_index=d['task_index'], task_type=task_type[d['task_index']],
                                 trial=d['repeat'], success=int(bool(d['success'])), steps=len(d['steps'])))
    rows.sort(key=lambda r: (r['arm'], ROOMS.index(r['room']), r['task_index'], r['trial']))
    out = ROOT / 'results/alfworld_trials.csv'
    out.parent.mkdir(exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} trials -> {out.relative_to(ROOT)}\n')
    print('| arm | ' + ' | '.join(ROOMS) + ' | mean |')
    print('|---|' + '---|' * (len(ROOMS) + 1))
    for arm in ARMS:
        cells, avgs, fins = [], [], []
        for room in ROOMS:
            rs = [r for r in rows if r['arm'] == arm and r['room'] == room]
            avg = 100 * st.mean(r['success'] for r in rs)
            fin = 100 * st.mean(r['success'] for r in rs if r['trial'] == 5)
            cells.append(f'{avg:.2f} / {fin:.2f}')
            avgs.append(avg)
            fins.append(fin)
        n = sum(r['arm'] == arm for r in rows)
        assert n == 370, (arm, n)
        print(f'| {arm} | ' + ' | '.join(cells) + f' | {st.mean(avgs):.2f} / {st.mean(fins):.2f} |')


if __name__ == '__main__':
    main()

"""Progress, per-room Average / Final table and integrity checks for iterative-ACE runs (mean / EMA).

Works on a finished or a running RUN_DIR (from launch_ace_iter.py); several RUN_DIRs can be
given, e.g. the valid_seen folder from one machine and the valid_unseen folder from another.

  python analysis/agg_ace_iter.py experiments/alfworld_ace_sd/runs_new/ace_iter
  python analysis/agg_ace_iter.py RUN_DIR --check   # also recompute every step's score

--check verifies, per stream: trial k of a task has k teachers whose snapshots are the
playbook_before of trials 1..k; an empty snapshot gives log q = log p; the stored scores
equal log p + beta * combine(log q - log p) and the chosen candidate is their argmax; the
playbook chain is unbroken (playbook_before of a trial = playbook_after of the previous one).
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments/alfworld_ace_sd'), str(ROOT / 'ALFWorld')]
from agent_iter import combine  # noqa: E402

ROOMS = ('kitchen', 'living_room', 'bathroom', 'bedroom')
PLANS = {'valid_seen': ROOT / 'data/alfworld_valid_seen_74.json', 'valid_unseen': ROOT / 'data/alfworld_valid_unseen_59.json'}


def has_bullets(text):
    return any(line.lstrip().startswith('[') for line in text.splitlines())


def check_stream(trials):
    """trials: committed trials of one stream, in run order. Returns (steps, problems)."""
    problems, steps = [], 0
    for a, b in zip(trials, trials[1:]):
        if b['playbook_before'] != a['playbook_after']:
            problems.append(f"playbook chain broken before task {b['task_index']} trial {b['repeat']}")
    by_task = {}
    for d in trials:
        by_task.setdefault(d['task_index'], []).append(d)
    for task, ds in by_task.items():
        ds.sort(key=lambda d: d['repeat'])
        for d in ds:
            k, snaps = d['repeat'], [x['playbook_before'] for x in ds[:d['repeat']]]
            if d['n_teachers'] != k:
                problems.append(f'task {task} trial {k}: n_teachers={d["n_teachers"]}')
            for s in d['steps']:
                steps += 1
                if len(s['log_q']) != k:
                    problems.append(f'task {task} trial {k} step {s["index"]}: {len(s["log_q"])} teachers')
                    continue
                if any(lq != s['log_p'] for lq, m in zip(s['log_q'], snaps) if not has_bullets(m)):
                    problems.append(f'task {task} trial {k} step {s["index"]}: empty snapshot with q != p')
                ratios = [[q - p for q, p in zip(lq, s['log_p'])] for lq in s['log_q']]
                exp = [p + d['beta'] * x for p, x in zip(s['log_p'], combine(ratios, d['rule'], d['lam']))]
                if any(abs(x - y) > 1e-9 for x, y in zip(exp, s['scores'])):
                    problems.append(f'task {task} trial {k} step {s["index"]}: score mismatch')
                if max(range(len(exp)), key=exp.__getitem__) != s['chosen']:
                    problems.append(f'task {task} trial {k} step {s["index"]}: argmax mismatch')
    return steps, problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dirs', nargs='+', type=Path)
    p.add_argument('--check', action='store_true')
    args = p.parse_args()

    for run_dir in args.run_dirs:
        for split, plan_path in PLANS.items():
            split_dir = run_dir / split
            if not split_dir.exists():
                continue
            plan = json.loads(plan_path.read_text())
            rows = plan['splits'][split]
            target = {r: plan['eval_repeats'] * sum(x['room_type'] == r for x in rows) for r in ROOMS}
            print(f'\n## {split} ({run_dir})  tasks {len(rows)}, trials per rule {sum(target.values())}\n')
            print('| rule | ' + ' | '.join(f'{r} (n/{target[r]})' for r in ROOMS) + ' | mean (complete rooms) |')
            print('|---|' + '---|' * (len(ROOMS) + 1))
            for rule_dir in sorted(x for x in split_dir.iterdir() if (x / 'beta_1').is_dir()):
                cells, avgs, fins, issues = [], [], [], []
                for room in ROOMS:
                    files = sorted((rule_dir / 'beta_1' / room / 'results').glob('*.json'))
                    ds = sorted((json.loads(f.read_text()) for f in files), key=lambda d: d['episode_index'])
                    if not ds:
                        cells.append(f'0/{target[room]}')
                        continue
                    avg = 100 * st.mean(d['success'] for d in ds)
                    last = [d['success'] for d in ds if d['repeat'] == plan['eval_repeats']]
                    fin = f'{100 * st.mean(last):.2f}' if last else '-'
                    cells.append(f'{avg:.2f} / {fin} ({len(ds)}/{target[room]})')
                    if len(ds) == target[room]:
                        avgs.append(avg)
                        fins.append(100 * st.mean(last))
                    if args.check:
                        n, problems = check_stream(ds)
                        issues += [f'{room}: {x}' for x in problems]
                        print(f'  check {rule_dir.name}/{room}: {len(ds)} trials, {n} steps, {len(problems)} problems',
                              file=sys.stderr)
                total = f'{st.mean(avgs):.2f} / {st.mean(fins):.2f} ({len(avgs)}/4 rooms)' if avgs else '-'
                print(f'| {rule_dir.name} | ' + ' | '.join(cells) + f' | {total} |')
                for x in issues[:20]:
                    print(f'  PROBLEM {rule_dir.name} {x}', file=sys.stderr)
    print('\nCells: Average / Final success (%) and committed trials. Average = all 5 trials, Final = trial 5.')


if __name__ == '__main__':
    main()

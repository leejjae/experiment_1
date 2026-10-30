"""Print how the ACE playbook changes over the 5 trials of one task.

  python experiments/alfworld_ace_sd/show_playbook.py --task 58 --room kitchen --beta 1
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=int, required=True)
    p.add_argument('--room', required=True, choices=('kitchen', 'living_room', 'bathroom', 'bedroom'))
    p.add_argument('--beta', default='1')
    p.add_argument('--run', type=Path, default=HERE / 'runs/20261022')
    p.add_argument('--full', action='store_true', help='print the whole playbook after every changed trial')
    args = p.parse_args()

    files = sorted((args.run / f'beta_{args.beta}' / args.room / 'results').glob(f'task_{args.task:04d}_run_*.json'))
    for i, f in enumerate(files):
        d = json.loads(f.read_text())
        if i == 0:
            print(f"TASK {args.task}: {d['goal']}\n\n=== playbook before trial 1 ===\n{d['playbook_before']}\n")
        print(f"=== trial {d['repeat']}  success={d['success']}  steps={len(d['steps'])} ===")
        print('actions: ' + ' | '.join(s['action'] for s in d['steps']))
        refl = d['reflection']
        try:
            refl = json.loads(refl)
            print(f"reflector error: {refl.get('error_identification')}\nreflector insight: {refl.get('key_insight')}")
        except (TypeError, ValueError):
            print(f'reflector: {refl}')
        if d['operations']:
            for op in d['operations']:
                print(f"curator {op['type']} [{op.get('section')}]: {op.get('content')}")
        else:
            print('curator: no change')
        if args.full and d['playbook_after'] != d['playbook_before']:
            print(f"--- playbook after ---\n{d['playbook_after']}")
        print()


if __name__ == '__main__':
    main()

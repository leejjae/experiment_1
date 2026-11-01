"""Supervise the iterative-ACE streams of one split: rules (mean, ema) x 4 rooms = 8 streams, beta = 1, lam = 0.5.

Each stream is sequential (the room playbook depends on earlier trials, as in ACE-SD);
streams are independent and spread over the actor endpoints. A crashed stream is
restarted (run_ace_iter.py resumes after its last committed trial), at most 3 times.
SPLIT is required (valid_seen or valid_unseen); the two splits are meant to run on
separate machines, each with its own RUN_DIR, and are merged afterwards by copying
RUN_DIR/<split> folders side by side (see docs/alfworld_ace_sd/ace_iter.md).

  SPLIT=valid_seen RUN_DIR=runs_new/ace_iter nohup python experiments/alfworld_ace_sd/launch_ace_iter.py > launch_ace_iter.log 2>&1 &
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from run_ace_iter import PLANS  # noqa: E402

PYTHON = sys.executable
SPLIT = os.environ.get('SPLIT')
assert SPLIT in PLANS, f'Set SPLIT to one of {tuple(PLANS)}'
OUT = Path(os.environ.get('RUN_DIR', str(HERE / 'runs/new_ace_iter'))).resolve()
BETA, LAM = 1.0, 0.5
RULES = ('mean', 'ema')
ROOMS = ('kitchen', 'living_room', 'bathroom', 'bedroom')
PORTS = tuple(int(p) for p in os.environ.get('VLLM_PORTS', '8000,8002').split(','))  # one vLLM server per port
PLAN = json.loads(PLANS[SPLIT][0].read_text())
TARGET = {r: PLAN['eval_repeats'] * sum(x['room_type'] == r for x in PLAN['splits'][SPLIT]) for r in ROOMS}
SOURCES = [HERE / f for f in ('run_ace_iter.py', 'agent_ace_iter.py', 'agent_iter.py', 'run.py', 'agent.py',
                              'playbook.py', 'launch_ace_iter.py', 'prompts/reflector.txt', 'prompts/curator.txt')] + \
          [ROOT / 'ALFWorld/src' / f for f in ('jitrl_agent.py', 'utils.py', 'env.py', 'action_utils.py', 'openai_helpers.py')] + \
          [PLANS[SPLIT][0]]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    split_dir = OUT / SPLIT
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest = split_dir / 'manifest.json'
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in SOURCES}
    if manifest.exists():
        old = json.loads(manifest.read_text())
        assert old['split'] == SPLIT, (old['split'], SPLIT)
        changed = [p for p in hashes if old['source_sha256'].get(p) != hashes[p]]
        assert not changed, f'Source changed since start: {changed}'
    else:
        manifest.write_text(json.dumps(dict(created_utc=now(), split=SPLIT, rules=RULES, beta=BETA, lam=LAM, rooms=ROOMS,
                                            ports=PORTS, target_trials=TARGET, total_trials=len(RULES) * sum(TARGET.values()),
                                            protocol=f'{SPLIT} {len(PLAN["splits"][SPLIT])} x {PLAN["eval_repeats"]}, '
                                                     f'order/seeds/actor/T/steps as {PLANS[SPLIT][0].name}; '
                                                     'room playbook updated on-policy by Reflector + Curator after every trial (as ACE-SD); '
                                                     'trial k of a task combines the teachers of the snapshots before trials 1..k',
                                            source_sha256=hashes), indent=2) + '\n')
    streams = [(rule, r) for rule in RULES for r in ROOMS]
    logs = split_dir / 'logs'
    logs.mkdir(exist_ok=True)
    active, attempts = {}, {}
    print(f'{now()} START split={SPLIT} out={OUT} ports={PORTS} target={TARGET}', flush=True)

    def done(rule, r):
        return len(list((split_dir / rule / f'beta_{BETA:g}' / r / 'results').glob('*.json')))

    while True:
        for key, p in list(active.items()):
            if p.poll() is None:
                continue
            del active[key]
            if done(*key) < TARGET[key[1]]:
                attempts[key] = attempts.get(key, 0) + 1
                print(f'{now()} RECOVERY {key[0]} {key[1]} rc={p.returncode} attempt={attempts[key]}', flush=True)
                if attempts[key] > 3:
                    print(f'{now()} GIVE_UP {key[0]} {key[1]}', flush=True)
        if all(done(*k) >= TARGET[k[1]] for k in streams):
            print(f'{now()} COMPLETE', flush=True)
            return
        for i, key in enumerate(streams):
            if key in active or done(*key) >= TARGET[key[1]] or attempts.get(key, 0) > 3:
                continue
            rule, r = key
            cmd = [PYTHON, '-u', str(HERE / 'run_ace_iter.py'), '--split', SPLIT, '--rule', rule, '--beta', f'{BETA:g}',
                   '--lam', f'{LAM:g}', '--room', r, '--port', str(PORTS[i % len(PORTS)]), '--out', str(OUT)]
            with (logs / f'{rule}_{r}.log').open('a') as f:
                f.write(f'\nSTART {now()}\n')
                f.flush()
                active[key] = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        if not active:
            print(f'{now()} STOPPED (streams gave up)', flush=True)
            sys.exit(1)
        time.sleep(30)


if __name__ == '__main__':
    main()

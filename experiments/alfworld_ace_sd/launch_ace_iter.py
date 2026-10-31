"""Supervise the iterative-ACE streams: rules (mean, ema) x 4 rooms = 8 streams, beta = 1, lam = 0.5.

Each stream is sequential (the room playbook depends on earlier trials, as in ACE-SD);
streams are independent and spread over the actor endpoints. A crashed stream is
restarted (run_ace_iter.py resumes after its last committed trial), at most 3 times.

  RUN_DIR=runs_new/ace_iter nohup python experiments/alfworld_ace_sd/launch_ace_iter.py > launch_ace_iter.log 2>&1 &
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
PYTHON = sys.executable
OUT = Path(os.environ.get('RUN_DIR', str(HERE / 'runs/new_ace_iter'))).resolve()
BETA, LAM = 1.0, 0.5
RULES = ('mean', 'ema')
ROOMS = ('kitchen', 'living_room', 'bathroom', 'bedroom')
PORTS = tuple(int(p) for p in os.environ.get('VLLM_PORTS', '8000,8002').split(','))  # one vLLM server per port
TARGET = {'kitchen': 100, 'living_room': 70, 'bathroom': 100, 'bedroom': 100}
SOURCES = [HERE / f for f in ('run_ace_iter.py', 'agent_ace_iter.py', 'agent_iter.py', 'run.py', 'agent.py',
                              'playbook.py', 'launch_ace_iter.py', 'prompts/reflector.txt', 'prompts/curator.txt')] + \
          [ROOT / 'ALFWorld/src' / f for f in ('jitrl_agent.py', 'utils.py', 'env.py', 'action_utils.py', 'openai_helpers.py')]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / 'manifest.json'
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in SOURCES}
    if manifest.exists():
        old = json.loads(manifest.read_text())['source_sha256']
        assert old == hashes, f'Source changed since start: {[p for p in hashes if old.get(p) != hashes[p]]}'
    else:
        manifest.write_text(json.dumps(dict(created_utc=now(), rules=RULES, beta=BETA, lam=LAM, rooms=ROOMS, ports=PORTS,
                                            protocol='valid_seen 74 x 5, order/seeds/actor/T/steps as data/alfworld_valid_seen_74.json; '
                                                     'room playbook updated on-policy by Reflector + Curator after every trial (as ACE-SD); '
                                                     'trial k of a task combines the teachers of the snapshots before trials 1..k',
                                            source_sha256=hashes), indent=2) + '\n')
    streams = [(rule, r) for rule in RULES for r in ROOMS]
    logs = OUT / 'logs'
    logs.mkdir(exist_ok=True)
    active, attempts = {}, {}

    def done(rule, r):
        return len(list((OUT / rule / f'beta_{BETA:g}' / r / 'results').glob('*.json')))

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
            cmd = [PYTHON, '-u', str(HERE / 'run_ace_iter.py'), '--rule', rule, '--beta', f'{BETA:g}', '--lam', f'{LAM:g}',
                   '--room', r, '--port', str(PORTS[i % len(PORTS)]), '--out', str(OUT)]
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

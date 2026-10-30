"""Supervise all ACE + self-distillation streams: 5 betas x 4 rooms = 20 streams.

Each stream is sequential (the playbook depends on earlier trials); streams are
independent and spread over the actor endpoints. A crashed stream is restarted
(run.py resumes after its last committed trial), at most 3 times.

  nohup python experiments/alfworld_ace_sd/launch.py > experiments/alfworld_ace_sd/logs/launch.log 2>&1 &
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
OUT = Path(os.environ.get('RUN_DIR', str(HERE / 'runs/20261022'))).resolve()  # published run; set RUN_DIR for a new one
BETAS = (0.25, 0.5, 0.75, 1.0, 1.5)
ROOMS = ('kitchen', 'living_room', 'bathroom', 'bedroom')
PORTS = tuple(int(p) for p in os.environ.get('VLLM_PORTS', '8000,8002').split(','))  # one vLLM server per port
TARGET = {'kitchen': 100, 'living_room': 70, 'bathroom': 100, 'bedroom': 100}
SOURCES = [HERE / f for f in ('run.py', 'agent.py', 'playbook.py', 'launch.py', 'prompts/reflector.txt', 'prompts/curator.txt')] + \
          [ROOT / 'ALFWorld/src' / f for f in ('jitrl_agent.py', 'utils.py', 'env.py', 'action_utils.py', 'openai_helpers.py')]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / 'manifest.json'
    hashes = {str(p): digest(p) for p in SOURCES}
    if manifest.exists():
        old = json.loads(manifest.read_text())['source_sha256']
        assert old == hashes, f'Source changed since start: {[p for p in hashes if old.get(p) != hashes[p]]}'
    else:
        manifest.write_text(json.dumps(dict(created_utc=now(), betas=BETAS, rooms=ROOMS, ports=PORTS,
                                            protocol='valid_seen 74 x 5, order/seeds/actor/T/steps as data/alfworld_valid_seen_74.json',
                                            source_sha256=hashes), indent=2) + '\n')
    streams = [(b, r) for b in BETAS for r in ROOMS]
    logs = HERE / 'logs'
    logs.mkdir(exist_ok=True)
    active, attempts = {}, {}

    def done(b, r):
        return len(list((OUT / f'beta_{b:g}' / r / 'results').glob('*.json')))

    while True:
        for key, p in list(active.items()):
            if p.poll() is None:
                continue
            del active[key]
            if done(*key) < TARGET[key[1]]:
                attempts[key] = attempts.get(key, 0) + 1
                print(f'{now()} RECOVERY beta={key[0]:g} {key[1]} rc={p.returncode} attempt={attempts[key]}', flush=True)
                if attempts[key] > 3:
                    print(f'{now()} GIVE_UP beta={key[0]:g} {key[1]}', flush=True)
        if all(done(*k) >= TARGET[k[1]] for k in streams):
            print(f'{now()} COMPLETE', flush=True)
            return
        for i, key in enumerate(streams):
            if key in active or done(*key) >= TARGET[key[1]] or attempts.get(key, 0) > 3:
                continue
            b, r = key
            cmd = [PYTHON, '-u', str(HERE / 'run.py'), '--beta', f'{b:g}', '--room', r,
                   '--port', str(PORTS[i % len(PORTS)]), '--out', str(OUT)]
            with (logs / f'beta_{b:g}_{r}.log').open('a') as f:
                f.write(f'\nSTART {now()}\n')
                f.flush()
                active[key] = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        if not active:
            print(f'{now()} STOPPED (streams gave up)', flush=True)
            sys.exit(1)
        time.sleep(30)


if __name__ == '__main__':
    main()

"""Supervise the Reflexion / EvoTest / AWM runs on ALFWorld (valid_seen 74 x 5).

Reflexion and EvoTest keep memory per task, so each room is split by task over
SHARDS streams; AWM keeps one workflow set per room and runs one stream per room.
A crashed stream is restarted (run_baseline.py resumes after its last trial), at most 3 times.

  nohup python experiments/alfworld_baselines/launch.py > experiments/alfworld_baselines/logs/launch.log 2>&1 &
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
OUT = Path(os.environ.get('RUN_DIR', str(HERE / 'runs/20261029'))).resolve()  # published run; set RUN_DIR for a new one
SHARDS = {'reflexion': 4, 'evotest': 4, 'awm': 1}
ROOMS = ('kitchen', 'living_room', 'bathroom', 'bedroom')
PORTS = tuple(int(p) for p in os.environ.get('VLLM_PORTS', '8000,8002').split(','))  # one vLLM server per port
PLAN = ROOT / 'data/alfworld_valid_seen_74.json'
SOURCES = [HERE / f for f in ('run_baseline.py', 'methods.py', 'launch.py')] + sorted((HERE / 'prompts').glob('*.txt')) + \
          [ROOT / 'experiments/alfworld_ace_sd/run.py'] + \
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
        manifest.write_text(json.dumps(dict(created_utc=now(), shards=SHARDS, rooms=ROOMS, ports=PORTS,
                                            protocol='valid_seen 74 x 5, order/seeds/actor/T/steps as data/alfworld_valid_seen_74.json; '
                                                     'no env outcome to any role, LLM judge where a method needs success',
                                            source_sha256=hashes), indent=2) + '\n')
    plan = json.loads(PLAN.read_text())
    tasks = {r: [x['index'] for x in plan['splits']['valid_seen'] if x['room_type'] == r] for r in ROOMS}
    streams = [(m, r, s) for m in SHARDS for r in ROOMS for s in range(SHARDS[m])]
    logs = HERE / 'logs'
    logs.mkdir(exist_ok=True)
    active, attempts = {}, {}

    def target(m, r, s):
        return len(tasks[r][s::SHARDS[m]]) * plan['eval_repeats']

    def done(m, r, s):
        res = OUT / m / r / 'results'
        return sum(len(list(res.glob(f'task_{i:04d}_run_*.json'))) for i in tasks[r][s::SHARDS[m]])

    while True:
        for key, p in list(active.items()):
            if p.poll() is None:
                continue
            del active[key]
            if done(*key) < target(*key):
                attempts[key] = attempts.get(key, 0) + 1
                print(f'{now()} RECOVERY {key} rc={p.returncode} attempt={attempts[key]}', flush=True)
                if attempts[key] > 3:
                    print(f'{now()} GIVE_UP {key}', flush=True)
        if all(done(*k) >= target(*k) for k in streams):
            print(f'{now()} COMPLETE', flush=True)
            return
        for i, key in enumerate(streams):
            if key in active or done(*key) >= target(*key) or attempts.get(key, 0) > 3:
                continue
            m, r, s = key
            cmd = [PYTHON, '-u', str(HERE / 'run_baseline.py'), '--method', m, '--room', r, '--shard', str(s),
                   '--shards', str(SHARDS[m]), '--port', str(PORTS[i % len(PORTS)]), '--out', str(OUT)]
            with (logs / f'{m}_{r}_{s}.log').open('a') as f:
                f.write(f'\nSTART {now()}\n')
                f.flush()
                active[key] = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        if not active:
            print(f'{now()} STOPPED (streams gave up)', flush=True)
            sys.exit(1)
        time.sleep(30)


if __name__ == '__main__':
    main()

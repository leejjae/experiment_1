"""Supervise one iterative self-distillation run (one beta): 4 rooms x SHARDS task shards.

Tasks are independent (feedback resets per task), so each room is split by task
over SHARDS streams spread across the actor endpoints. A crashed stream is
restarted (run_iter.py resumes per task), at most 3 times.

  nohup python experiments/alfworld_ace_sd/launch_iter.py 0.5 > experiments/alfworld_ace_sd/logs/launch_iter_0.5.log 2>&1 &
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
OUT = Path(os.environ.get('RUN_DIR', str(HERE / 'runs/20261029_iter'))).resolve()  # published run; set RUN_DIR for a new one
BETA = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
ROOMS = ('kitchen', 'living_room', 'bathroom', 'bedroom')
SHARDS = 4
PORTS = tuple(int(p) for p in os.environ.get('VLLM_PORTS', '8000,8002').split(','))  # one vLLM server per port
PLAN = ROOT / 'data/alfworld_valid_seen_74.json'
SOURCES = [HERE / f for f in ('run_iter.py', 'agent_iter.py', 'run.py', 'agent.py', 'launch_iter.py', 'prompts/reflector.txt')] + \
          [ROOT / 'ALFWorld/src' / f for f in ('jitrl_agent.py', 'utils.py', 'env.py', 'action_utils.py', 'openai_helpers.py')]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # beta=1 (first run) kept the original manifest name.
    manifest = OUT / ('manifest.json' if BETA == 1.0 else f'manifest_beta_{BETA:g}.json')
    hashes = {str(p): digest(p) for p in SOURCES}
    if manifest.exists():
        old = json.loads(manifest.read_text())['source_sha256']
        assert old == hashes, f'Source changed since start: {[p for p in hashes if old.get(p) != hashes[p]]}'
    else:
        manifest.write_text(json.dumps(dict(created_utc=now(), beta=BETA, rooms=ROOMS, shards=SHARDS, ports=PORTS,
                                            protocol='valid_seen 74 x 5, order/seeds/actor/T/steps as data/alfworld_valid_seen_74.json',
                                            source_sha256=hashes), indent=2) + '\n')
    plan = json.loads(PLAN.read_text())
    tasks = {r: [x['index'] for x in plan['splits']['valid_seen'] if x['room_type'] == r] for r in ROOMS}
    streams = [(r, s) for r in ROOMS for s in range(SHARDS)]
    logs = HERE / 'logs'
    logs.mkdir(exist_ok=True)
    active, attempts = {}, {}

    def target(r, s):
        return len(tasks[r][s::SHARDS]) * plan['eval_repeats']

    def done(r, s):
        res = OUT / f'beta_{BETA:g}' / r / 'results'
        return sum(len(list(res.glob(f'task_{i:04d}_run_*.json'))) for i in tasks[r][s::SHARDS])

    while True:
        for key, p in list(active.items()):
            if p.poll() is None:
                continue
            del active[key]
            if done(*key) < target(*key):
                attempts[key] = attempts.get(key, 0) + 1
                print(f'{now()} RECOVERY {key[0]} shard={key[1]} rc={p.returncode} attempt={attempts[key]}', flush=True)
                if attempts[key] > 3:
                    print(f'{now()} GIVE_UP {key[0]} shard={key[1]}', flush=True)
        if all(done(*k) >= target(*k) for k in streams):
            print(f'{now()} COMPLETE', flush=True)
            return
        for i, key in enumerate(streams):
            if key in active or done(*key) >= target(*key) or attempts.get(key, 0) > 3:
                continue
            r, s = key
            cmd = [PYTHON, '-u', str(HERE / 'run_iter.py'), '--beta', f'{BETA:g}', '--room', r,
                   '--shard', str(s), '--shards', str(SHARDS), '--port', str(PORTS[i % len(PORTS)]), '--out', str(OUT)]
            with (logs / (f'iter_{r}_{s}.log' if BETA == 1.0 else f'iter_b{BETA:g}_{r}_{s}.log')).open('a') as f:
                f.write(f'\nSTART {now()}\n')
                f.flush()
                active[key] = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        if not active:
            print(f'{now()} STOPPED (streams gave up)', flush=True)
            sys.exit(1)
        time.sleep(30)


if __name__ == '__main__':
    main()

"""Supervise memory-on JitRL (Eq. 25 + logit, judge-only reward) on ALFWorld (valid_seen 74 x 5).

One stream per room (the memory is shared within a room). A crashed stream is
restarted (run.py restores the memory snapshot and resumes), at most 3 times.

Servers: actor Qwen3-32B-AWQ (`qwen3-32b`) and judge gpt-oss-20b (`gpt-oss-20b`),
see scripts/serve_vllm.sh jitrl | jitrl-tp. Environment variables:
  RUN_DIR       output folder (default: runs/new)
  ACTOR_PORTS   actor vLLM port(s), comma separated (default 8000)
  JUDGE_URL     judge endpoint (default http://localhost:8001/v1)
  JUDGE_MODEL   judge served-model name (default gpt-oss-20b; the reported run used it)
  ARM           eq25_logit (default) or eq25

  RUN_DIR=runs_new/jitrl nohup python experiments/alfworld_jitrl/launch.py > launch_jitrl.log 2>&1 &
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = sys.executable
OUT = Path(os.environ.get('RUN_DIR', str(HERE / 'runs/new'))).resolve()
ARM = os.environ.get('ARM', 'eq25_logit')
ROOMS = ('kitchen', 'living_room', 'bathroom', 'bedroom')
ACTOR_PORTS = tuple(int(p) for p in os.environ.get('ACTOR_PORTS', '8000').split(','))
JUDGE_URL = os.environ.get('JUDGE_URL', 'http://localhost:8001/v1')
JUDGE_MODEL = os.environ.get('JUDGE_MODEL', 'gpt-oss-20b')
PLAN = ROOT / 'data/alfworld_valid_seen_74.json'
SOURCES = [HERE / f for f in ('run.py', 'agent.py', 'upstream_retrieval.py', 'judge.py', 'launch.py')] + \
          [ROOT / 'experiments/alfworld_no_feedback/agent.py', ROOT / 'ALFWorld/main.py'] + \
          [ROOT / 'ALFWorld/src' / f for f in ('jitrl_agent.py', 'cross_episode_memory.py', 'utils.py', 'env.py',
                                                'state_construction.py', 'action_utils.py', 'openai_helpers.py')]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_server(url, model):
    with urllib.request.urlopen(f'{url}/models', timeout=15) as f:
        served = [m['id'] for m in json.load(f)['data']]
    assert model in served, f'{url} serves {served}, expected {model}'


def main():
    for port in ACTOR_PORTS:
        check_server(f'http://localhost:{port}/v1', 'qwen3-32b')
    check_server(JUDGE_URL, JUDGE_MODEL)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / 'manifest.json'
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in SOURCES}
    settings = dict(arm=ARM, judge_model=JUDGE_MODEL, judge_reward_scale=0.1)
    if manifest.exists():
        old = json.loads(manifest.read_text())
        assert old['source_sha256'] == hashes, \
            f"Source changed since start: {[p for p in hashes if old['source_sha256'].get(p) != hashes[p]]}"
        assert old['settings'] == settings, f"Settings changed since start: {old['settings']} -> {settings}"
    else:
        manifest.write_text(json.dumps(dict(
            created_utc=now(), settings=settings, rooms=ROOMS, actor_ports=ACTOR_PORTS, judge_url=JUDGE_URL,
            protocol='valid_seen 74 x 5, order/seeds/actor/T/steps as data/alfworld_valid_seen_74.json; '
                     'JitRL memory per room (fresh, empty), Eq. 25 advantage normalization, WebArena upstream '
                     'retrieval k=3, rewards = judge step scores x 0.1; no env outcome to actor, tracker, judge or memory',
            source_sha256=hashes), indent=2) + '\n')
    plan = json.loads(PLAN.read_text())
    target = {r: plan['eval_repeats'] * sum(x['room_type'] == r for x in plan['splits']['valid_seen']) for r in ROOMS}
    logs = OUT / 'logs'
    logs.mkdir(exist_ok=True)
    active, attempts = {}, {}

    def done(room):
        return len(list((OUT / room / 'results').glob('*.json')))

    while True:
        for room, p in list(active.items()):
            if p.poll() is None:
                continue
            del active[room]
            if done(room) < target[room]:
                attempts[room] = attempts.get(room, 0) + 1
                print(f'{now()} RECOVERY {room} rc={p.returncode} attempt={attempts[room]}', flush=True)
                if attempts[room] > 3:
                    print(f'{now()} GIVE_UP {room}', flush=True)
        if all(done(r) >= target[r] for r in ROOMS):
            print(f'{now()} COMPLETE {dict((r, done(r)) for r in ROOMS)}', flush=True)
            return
        for i, room in enumerate(ROOMS):
            if room in active or done(room) >= target[room] or attempts.get(room, 0) > 3:
                continue
            cmd = [PYTHON, '-u', str(HERE / 'run.py'), '--room', room, '--out', str(OUT), '--arm', ARM,
                   '--port', str(ACTOR_PORTS[i % len(ACTOR_PORTS)]),
                   '--judge_model', JUDGE_MODEL, '--judge_url', JUDGE_URL]
            with (logs / f'{room}.log').open('a') as f:
                f.write(f'\nSTART {now()}\n')
                f.flush()
                active[room] = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                                start_new_session=True)
        if not active:
            print(f'{now()} STOPPED (streams gave up)', flush=True)
            sys.exit(1)
        time.sleep(30)


if __name__ == '__main__':
    main()

"""Run memory-on JitRL (judge-only reward) on one ALFWorld room.

Agent = Eq. 25 advantage normalization + WebArena upstream retrieval (k = 3) +
judge-only scaled rewards (experiments/alfworld_no_feedback/agent.py). The
environment outcome never reaches the actor, the state tracker, the judge or the
memory; it is only recorded in the result JSON for evaluation.

Tasks, order, repeats, step budget, temperature and seeds come from
data/alfworld_valid_seen_74.json. One stream per room, because the memory is
shared across all tasks and trials of the room (fresh, empty memory per room).

Resume: before each trial the memory files are snapshotted. If the stream dies
mid-trial, the next start restores the snapshot and reruns that trial from its seed.

  python experiments/alfworld_jitrl/run.py --room kitchen --out runs_new/jitrl
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ALF = ROOT / 'ALFWorld'
sys.path.insert(0, str(ALF))
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from main import build_parser
from src.env import AlfWorldEnvFactory
from src.evaluation import StateNode
from experiments.alfworld_jitrl import judge
from experiments.alfworld_jitrl.agent import PaperNormalizationAgent

PLAN = ROOT / 'data/alfworld_valid_seen_74.json'
SPLIT = 'valid_seen'
ROOMS = ('kitchen', 'living_room', 'bathroom', 'bedroom')
MEMORY_FILES = ('episodes.jsonl', 'step_metadata.pkl', 'prompt_history.jsonl')
JUDGE_REWARD_SCALE = 0.1  # judge scores in [-10, 10] -> rewards in [-1, 1]


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def restore_snapshot(stream_dir, memory_dir):
    """Undo the memory writes of a trial that did not commit."""
    marker = stream_dir / 'transaction.json'
    if not marker.exists():
        return
    if not Path(json.loads(marker.read_text())['result']).exists():
        for name in MEMORY_FILES:
            source, target = stream_dir / 'snapshot' / name, memory_dir / name
            if source.exists():
                shutil.copy2(source, target)
            elif target.exists():
                target.unlink()
        print('RECOVERY restored the memory snapshot of an uncommitted trial', flush=True)
    marker.unlink()


def begin_transaction(stream_dir, memory_dir, result):
    snap = stream_dir / 'snapshot'
    shutil.rmtree(snap, ignore_errors=True)
    snap.mkdir(parents=True)
    for name in MEMORY_FILES:
        if (memory_dir / name).exists():
            shutil.copy2(memory_dir / name, snap / name)
    atomic(stream_dir / 'transaction.json', dict(result=str(result)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--room', required=True, choices=ROOMS)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arm', default='eq25_logit', choices=['eq25_logit', 'eq25'])
    p.add_argument('--port', type=int, default=8000, help='actor (Qwen3-32B-AWQ) vLLM port')
    p.add_argument('--judge_model', default='gpt-oss-20b')
    p.add_argument('--judge_url', default='http://localhost:8001/v1')
    p.add_argument('--limit_tasks', type=int, default=None, help='smoke test: first N tasks of the room')
    p.add_argument('--repeats', type=int, default=None, help='override plan eval_repeats (smoke test)')
    args = p.parse_args()

    plan = json.loads(PLAN.read_text())
    repeats = args.repeats or plan['eval_repeats']
    out = args.out.resolve()
    stream_dir = out / args.room
    results = stream_dir / 'results'

    cfg = build_parser().parse_args(['--room_type', args.room])
    cfg.arm = f'jitrl_{args.arm}'
    cfg.normalization_arm = args.arm
    cfg.judge_reward_scale = JUDGE_REWARD_SCALE
    cfg.output_path = str(stream_dir / 'memory')
    cfg.enable_cross_mem = True
    cfg.env_step_limit = plan['max_steps']
    cfg.llm_temperature = plan['eval_temperature']
    cfg.llm_base_url = f'http://localhost:{args.port}/v1'
    cfg.eval_llm_model = args.judge_model
    cfg.eval_base_url = args.judge_url
    os.environ.setdefault('ALFWORLD_DATA', str(Path.home() / '.cache/alfworld'))  # alfworld-download default
    factory = AlfWorldEnvFactory(yaml.safe_load((ALF / 'configs/base_config.yaml').read_text()),
                                 split='eval_in_distribution')
    call_errors = judge.install(stream_dir / 'judge_audit')

    rows = [r for r in plan['splits'][SPLIT] if r['room_type'] == args.room][:args.limit_tasks]
    pairs = [(r, rep) for r in rows for rep in range(1, repeats + 1)]
    dests = [results / f"{r['index']:04d}_{rep}.json" for r, rep in pairs]
    done = [d.exists() for d in dests]
    if any(done[i] and not all(done[:i]) for i in range(len(done))):
        raise ValueError('Non-contiguous committed trials; the memory stream must be sequential')

    # The memory directory is created by the agent; restore before it loads the files.
    memory_dir = Path(cfg.output_path) / 'alfworld' / args.room / cfg.arm / cfg.llm_model.replace('/', '_')
    stream_dir.mkdir(parents=True, exist_ok=True)
    restore_snapshot(stream_dir, memory_dir)
    agent = PaperNormalizationAgent(cfg)
    assert Path(agent.cross_mem_dir) == memory_dir, (agent.cross_mem_dir, memory_dir)
    if agent.cross_mem.current_episode_number != sum(done):
        raise ValueError(f'Memory holds {agent.cross_mem.current_episode_number} episodes '
                         f'but {sum(done)} trials are committed')

    for position, ((record, repeat), dest) in enumerate(zip(pairs, dests)):
        if dest.exists():
            continue
        seed = plan['seed'] + record['index'] * 101 + repeat
        random.seed(seed)
        np.random.seed(seed)
        started = time.monotonic()
        begin_transaction(stream_dir, memory_dir, dest)
        env = factory.make(record, step_limit=plan['max_steps'])
        env.reveal_outcome_to_tracker = False
        try:
            observation, info = env.reset()
            goal = info['goal_text']
            agent.start_episode()
            memory_before = agent.cross_mem.get_stats()
            steps = []
            for i in range(plan['max_steps']):
                call_errors.clear()
                node = StateNode(state=observation, goal=info['goal'], goal_text=goal, phase=info['phase'])
                action, raw = agent.generate_action(node, info={k: v for k, v in info.items() if k not in ('won', 'score')})
                if call_errors:
                    raise RuntimeError('Actor request failed: ' + call_errors[-1])
                entry = dict(index=i, observation=observation, action=action,
                             inventory=info['inv'], phase=info['phase'], raw_actor=raw)
                observation, reward, done_env, info = env.step(action)
                entry.update(environment_reward=float(reward), terminated=bool(done_env))  # logged only
                steps.append(entry)
                if done_env:
                    break
            call_errors.clear()
            agent.end_episode(state=observation)  # judge scores the trajectory; memory is updated
            if call_errors:
                raise RuntimeError('Judge request failed: ' + call_errors[-1])
            result = dict(status='complete', arm=cfg.arm, target=args.room, split=SPLIT,
                          task_index=record['index'], game_id=record['game_id'], task_type=record['task_type'],
                          repeat=repeat, seed=seed, episode_index=position, goal=goal, steps=steps,
                          success=bool(info['won']), outcome=float(bool(info['won'])),
                          reward_protocol=agent.cross_mem.reward_protocol, judge_reward_scale=agent.cross_mem.reward_scale,
                          judge_model=args.judge_model, actor_endpoint=cfg.llm_base_url,
                          memory_before=memory_before, memory_after=agent.cross_mem.get_stats(),
                          counters=agent.counters, elapsed_seconds=round(time.monotonic() - started, 3),
                          completed_utc=datetime.now(timezone.utc).isoformat())
            atomic(dest, result)
            (stream_dir / 'transaction.json').unlink()
            print(f"COMMITTED {cfg.arm}_{args.room} {position + 1}/{len(pairs)} task={record['index']} "
                  f"repeat={repeat} success={info['won']} steps={len(steps)}", flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()

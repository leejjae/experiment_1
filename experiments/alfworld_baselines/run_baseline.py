"""One Reflexion / EvoTest / AWM stream on ALFWorld.

Protocol as the self-distillation runs (experiments/alfworld_ace_sd/run.py):
valid_seen, original order, 5 consecutive trials per task, seed = 42 +
index*101 + trial, Qwen3-32B-AWQ, actor T=0.7, 30 steps, no outcome to any role.
The actor is the memory-free JitRL agent (the "memory off" arm); the method
only sets its guiding prompt. Per-task methods (reflexion, evotest) reset at
each task and can be sharded by task; AWM keeps one workflow set per room, so it
runs as a single stream per room.

  python experiments/alfworld_baselines/run_baseline.py --method reflexion --room kitchen --port 8000
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments/alfworld_ace_sd'))
sys.path.insert(0, str(HERE))

from run import PLAN, SPLIT, atomic, now, render_trajectory  # alfworld_ace_sd/run.py (also sets sys.path)

import numpy as np
import yaml
from main import build_parser
from src.env import AlfWorldEnvFactory
from src.evaluation import StateNode
from src.jitrl_agent import JitRLAgent
from experiments.alfworld_no_feedback.agent import without_outcome
from methods import METHODS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=sorted(METHODS))
    p.add_argument('--room', required=True, choices=('kitchen', 'living_room', 'bathroom', 'bedroom'))
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--out', type=Path, default=HERE / 'runs/20261029')
    p.add_argument('--limit_tasks', type=int, default=None, help='pilot: first N tasks')
    p.add_argument('--trials', type=int, default=None, help='pilot: override 5 trials')
    args = p.parse_args()

    plan = json.loads(PLAN.read_text())
    trials = args.trials or plan['eval_repeats']
    cfg = build_parser().parse_args(['--room_type', args.room])
    cfg.llm_temperature = plan['eval_temperature']
    cfg.llm_base_url = f'http://localhost:{args.port}/v1'
    cfg.enable_cross_mem = False
    method_cls = METHODS[args.method]
    assert method_cls.scope == 'task' or args.shards == 1, 'room-scoped memory cannot be sharded'
    results = args.out / args.method / args.room / 'results'
    os.environ.setdefault('ALFWORLD_DATA', str(Path.home() / '.cache/alfworld'))  # alfworld-download default
    factory = AlfWorldEnvFactory(yaml.safe_load((ROOT / 'ALFWorld/configs/base_config.yaml').read_text()),
                                 split='eval_in_distribution')

    rows = [r for r in plan['splits'][SPLIT] if r['room_type'] == args.room][:args.limit_tasks]
    rows = rows[args.shard::args.shards]
    pairs = [(r, k) for r in rows for k in range(1, trials + 1)]
    paths = [results / f"task_{r['index']:04d}_run_{k:02d}.json" for r, k in pairs]
    done = [q.exists() for q in paths]
    assert done == sorted(done, reverse=True), 'Non-contiguous completed trials'

    method = method_cls(cfg)
    if any(done):  # resume from the last committed trial's method state
        last = json.loads(paths[sum(done) - 1].read_text())
        if method.scope == 'room' or last['repeat'] < trials:
            method.load(last['method_state_after'])
    agent = JitRLAgent(cfg)
    for (record, trial), dest in zip(pairs, paths):
        if dest.exists():
            continue
        if method.scope == 'task' and trial == 1:
            method = method_cls(cfg)
        seed = plan['seed'] + record['index'] * 101 + trial
        random.seed(seed)
        np.random.seed(seed)
        started = time.monotonic()
        env = factory.make(record, step_limit=plan['max_steps'])
        env.reveal_outcome_to_tracker = False
        try:
            observation, info = env.reset()
            initial_observation, goal = observation, info['goal_text']
            guide, guide_log = method.guide(goal)
            agent.guiding_prompt = guide
            agent.start_episode()
            steps = []
            for i in range(plan['max_steps']):
                node = StateNode(state=observation, goal=info['goal'], goal_text=goal, phase=info['phase'])
                action, raw = agent.generate_action(node, info=without_outcome(info))
                prev = observation
                observation, _reward, done_env, info = env.step(action)
                steps.append(dict(index=i, observation=prev, action=action, next_observation=observation,
                                  inventory=info['inv'], phase=info['phase'], raw_actor=raw))
                if done_env:
                    break
            success = bool(info['won'])  # reporting only; never shown to any role
        finally:
            env.close()
        trajectory = render_trajectory(initial_observation, steps)
        method_log = method.after_trial(goal, initial_observation, steps, trajectory)
        atomic(dest, dict(status='complete', method=args.method, target=args.room, split=SPLIT,
                          task_index=record['index'], game_id=record['game_id'], task_type=record['task_type'],
                          repeat=trial, seed=seed, goal=goal, steps=steps, success=success, outcome=float(success),
                          counters=agent.counters, trajectory_text=trajectory, guide_used=guide, guide_log=guide_log,
                          method_log=method_log, method_state_after=method.state(),
                          actor_endpoint=cfg.llm_base_url, elapsed_seconds=round(time.monotonic() - started, 3),
                          completed_utc=now()))
        print(f"COMMITTED {args.method} {args.room} shard={args.shard} task={record['index']} trial={trial} "
              f"success={success} judged={method_log['judge']['judged_success']} steps={len(steps)} "
              f"guide_chars={len(guide)}", flush=True)


if __name__ == '__main__':
    main()

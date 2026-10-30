"""One iterative self-distillation stream: a (room, shard) pair, beta fixed.

Protocol as run.py (valid_seen, original order, 5 consecutive trials per task,
seed = 42 + index*101 + trial, T=0.7, 30 steps, no outcome to any role). After
each trial only the Reflector runs (same prompt, empty playbook); its output is
f_k. Trial k of a task is scored with that task's f_1..f_{k-1}; every task starts
again from pi_1, so tasks are independent and can be sharded.

  python experiments/alfworld_ace_sd/run_iter.py --room kitchen --port 8000 --shard 0 --shards 4

With --feedback_from RUN_ROOT (e.g. runs/20261029_iter/beta_1) the Reflector is not
called: f_k of a task is read from that run's trial-k `reflection`, so every
--rule sees exactly the same feedback text.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

from run import PLAN, REFLECTOR, ROOT, SPLIT, HERE, atomic, now, render_trajectory, role_call

import numpy as np
import yaml
from main import build_parser
from src.env import AlfWorldEnvFactory
from src.evaluation import StateNode
from experiments.alfworld_no_feedback.agent import without_outcome
from agent_iter import RULES, IterativeSelfDistillAgent

EMPTY_PLAYBOOK = '(empty)'


def reflect(cfg, goal, trajectory):
    return role_call(cfg, REFLECTOR.replace('{{task}}', goal).replace('{{trajectory}}', trajectory)
                     .replace('{{playbook}}', EMPTY_PLAYBOOK))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--room', required=True, choices=('kitchen', 'living_room', 'bathroom', 'bedroom'))
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--out', type=Path, default=HERE / 'runs/20261029_iter')
    p.add_argument('--rule', default='sum', choices=RULES)
    p.add_argument('--lam', type=float, default=0.5, help='EMA weight on the newest r_k')
    p.add_argument('--feedback_from', type=Path, default=None, help='reuse f_k from this run root')
    p.add_argument('--limit_tasks', type=int, default=None, help='pilot: first N tasks')
    p.add_argument('--trials', type=int, default=None, help='pilot: override 5 trials')
    args = p.parse_args()

    plan = json.loads(PLAN.read_text())
    trials = args.trials or plan['eval_repeats']
    cfg = build_parser().parse_args(['--room_type', args.room])
    cfg.llm_temperature = plan['eval_temperature']
    cfg.llm_base_url = f'http://localhost:{args.port}/v1'
    results = args.out / f'beta_{args.beta:g}' / args.room / 'results'
    os.environ.setdefault('ALFWORLD_DATA', str(Path.home() / '.cache/alfworld'))  # alfworld-download default
    factory = AlfWorldEnvFactory(yaml.safe_load((ROOT / 'ALFWorld/configs/base_config.yaml').read_text()),
                                 split='eval_in_distribution')

    rows = [r for r in plan['splits'][SPLIT] if r['room_type'] == args.room][:args.limit_tasks]
    rows = rows[args.shard::args.shards]
    agent = IterativeSelfDistillAgent(cfg, beta=args.beta, rule=args.rule, lam=args.lam)
    for record in rows:
        paths = [results / f"task_{record['index']:04d}_run_{k:02d}.json" for k in range(1, trials + 1)]
        done = [q.exists() for q in paths]
        assert done == sorted(done, reverse=True), f"Non-contiguous trials for task {record['index']}"
        if args.feedback_from:
            src = [args.feedback_from / args.room / 'results' / q.name for q in paths]
            fixed = [json.loads(q.read_text())['reflection'] for q in src]
            feedbacks = fixed[:sum(done)]
        else:
            feedbacks = [json.loads(q.read_text())['reflection'] for q in paths if q.exists()]  # resume
        for trial, dest in enumerate(paths, 1):
            if dest.exists():
                continue
            seed = plan['seed'] + record['index'] * 101 + trial
            random.seed(seed)
            np.random.seed(seed)
            started = time.monotonic()
            env = factory.make(record, step_limit=plan['max_steps'])
            env.reveal_outcome_to_tracker = False
            try:
                observation, info = env.reset()
                initial_observation, goal = observation, info['goal_text']
                agent.start_trial(feedbacks)
                steps = []
                for i in range(plan['max_steps']):
                    node = StateNode(state=observation, goal=info['goal'], goal_text=goal, phase=info['phase'])
                    action, _ = agent.generate_action(node, info=without_outcome(info))
                    prev = observation
                    observation, _reward, done_env, info = env.step(action)
                    steps.append(dict(index=i, observation=prev, action=action, next_observation=observation,
                                      inventory=info['inv'], phase=info['phase'], **agent.trace[-1]))
                    if done_env:
                        break
                success = bool(info['won'])  # reporting only; never shown to any role
            finally:
                env.close()
            trajectory = render_trajectory(initial_observation, steps)
            reflection = fixed[trial - 1] if args.feedback_from else reflect(cfg, goal, trajectory)
            atomic(dest, dict(status='complete', method='iterative_self_distillation', beta=args.beta,
                              rule=args.rule, lam=args.lam,
                              feedback_from=str(args.feedback_from) if args.feedback_from else None,
                              target=args.room, split=SPLIT, task_index=record['index'], game_id=record['game_id'],
                              task_type=record['task_type'], repeat=trial, seed=seed, goal=goal, steps=steps,
                              success=success, outcome=float(success), counters=agent.counters,
                              trajectory_text=trajectory, feedbacks_used=list(feedbacks), reflection=reflection,
                              actor_endpoint=cfg.llm_base_url, elapsed_seconds=round(time.monotonic() - started, 3),
                              completed_utc=now()))
            feedbacks.append(reflection)
            print(f"COMMITTED {args.room} shard={args.shard} task={record['index']} trial={trial} "
                  f"rule={args.rule} success={success} steps={len(steps)} n_feedback={trial - 1} "
                  f"teacher_changed={agent.counters['n_teacher_changed']}", flush=True)


if __name__ == '__main__':
    main()

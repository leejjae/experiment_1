"""One iterative-ACE stream: a (rule, room) pair, beta fixed.

Same as run.py (ACE-SD): valid_seen, original order, 5 consecutive trials per task,
seed = 42 + index*101 + trial, T=0.7, 30 steps, no outcome to any role; the
Reflector and Curator update one playbook per room after every trial, on-policy.
The only change is the scoring (agent_ace_iter.py): trial k of a task combines the
teachers of the playbook snapshots taken before trials 1..k of that task with
--rule (mean or ema). The playbook depends on all earlier trials, so a room is one
sequential stream. One JSON per trial; reruns resume after the last trial.

  python experiments/alfworld_ace_sd/run_ace_iter.py --rule mean --room kitchen --port 8000
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

from run import INITIAL_PLAYBOOK, PLAN, ROOT, SPLIT, HERE, atomic, now, reflect_and_curate, render_trajectory

import numpy as np
import yaml
from main import build_parser
from src.env import AlfWorldEnvFactory
from src.evaluation import StateNode
from experiments.alfworld_no_feedback.agent import without_outcome
from agent_ace_iter import IterativeAceAgent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rule', required=True, choices=('mean', 'ema', 'sum', 'replace'))
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--lam', type=float, default=0.5, help='EMA weight on the newest r_k')
    p.add_argument('--room', required=True, choices=('kitchen', 'living_room', 'bathroom', 'bedroom'))
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--out', type=Path, default=HERE / 'runs/new_ace_iter')
    p.add_argument('--limit_tasks', type=int, default=None, help='pilot: first N tasks')
    p.add_argument('--trials', type=int, default=None, help='pilot: override 5 trials')
    args = p.parse_args()

    plan = json.loads(PLAN.read_text())
    trials = args.trials or plan['eval_repeats']
    cfg = build_parser().parse_args(['--room_type', args.room])
    cfg.llm_temperature = plan['eval_temperature']
    cfg.llm_base_url = f'http://localhost:{args.port}/v1'
    stream = args.out / args.rule / f'beta_{args.beta:g}' / args.room
    os.environ.setdefault('ALFWORLD_DATA', str(Path.home() / '.cache/alfworld'))  # alfworld-download default
    factory = AlfWorldEnvFactory(yaml.safe_load((ROOT / 'ALFWorld/configs/base_config.yaml').read_text()),
                                 split='eval_in_distribution')

    rows = [r for r in plan['splits'][SPLIT] if r['room_type'] == args.room][:args.limit_tasks]
    pairs = [(r, k) for r in rows for k in range(1, trials + 1)]
    paths = [stream / 'results' / f"task_{r['index']:04d}_run_{k:02d}.json" for r, k in pairs]
    done = [q.exists() for q in paths]
    assert done == sorted(done, reverse=True), 'Non-contiguous completed trials'
    playbook_text = json.loads(paths[sum(done) - 1].read_text())['playbook_after'] if any(done) else INITIAL_PLAYBOOK

    agent = IterativeAceAgent(cfg, beta=args.beta, rule=args.rule, lam=args.lam)
    snapshots = []
    for n, ((record, trial), dest) in enumerate(zip(pairs, paths)):
        if trial == 1:
            snapshots = []
        if dest.exists():
            snapshots.append(json.loads(dest.read_text())['playbook_before'])  # resume
            continue
        snapshots.append(playbook_text)
        seed = plan['seed'] + record['index'] * 101 + trial
        random.seed(seed)
        np.random.seed(seed)
        started = time.monotonic()
        env = factory.make(record, step_limit=plan['max_steps'])
        env.reveal_outcome_to_tracker = False
        try:
            observation, info = env.reset()
            initial_observation, goal = observation, info['goal_text']
            agent.start_trial(snapshots)
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
        roles, new_playbook = reflect_and_curate(cfg, goal, trajectory, playbook_text)
        atomic(dest, dict(status='complete', method='iterative_ace_self_distillation', rule=args.rule,
                          lam=args.lam, beta=args.beta, target=args.room, split=SPLIT,
                          task_index=record['index'], game_id=record['game_id'], task_type=record['task_type'],
                          repeat=trial, seed=seed, episode_index=n, goal=goal, steps=steps, success=success,
                          outcome=float(success), counters=agent.counters, trajectory_text=trajectory,
                          n_teachers=len(snapshots), playbook_before=playbook_text, playbook_after=new_playbook,
                          **roles, actor_endpoint=cfg.llm_base_url,
                          elapsed_seconds=round(time.monotonic() - started, 3), completed_utc=now()))
        playbook_text = new_playbook
        print(f"COMMITTED rule={args.rule} beta={args.beta:g} {args.room} {n + 1}/{len(pairs)} task={record['index']} "
              f"trial={trial} teachers={len(snapshots)} success={success} steps={len(steps)} "
              f"ops={len(roles['operations'])} teacher_changed={agent.counters['n_teacher_changed']}", flush=True)


if __name__ == '__main__':
    main()

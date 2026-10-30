"""Run the ReAct baselines and the memory-free JitRL arm on ALFWorld.

Tasks, order, repeats, step budget, temperature and seeds come from
data/alfworld_valid_seen_74.json (valid_seen 74: 20/14/20/20,
seed = seed + index*101 + trial).
One JSON per (task, repeat); existing files are skipped, so reruns resume.

  python experiments/alfworld_react/run.py --rooms kitchen bedroom --port 8000
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ALF = ROOT / 'ALFWorld'
sys.path.insert(0, str(ALF))
sys.path.insert(0, str(HERE))

import numpy as np
import yaml
from main import build_parser
from src.env import AlfWorldEnvFactory
from src.evaluation import StateNode
from react_agent import ReActAgent
from src.jitrl_agent import JitRLAgent

PLAN = ROOT / 'data/alfworld_valid_seen_74.json'
SPLIT = 'valid_seen'
ROOMS = ('kitchen', 'living_room', 'bathroom', 'bedroom')


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rooms', nargs='+', default=list(ROOMS), choices=ROOMS)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--out', type=Path, default=HERE / 'runs/20261020')
    p.add_argument('--limit_tasks', type=int, default=None, help='smoke test: first N tasks per room')
    p.add_argument('--repeats', type=int, default=None, help='override plan eval_repeats (smoke test)')
    p.add_argument('--arm', default='react', choices=['react', 'react0', 'jitrl_nomem'],
                   help='react0: zero-shot ReAct (no demos); jitrl_nomem: JitRLAgent prompt/summary/top-3 JSON with cross-episode memory off (ablation)')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1, help='episodes are independent (no memory), so shards can run in parallel')
    args = p.parse_args()

    plan = json.loads(PLAN.read_text())
    repeats = args.repeats or plan['eval_repeats']
    cfg = build_parser().parse_args(['--room_type', 'kitchen'])  # only llm_model/action_abstraction defaults are used
    cfg.llm_temperature = plan['eval_temperature']
    cfg.llm_base_url = f'http://localhost:{args.port}/v1'
    cfg.enable_cross_mem = False
    os.environ.setdefault('ALFWORLD_DATA', str(Path.home() / '.cache/alfworld'))  # alfworld-download default
    factory = AlfWorldEnvFactory(yaml.safe_load((ALF / 'configs/base_config.yaml').read_text()),
                                 split='eval_in_distribution')

    for room in args.rooms:
        rows = [r for r in plan['splits'][SPLIT] if r['room_type'] == room][:args.limit_tasks]
        pairs = [(r, rep) for r in rows for rep in range(1, repeats + 1)]
        for position, (record, repeat) in enumerate(pairs):
            dest = args.out / 'results' / f'{args.arm}_{room}' / f"{record['index']:04d}_{repeat}.json"
            if dest.exists() or position % args.shards != args.shard:
                continue
            seed = plan['seed'] + record['index'] * 101 + repeat
            random.seed(seed)
            np.random.seed(seed)
            started = time.monotonic()
            env = factory.make(record, step_limit=plan['max_steps'])
            try:
                observation, info = env.reset()
                goal = info['goal_text']
                if args.arm in ('react', 'react0'):
                    agent = ReActAgent(cfg, record['task_type'], shots=2 if args.arm == 'react' else 0)
                    agent.start_episode(observation)
                else:
                    # Mirrors the no-verifiable-feedback JitRL run: tracker never sees `won`.
                    env.reveal_outcome_to_tracker = False
                    agent = JitRLAgent(cfg)
                    agent.start_episode()
                steps = []
                for i in range(plan['max_steps']):
                    node = StateNode(state=observation, goal=info['goal'], goal_text=goal, phase=info['phase'])
                    action, raws = agent.generate_action(node, info=info)
                    entry = dict(index=i, observation=observation, action=action,
                                 proposed=agent.game_history[-1].get('proposed'),
                                 thoughts=agent.game_history[-1].get('thoughts'),
                                 inventory=info['inv'], phase=info['phase'], raw_actor=raws)
                    observation, reward, done, info = env.step(action)
                    agent.update_game_history_reward(reward, info['score'])
                    if args.arm in ('react', 'react0'):
                        agent.observe(observation)
                    entry.update(environment_reward=float(reward), terminated=bool(done))
                    steps.append(entry)
                    if done:
                        break
                result = dict(status='complete', arm={'react': 'react_static_fewshot', 'react0': 'react_zero_shot', 'jitrl_nomem': 'jitrl_nomem'}[args.arm], target=room, split=SPLIT,
                              task_index=record['index'], game_id=record['game_id'], task_type=record['task_type'],
                              repeat=repeat, seed=seed, episode_index=position, goal=goal, steps=steps,
                              success=bool(info['won']), outcome=float(bool(info['won'])),
                              demos=getattr(agent, 'demo_keys', None),
                              final_transcript=getattr(agent, 'transcript', None), actor_endpoint=cfg.llm_base_url,
                              counters=agent.counters, elapsed_seconds=round(time.monotonic() - started, 3),
                              completed_utc=datetime.now(timezone.utc).isoformat())
                atomic(dest, result)
                print(f"COMMITTED {args.arm}_{room} {position + 1}/{len(pairs)} task={record['index']} "
                      f"repeat={repeat} success={info['won']} steps={len(steps)}", flush=True)
            finally:
                env.close()


if __name__ == '__main__':
    main()

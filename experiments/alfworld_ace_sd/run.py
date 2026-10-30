"""One ACE + self-distillation stream: a single (beta, room) pair.

Protocol inherited from the JitRL table (data/alfworld_valid_seen_74.json):
valid_seen 74 tasks (this room's subset), original order, 5 consecutive trials per
task, seed = 42 + index*101 + trial, Qwen3-32B-AWQ, proposal T=0.7, 30 steps.
No reward, success flag or judge reaches the agent, Reflector or Curator.
The playbook starts from the three empty ACE sections and persists across trials
and tasks of this room. One JSON per trial; reruns resume after the last trial.

  python experiments/alfworld_ace_sd/run.py --beta 0.5 --room kitchen --port 8000
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
sys.path.insert(0, str(ROOT / 'ALFWorld'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from main import build_parser
from src.env import AlfWorldEnvFactory
from src.evaluation import StateNode
from src.openai_helpers import chat_completion_with_retries, response_text
from experiments.alfworld_no_feedback.agent import without_outcome
from agent import AceSelfDistillAgent
import playbook as ace

PLAN = ROOT / 'data/alfworld_valid_seen_74.json'
SPLIT = 'valid_seen'
INITIAL_PLAYBOOK = '## STRATEGIES AND HARD RULES\n## VERIFICATION CHECKLIST\n## OTHERS'
REFLECTOR = (HERE / 'prompts/reflector.txt').read_text()
CURATOR = (HERE / 'prompts/curator.txt').read_text()
ROLE_MAX_TOKENS = 4096


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def render_trajectory(initial_observation, steps):
    lines = [f'[0] OBSERVATION: {initial_observation}']
    for s in steps:
        lines.append(f"[{s['index'] + 1}] ACTION: {s['action']}")
        lines.append(f"[{s['index'] + 1}] OBSERVATION: {s['next_observation']}")
    return '\n'.join(lines)


def role_call(cfg, prompt):
    # ACE config: temperature 0 for Reflector and Curator; single user message.
    res = chat_completion_with_retries(model=cfg.llm_model, sys_prompt='', prompt=prompt, max_tokens=ROLE_MAX_TOKENS,
                                       temperature=0.0, base_url=cfg.llm_base_url, enable_thinking=False)
    return response_text(res) or ''


def reflect_and_curate(cfg, goal, trajectory, playbook_text):
    reflector_prompt = (REFLECTOR.replace('{{task}}', goal).replace('{{trajectory}}', trajectory)
                        .replace('{{playbook}}', playbook_text))
    reflection = role_call(cfg, reflector_prompt)
    curator_prompt = (CURATOR.replace('{{question_context}}', goal).replace('{{current_playbook}}', playbook_text)
                      .replace('{{final_generated_code}}', trajectory).replace('{{guidebook}}', reflection))
    curator_raw = role_call(cfg, curator_prompt)
    parsed = ace.extract_json_from_text(curator_raw, 'operations')
    operations, error = [], None
    if isinstance(parsed, dict) and isinstance(parsed.get('operations'), list):
        operations = [op for op in parsed['operations'] if isinstance(op, dict) and op.get('type') == 'ADD']
    else:
        error = 'Curator JSON invalid; skipped (ACE behaviour)'
    new_playbook, _ = ace.apply_curator_operations(playbook_text, operations, ace.get_next_global_id(playbook_text))
    return dict(reflection=reflection, curator_raw=curator_raw, operations=operations, curator_error=error), new_playbook


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--beta', type=float, required=True)
    p.add_argument('--room', required=True, choices=('kitchen', 'living_room', 'bathroom', 'bedroom'))
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--out', type=Path, default=HERE / 'runs/20261022')
    p.add_argument('--limit_tasks', type=int, default=None, help='pilot: first N tasks')
    p.add_argument('--trials', type=int, default=None, help='pilot: override 5 trials')
    args = p.parse_args()

    plan = json.loads(PLAN.read_text())
    trials = args.trials or plan['eval_repeats']
    cfg = build_parser().parse_args(['--room_type', args.room])
    cfg.llm_temperature = plan['eval_temperature']
    cfg.llm_base_url = f'http://localhost:{args.port}/v1'
    stream = args.out / f'beta_{args.beta:g}' / args.room
    os.environ.setdefault('ALFWORLD_DATA', str(Path.home() / '.cache/alfworld'))  # alfworld-download default
    factory = AlfWorldEnvFactory(yaml.safe_load((ROOT / 'ALFWorld/configs/base_config.yaml').read_text()),
                                 split='eval_in_distribution')

    rows = [r for r in plan['splits'][SPLIT] if r['room_type'] == args.room][:args.limit_tasks]
    pairs = [(r, k) for r in rows for k in range(1, trials + 1)]
    paths = [stream / 'results' / f"task_{r['index']:04d}_run_{k:02d}.json" for r, k in pairs]
    done = [q.exists() for q in paths]
    assert done == sorted(done, reverse=True), 'Non-contiguous completed trials'
    playbook_text = json.loads(paths[sum(done) - 1].read_text())['playbook_after'] if any(done) else INITIAL_PLAYBOOK

    agent = AceSelfDistillAgent(cfg, beta=args.beta)
    for n, ((record, trial), dest) in enumerate(zip(pairs, paths)):
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
            agent.start_trial(playbook_text)
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
        atomic(dest, dict(status='complete', method='ace_self_distillation', beta=args.beta, target=args.room,
                          split=SPLIT, task_index=record['index'], game_id=record['game_id'],
                          task_type=record['task_type'], repeat=trial, seed=seed, episode_index=n, goal=goal,
                          steps=steps, success=success, outcome=float(success), counters=agent.counters,
                          trajectory_text=trajectory, playbook_before=playbook_text, playbook_after=new_playbook,
                          **roles, actor_endpoint=cfg.llm_base_url,
                          elapsed_seconds=round(time.monotonic() - started, 3), completed_utc=now()))
        playbook_text = new_playbook
        print(f"COMMITTED beta={args.beta:g} {args.room} {n + 1}/{len(pairs)} task={record['index']} trial={trial} "
              f"success={success} steps={len(steps)} ops={len(roles['operations'])} "
              f"teacher_changed={agent.counters['n_teacher_changed']}", flush=True)


if __name__ == '__main__':
    main()

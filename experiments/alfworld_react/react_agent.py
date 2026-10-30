"""ReAct baseline for ALFWorld: static few-shot, no cross-episode memory.

Follows Yao et al. (2023) for ALFWorld: two hand-written ReAct demos of the same
task type (react_{type}_1, react_{type}_0 from the official
alfworld_3prompts.json), followed by the current episode as a `> action` /
observation transcript. `think:` turns are answered with "OK." and do not use
an environment step, so the action budget matches JitRL (max_steps actions).

For fairness with JitRL and NaiveAgent the prompt also carries the same extra
information those agents get: [PHASE]/[SEARCHED] (`prompt_block`), the
inventory, and the admissible-command list. Actions pass through the same
`repair_action`.
"""

import json
import random
from pathlib import Path

from src.action_utils import repair_action
from src.openai_helpers import chat_completion_with_retries, response_text

PROMPTS = json.loads((Path(__file__).resolve().parent / 'prompts/alfworld_3prompts.json').read_text())
TASK_PREFIX = {
    'pick_and_place_simple': 'put',
    'pick_clean_then_place_in_recep': 'clean',
    'pick_heat_then_place_in_recep': 'heat',
    'pick_cool_then_place_in_recep': 'cool',
    'look_at_obj_in_light': 'examine',
    'pick_two_obj_and_place': 'puttwo',
}
SYS_PROMPT = 'Interact with a household to solve a task.'
MAX_CONSECUTIVE_THINKS = 2


def demo_keys(task_type):
    key = TASK_PREFIX[task_type]
    return [f'react_{key}_1', f'react_{key}_0']


def strip_header(ob):
    # Same as the official ReAct runner: drop the "-= Welcome to TextWorld =-" banner.
    parts = ob.split('\n\n')
    return '\n'.join(parts[1:]) if len(parts) > 1 and 'Welcome to TextWorld' in parts[0] else ob


class ReActAgent:
    def __init__(self, args, task_type, shots=2):
        # shots=0 is the zero-shot ablation: same format, transcript and info, no demos.
        assert shots in (0, 2)
        self.args = args
        self.shots = shots
        self.demo_keys = demo_keys(task_type) if shots else []
        self.demos = ''.join(PROMPTS[k] for k in self.demo_keys)
        self.game_history = []
        self.transcript = ''
        self.counters = {}

    def start_episode(self, initial_observation):
        self.game_history = []
        self.transcript = strip_header(initial_observation).strip() + '\n'
        self.counters = {'n_parse_failures': 0, 'n_action_repairs': 0, 'n_thinks': 0, 'n_forced_actions': 0}

    def update_game_history_reward(self, reward, score):
        if self.game_history:
            self.game_history[-1]['reward'] = reward
            self.game_history[-1]['score'] = score

    def observe(self, observation):
        self.transcript += observation.strip() + '\n'

    def generate_action(self, state_node, info=None):
        """Return (env_action, raw_responses). Think turns are resolved internally."""
        info = info or {}
        valid = list(info.get('valid', []))
        raws, thoughts = [], []
        action_text = None
        for attempt in range(MAX_CONSECUTIVE_THINKS + 1):
            force = attempt == MAX_CONSECUTIVE_THINKS
            raw = self._call(info, valid, force)
            raws.append(raw)
            kind, text = self._parse(raw)
            if kind == 'think' and not force:
                thoughts.append(text)
                self.counters['n_thinks'] += 1
                self.transcript += f'> think: {text}\nOK.\n'
                continue
            if kind == 'think':
                self.counters['n_forced_actions'] += 1
            action_text = text if kind == 'action' else None
            break
        if action_text is None:
            self.counters['n_parse_failures'] += 1
        action, repaired = repair_action(action_text or '', valid, info.get('goal'), mode=self.args.action_abstraction)
        if repaired:
            self.counters['n_action_repairs'] += 1
        self.transcript += f'> {action}\n'
        self.game_history.append(dict(state=state_node.state, action=action, proposed=action_text,
                                      thoughts=thoughts, reward=None, score=None))
        return action.strip(), raws

    def get_prompt(self, info, valid, force_action=False):
        shuffled = list(valid)
        random.shuffle(shuffled)  # global RNG is seeded per episode by run.py
        status = f"""{info.get('prompt_block', '')}
INVENTORY: {info.get('inv', 'unknown')}
ADMISSIBLE ACTIONS (unordered): {shuffled}"""
        fmt = 'in the same format as the examples' if self.shots else 'in the same format as the transcript'
        rule = ('Respond with exactly one line: an action copied exactly from ADMISSIBLE ACTIONS.'
                if force_action else
                f'Respond with exactly one line, {fmt}: either '
                '"think: <your reasoning>" or an action copied exactly from ADMISSIBLE ACTIONS.')
        examples = f"Here are two examples.\n{self.demos}\n\n" if self.shots else ''
        return f"""{examples}Here is the task.
{self.transcript}
[CURRENT STATUS]
{status}

{rule} Do not write the leading ">"."""

    def _call(self, info, valid, force_action):
        res = chat_completion_with_retries(
            model=self.args.llm_model, sys_prompt=SYS_PROMPT,
            prompt=self.get_prompt(info, valid, force_action),
            max_tokens=200, temperature=self.args.llm_temperature,
            base_url=self.args.llm_base_url, enable_thinking=False)
        return response_text(res) or ''

    @staticmethod
    def _parse(raw):
        for line in raw.strip().splitlines():
            line = line.strip().lstrip('>').strip().strip('`').strip()
            if not line:
                continue
            if line.lower().startswith('think:'):
                return 'think', line.split(':', 1)[1].strip()
            if line.lower().startswith('action:'):
                line = line.split(':', 1)[1].strip()
            return 'action', line
        return None, None

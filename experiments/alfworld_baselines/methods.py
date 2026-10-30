"""Reflexion, EvoTest and AWM for ALFWorld, without environment reward.

All three steer the same memory-free JitRL actor through its `guiding_prompt`
(appended to the system prompt as "Follow this guide: ..."). Where the original
method needs success or a score, the same LLM judges the trajectory (`judge`);
the true outcome never reaches any role.

  Reflexion (Shinn et al. 2023, alfworld_runs/generate_reflections.py): after a
    trial judged failed, reflect with the official ALFWorld query and few-shot
    examples; the last 3 reflections of this task are the actor's memory.
  EvoTest (He et al. 2026, src/our_agent.py): per task, a tree of guiding
    prompts. Before each trial the parent is chosen by
    UCB = score + c * d**depth * sqrt(log N / (1 + children)) (c=1.0, d=0.8,
    N=max(2, #nodes)), and the Evolver rewrites its prompt from its transcript.
    freeze_on_win: once a node is judged successful, the best node is rerun.
    Omitted: extract_state code and score-delta memory (no ALFWorld score).
  AWM (Wang et al. 2024, online): after a trial judged successful, induce
    workflows from it into the room's workflow set, which persists across tasks.

Every method keeps its full state as a JSON-able dict so a run can resume from
the last committed trial.
"""
import math
from pathlib import Path

from src.openai_helpers import chat_completion_with_retries, response_text

HERE = Path(__file__).resolve().parent
PROMPTS = {p.stem: p.read_text() for p in (HERE / 'prompts').glob('*.txt')}
ROLE_MAX_TOKENS = 4096


def call(cfg, prompt, temperature=0.0):
    res = chat_completion_with_retries(model=cfg.llm_model, sys_prompt='', prompt=prompt, max_tokens=ROLE_MAX_TOKENS,
                                       temperature=temperature, base_url=cfg.llm_base_url, enable_thinking=False)
    return (response_text(res) or '').strip()


def judge(cfg, goal, trajectory):
    """p(success) from the YES/NO label logprobs of one T=0 token."""
    prompt = PROMPTS['judge'].replace('{{task}}', goal).replace('{{trajectory}}', trajectory)
    res = chat_completion_with_retries(model=cfg.llm_model, sys_prompt='', prompt=prompt, max_tokens=1, temperature=0.0,
                                       top_logprobs=20, base_url=cfg.llm_base_url, enable_thinking=False)
    top = res.choices[0].logprobs.content[0].top_logprobs
    found = {}
    for cand in top:
        found.setdefault(cand.token.strip().upper(), cand.logprob)
    floor = min(c.logprob for c in top) - 1.0
    yes, no = found.get('YES', floor), found.get('NO', floor)
    p = 1.0 / (1.0 + math.exp(no - yes))
    return dict(p_success=p, judged_success=p > 0.5, label_missing=[k for k in ('YES', 'NO') if k not in found])


def transcript(initial_observation, steps):
    """ReAct/Reflexion-style log: observation, then `> action` / observation lines."""
    lines = [initial_observation.strip()]
    for s in steps:
        lines += [f"> {s['action']}", s['next_observation'].strip()]
    return '\n'.join(lines)


class Reflexion:
    scope = 'task'

    def __init__(self, cfg):
        self.cfg = cfg
        self.memory = []

    def state(self):
        return dict(memory=self.memory)

    def load(self, state):
        self.memory = list(state['memory'])

    def guide(self, goal):
        if not self.memory:
            return '', {}
        body = '\n'.join(f'Trial {i}:\n{m}' for i, m in enumerate(self.memory[-3:]))
        return f'Your memory for the task below (plans from your past attempts):\n{body}', {}

    def after_trial(self, goal, initial_observation, steps, trajectory):
        verdict = judge(self.cfg, goal, trajectory)
        log = dict(judge=verdict, reflection=None)
        if not verdict['judged_success']:
            scenario = transcript(initial_observation, steps) + '\nSTATUS: FAIL'
            query = ('You will be given the history of a past experience in which you were placed in an environment and '
                     'given a task to complete. You were unsuccessful in completing the task. Do not summarize your '
                     'environment, but rather think about the strategy and path you took to attempt to complete the task. '
                     'Devise a concise, new plan of action that accounts for your mistake with reference to specific '
                     'actions that you should have taken. For example, if you tried A and B but forgot C, then devise a '
                     'plan to achieve C with environment-specific actions. You will need this later when you are solving '
                     'the same task. Give your plan after "Plan". Here are two examples:\n\n'
                     f"{PROMPTS['reflexion_few_shot_examples']}\n\n{scenario}")
            memory = self.memory[-3:]
            if memory:
                query += '\n\nPlans from past attempts:\n' + ''.join(f'Trial #{i}: {m}\n' for i, m in enumerate(memory))
            query += '\n\nNew plan:'
            log['reflection'] = call(self.cfg, query)
            self.memory.append(log['reflection'])
        return log


class EvoTest:
    scope = 'task'
    ROOT_PROMPT = 'Explore systematically and examine objects to make progress.'
    C, D = 1.0, 0.8
    EVOL_TEMPERATURE = 0.7

    def __init__(self, cfg):
        self.cfg = cfg
        self.nodes = []  # dict(prompt, parent, depth, children, score, success, trajectory)
        self.current = None

    def state(self):
        return dict(nodes=self.nodes, current=self.current)

    def load(self, state):
        self.nodes, self.current = [dict(n) for n in state['nodes']], state['current']

    def _ucb(self, n):
        big_n = max(2, len(self.nodes))
        return n['score'] + self.C * self.D ** n['depth'] * math.sqrt(math.log(big_n) / (1 + n['children']))

    def guide(self, goal):
        if not self.nodes:
            self.nodes.append(dict(prompt=self.ROOT_PROMPT, parent=None, depth=0, children=0))
            self.current = 0
            return self.ROOT_PROMPT, dict(evo_action='root', node=0)
        scored = [i for i, n in enumerate(self.nodes) if 'score' in n]
        if any(self.nodes[i]['success'] for i in scored):  # freeze_on_win
            self.current = max(scored, key=lambda i: self.nodes[i]['score'])
            return self.nodes[self.current]['prompt'], dict(evo_action='freeze', node=self.current)
        ucb = {i: self._ucb(self.nodes[i]) for i in scored}
        parent = max(ucb, key=ucb.get)
        p = self.nodes[parent]
        prompt = (PROMPTS['evotest_evolver'].replace('{{prompt}}', p['prompt']).replace('{{task}}', goal)
                  .replace('{{trajectory}}', p['trajectory']))
        new = call(self.cfg, prompt, temperature=self.EVOL_TEMPERATURE)
        if len(new) <= 10:  # EvoTest keeps the parent prompt when the evolver output is unusable
            new = p['prompt']
        p['children'] += 1
        self.nodes.append(dict(prompt=new, parent=parent, depth=p['depth'] + 1, children=0))
        self.current = len(self.nodes) - 1
        return new, dict(evo_action='evolve', node=self.current, parent=parent, ucb=ucb)

    def after_trial(self, goal, initial_observation, steps, trajectory):
        verdict = judge(self.cfg, goal, trajectory)
        n = self.nodes[self.current]
        if 'score' in n:  # rerun of a frozen node: running mean
            n['runs'] = n.get('runs', 1) + 1
            n['score'] += (verdict['p_success'] - n['score']) / n['runs']
        else:
            n['score'] = verdict['p_success']
        n['success'] = n['score'] > 0.5
        n['trajectory'] = trajectory
        return dict(judge=verdict, node=self.current)


class AWM:
    scope = 'room'

    def __init__(self, cfg):
        self.cfg = cfg
        self.workflows = ''

    def state(self):
        return dict(workflows=self.workflows)

    def load(self, state):
        self.workflows = state['workflows']

    def guide(self, goal):
        if not self.workflows:
            return '', {}
        return f'Use these workflows induced from previously solved household tasks when they apply.\n{self.workflows}', {}

    def after_trial(self, goal, initial_observation, steps, trajectory):
        verdict = judge(self.cfg, goal, trajectory)
        log = dict(judge=verdict, induced=None)
        if verdict['judged_success']:
            prompt = (PROMPTS['awm_induce'].replace('{{workflows}}', self.workflows or '(none)')
                      .replace('{{task}}', goal).replace('{{trajectory}}', trajectory))
            out = call(self.cfg, prompt)
            if 'Workflow' in out:
                self.workflows = out
                log['induced'] = out
        return log


METHODS = {'reflexion': Reflexion, 'evotest': EvoTest, 'awm': AWM}

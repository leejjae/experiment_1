"""ACE-conditioned self-distillation agent for ALFWorld (Generator side).

Per step:
  1. The memory-free JitRL prompt proposes K candidate actions (verbalized JSON,
     same prompt, temperature and repair as the JitRL runs).
  2. Base and teacher score the SAME candidates in the SAME order with one-token
     label logprobs. Teacher = identical prompt + the full ACE playbook block.
       p(a) = pi(a | s_t),  q(a) = pi(a | s_t, F_n)   (renormalized over candidates)
  3. Execute argmax_a  log p(a) + beta * (log q(a) - log p(a)).
Only the executed action and its observation enter the shared history.
"""
import math

from src.action_utils import repair_action
from src.jitrl_agent import JitRLAgent
from src.openai_helpers import chat_completion_with_retries, extract_json_from_response, response_text

LABEL_FLOOR_MARGIN = 1.0  # a label outside top-k gets (lowest top-k logprob - margin)
TOP_LOGPROBS = 20

TEACHER_BLOCK = """PLAYBOOK (lessons accumulated from your previous attempts):
{playbook}

Use the playbook guidelines that apply to the current observation when choosing the action.
"""

SCORING_TAIL = """CANDIDATE ACTIONS:
{candidates}

Which candidate is the best next action? Answer with the candidate number only ({labels})."""


class AceSelfDistillAgent(JitRLAgent):
    def __init__(self, args, beta):
        args.enable_cross_mem = False
        super().__init__(args)
        self.beta = beta
        self.playbook_text = ''
        self.trace = []

    def start_trial(self, playbook_text):
        super().start_episode()
        self.playbook_text = playbook_text  # fixed for the whole trial
        self.trace = []
        self.counters.update(n_label_floor=0, n_teacher_changed=0, n_scoring_failures=0)

    def generate_action(self, state_node, info=None):
        info = info or {}
        goal = info.get('goal')
        valid = info.get('valid', [])
        sys_prompt, user_prompt, _ = self.get_prompts(state_node, info=info)

        # 1. Proposal: identical to the memory-free JitRL call.
        res = chat_completion_with_retries(
            model=self.args.llm_model, sys_prompt=sys_prompt, prompt=user_prompt, max_tokens=1500,
            temperature=self.args.llm_temperature, response_format=self._response_format(),
            base_url=self.args.llm_base_url, enable_thinking=False)
        proposal_raw = response_text(res)
        parsed = extract_json_from_response(proposal_raw) or {}
        candidates = []
        for i in range(1, self.args.top_actions + 1):
            opt = parsed.get(f'option{i}')
            if not opt:
                continue
            opt, repaired = repair_action(opt, valid, goal, self.action_mode)
            if repaired:
                self.counters['n_action_repairs'] += 1
            if opt not in candidates:
                candidates.append(opt)
        if not parsed:
            self.counters['n_json_parse_failures'] += 1
        if not candidates:
            candidates = [valid[0] if valid else 'look']

        # 2. Score the same candidates, same order, with and without the playbook.
        context = user_prompt.split('TASK:\n1. Analyze progress')[0]
        tail = SCORING_TAIL.format(candidates='\n'.join(f'{i}. {a}' for i, a in enumerate(candidates, 1)),
                                   labels=', '.join(str(i) for i in range(1, len(candidates) + 1)))
        base_prompt = context + tail
        teacher_prompt = context + TEACHER_BLOCK.format(playbook=self.playbook_text) + '\n' + tail
        log_p = self._label_logprobs(sys_prompt, base_prompt, len(candidates))
        log_q = log_p if not self._playbook_has_bullets() else self._label_logprobs(sys_prompt, teacher_prompt, len(candidates))

        # 3. z_new = z + beta * log(q/p); z and log p differ by a constant.
        scores = [lp + self.beta * (lq - lp) for lp, lq in zip(log_p, log_q)]
        j = max(range(len(candidates)), key=lambda i: scores[i])
        base_j = max(range(len(candidates)), key=lambda i: log_p[i])
        if j != base_j:
            self.counters['n_teacher_changed'] += 1
        action = candidates[j]

        self._tried_this_episode.add(action)
        self._add_to_game_history(state_node, action, proposal_raw, info)
        self.trace.append(dict(candidates=candidates, log_p=log_p, log_q=log_q, scores=scores,
                               chosen=j, base_choice=base_j, proposal_raw=proposal_raw))
        return action, proposal_raw

    def _playbook_has_bullets(self):
        return any(line.lstrip().startswith('[') for line in self.playbook_text.splitlines())

    def _label_logprobs(self, sys_prompt, prompt, n):
        """One-token label logprobs renormalized over the n candidate labels."""
        res = chat_completion_with_retries(
            model=self.args.llm_model, sys_prompt=sys_prompt, prompt=prompt, max_tokens=1,
            temperature=0.0, top_logprobs=TOP_LOGPROBS, base_url=self.args.llm_base_url, enable_thinking=False)
        try:
            top = res.choices[0].logprobs.content[0].top_logprobs
        except (AttributeError, IndexError, TypeError):
            raise RuntimeError('Label scoring returned no logprobs')
        found = {}
        for cand in top:
            tok = cand.token.strip()
            if tok.isdigit() and 1 <= int(tok) <= n:
                found.setdefault(int(tok), cand.logprob)
        if not found:
            self.counters['n_scoring_failures'] += 1
            raise RuntimeError(f'No candidate label in top-{TOP_LOGPROBS} logprobs')
        floor = min(c.logprob for c in top) - LABEL_FLOOR_MARGIN
        raw = []
        for i in range(1, n + 1):
            if i not in found:
                self.counters['n_label_floor'] += 1
            raw.append(found.get(i, floor))
        norm = math.log(sum(math.exp(x) for x in raw))
        return [x - norm for x in raw]

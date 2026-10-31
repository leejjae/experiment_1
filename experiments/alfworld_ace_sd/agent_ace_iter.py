"""Iterative self-distillation with ACE playbook teachers (Proposed method, teacher = playbook).

ACE-SD scores with one teacher, the current playbook M: log p + beta * (log q - log p).
Here trial k of a task uses one teacher per playbook snapshot of this task,
M_1..M_k (M_j = the room playbook just before trial j of this task), and combines
their log-ratios r_j = log q_j - log p with the rule of agent_iter.combine:

  score = log p + beta * R,  R = mean_j r_j  or  EMA_j r_j (R_1 = r_1, R_j = (1 - lam) R_{j-1} + lam r_j)

With a single snapshot (trial 1) this is exactly ACE-SD. A snapshot without bullets
gives q = p (r = 0), as in ACE-SD. Identical snapshots are scored once.
"""
from src.action_utils import repair_action
from src.openai_helpers import chat_completion_with_retries, extract_json_from_response, response_text
from agent import SCORING_TAIL, TEACHER_BLOCK, AceSelfDistillAgent
from agent_iter import RULES, combine


def has_bullets(playbook_text):
    return any(line.lstrip().startswith('[') for line in playbook_text.splitlines())


class IterativeAceAgent(AceSelfDistillAgent):
    def __init__(self, args, beta, rule, lam=0.5):
        super().__init__(args, beta)
        assert rule in RULES
        self.rule, self.lam = rule, lam
        self.snapshots = []

    def start_trial(self, snapshots):
        super().start_trial(snapshots[-1])
        self.snapshots = list(snapshots)  # M_1..M_k of this task, fixed for the whole trial

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

        # 2. Score the same candidates, same order: base once, then one teacher per playbook snapshot.
        context = user_prompt.split('TASK:\n1. Analyze progress')[0]
        tail = SCORING_TAIL.format(candidates='\n'.join(f'{i}. {a}' for i, a in enumerate(candidates, 1)),
                                   labels=', '.join(str(i) for i in range(1, len(candidates) + 1)))
        log_p = self._label_logprobs(sys_prompt, context + tail, len(candidates))
        cache = {}
        for m in self.snapshots:
            if m not in cache:
                cache[m] = log_p if not has_bullets(m) else self._label_logprobs(
                    sys_prompt, context + TEACHER_BLOCK.format(playbook=m) + '\n' + tail, len(candidates))
        log_qs = [cache[m] for m in self.snapshots]

        # 3. log pi_k = log p + beta * R, R = rule-combined (log q_j - log p)
        ratios = [[q - p for q, p in zip(lq, log_p)] for lq in log_qs]
        scores = [lp + self.beta * x for lp, x in zip(log_p, combine(ratios, self.rule, self.lam))]
        j = max(range(len(candidates)), key=lambda i: scores[i])
        base_j = max(range(len(candidates)), key=lambda i: log_p[i])
        if j != base_j:
            self.counters['n_teacher_changed'] += 1
        action = candidates[j]

        self._tried_this_episode.add(action)
        self._add_to_game_history(state_node, action, proposal_raw, info)
        self.trace.append(dict(candidates=candidates, log_p=log_p, log_q=log_qs, scores=scores,
                               chosen=j, base_choice=base_j, proposal_raw=proposal_raw))
        return action, proposal_raw

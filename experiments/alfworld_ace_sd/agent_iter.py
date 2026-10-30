"""Iterative self-distillation agent for ALFWorld (per-task feedback chain).

  pi_1 = base policy;  q_j(a|s) = pi_1(a | s, f_j),  f_j = Reflector output after trial j
  pi_{k+1} ∝ pi_k (q_k / pi_1)^beta   =>   log pi_k = log pi_1 + beta * sum_{j<k} (log q_j - log pi_1)

Proposal, candidate repair and label scoring are identical to AceSelfDistillAgent;
only the teacher conditioning (one reflection per q_j instead of one playbook) and
the score differ. Trial 1 has no feedback, so it executes argmax log pi_1.

How the per-trial log-ratios r_j = log q_j - log pi_1 are combined into R (score = log pi_1 + beta * R):
  sum      R = sum_j r_j                       (the derivation above)
  replace  R = r_{k-1}                         (latest feedback only)
  mean     R = mean_j r_j
  ema      R_1 = r_1, R_j = (1 - lam) R_{j-1} + lam r_j
"""
from src.action_utils import repair_action
from src.openai_helpers import chat_completion_with_retries, extract_json_from_response, response_text
from agent import SCORING_TAIL, AceSelfDistillAgent

FEEDBACK_BLOCK = """FEEDBACK ON YOUR PREVIOUS ATTEMPT AT THIS TASK:
{feedback}

Use this feedback when it applies to the current observation when choosing the action.
"""


RULES = ('sum', 'replace', 'mean', 'ema')


def combine(ratios, rule, lam):
    """ratios: one list of r_j per feedback (oldest first), all over the same candidates."""
    if rule == 'sum':
        return [sum(col) for col in zip(*ratios)]
    if rule == 'replace':
        return list(ratios[-1])
    if rule == 'mean':
        return [sum(col) / len(col) for col in zip(*ratios)]
    if rule == 'ema':
        acc = list(ratios[0])
        for r in ratios[1:]:
            acc = [(1 - lam) * a + lam * x for a, x in zip(acc, r)]
        return acc
    raise ValueError(rule)


class IterativeSelfDistillAgent(AceSelfDistillAgent):
    def __init__(self, args, beta, rule='sum', lam=0.5):
        super().__init__(args, beta)
        assert rule in RULES
        self.rule, self.lam = rule, lam

    def start_trial(self, feedbacks):
        super().start_trial('')
        self.feedbacks = list(feedbacks)  # f_1..f_{k-1} of this task, fixed for the whole trial

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

        # 2. Score the same candidates, same order: base once, then one teacher per feedback.
        context = user_prompt.split('TASK:\n1. Analyze progress')[0]
        tail = SCORING_TAIL.format(candidates='\n'.join(f'{i}. {a}' for i, a in enumerate(candidates, 1)),
                                   labels=', '.join(str(i) for i in range(1, len(candidates) + 1)))
        log_p = self._label_logprobs(sys_prompt, context + tail, len(candidates))
        log_qs = [self._label_logprobs(sys_prompt, context + FEEDBACK_BLOCK.format(feedback=f) + '\n' + tail,
                                       len(candidates)) for f in self.feedbacks]

        # 3. log pi_k = log p + beta * R, R = rule-combined (log q_j - log p)
        if log_qs:
            ratios = [[q - p for q, p in zip(lq, log_p)] for lq in log_qs]
            scores = [lp + self.beta * x for lp, x in zip(log_p, combine(ratios, self.rule, self.lam))]
        else:
            scores = list(log_p)
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

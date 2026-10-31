"""JitRL with the paper's advantage normalization (arXiv 2601.18510, Eq. 25).

Code (ALFWorld port and upstream Jericho/WebArena) divides every advantage by the
largest *positive* advantage over all retrieved actions, so the best action is
always exactly 1 and negatives are unbounded. The paper's Eq. 25 divides by the
largest *absolute* advantage over the candidate set C:

    A_tilde(s, a) = A(s, a) / (max_{a' in C} |A(s, a')| + eps)

`update_scores` below is ALFWorld/src/jitrl_agent.py:184-307 verbatim except the
block marked PAPER EQ. 25. Everything else (retrieval = WebArena upstream k=3,
judge-only scaled rewards, exploration, episode weight) is inherited unchanged.

Arms (args.normalization_arm):
  eq25        Eq. 25 normalization, score = p + w*A_tilde        (update_rule prob)
  eq25_logit  Eq. 25 normalization, score = log p + w*A_tilde    (update_rule logprob,
              the paper's additive logit update Eq. 10). Best memory-on JitRL variant.
"""
import math
import random

from src.action_utils import denormalize_action
from experiments.alfworld_jitrl.upstream_retrieval import UpstreamRetrievalAgent

EPS = 1e-8
ARM_UPDATE_RULE = {'eq25': 'prob', 'eq25_logit': 'logprob'}


def paper_normalize(action_advantages, candidates):
    """Eq. 25: divide by max |A| over the candidate set C (plus eps)."""
    in_c = [abs(action_advantages[a]) for a in candidates if a in action_advantages]
    denom = (max(in_c) if in_c else 0.0) + EPS
    return {a: v / denom for a, v in action_advantages.items()}


class PaperNormalizationAgent(UpstreamRetrievalAgent):
    def __init__(self, args):
        arm = args.normalization_arm
        args.update_rule = ARM_UPDATE_RULE[arm]
        super().__init__(args)
        self.update_rule = args.update_rule
        self.normalization_arm = arm

    def update_scores(self, state_node, options_with_logits, k, r, memory_text, info=None):
        if not self.enable_cross_mem or self.cross_mem is None:
            return {}

        nearest = self.cross_mem.retrieve_similar(
            game_history=self.game_history,
            current_state=state_node.state,
            current_summary=memory_text,
            k=k, r=r, info=info,
        )
        if not nearest:
            return {}
        self.counters["n_steps_with_retrieval"] += 1

        goal = (info or {}).get("goal")
        valid = (info or {}).get("valid", [])

        action_rewards = {}
        for _sim, _dret, res in nearest:
            a = (res.get("action") or "").strip()
            if not a:
                continue
            action_rewards.setdefault(a, []).append(res.get("discounted_reward", 0))

        if not options_with_logits:
            options_with_logits = {}

        existing_norms = {od.get("normalized_action") for od in options_with_logits.values()
                          if isinstance(od, dict)}
        for norm_action, rewards in action_rewards.items():
            if norm_action in existing_norms:
                continue
            if not rewards or sum(rewards) / len(rewards) <= 0:
                continue
            grounded = denormalize_action(norm_action, valid, goal, self.action_mode,
                                          tried=self._tried_this_episode)
            if grounded is None:
                print(f"[Memory] Skipped historical action (not admissible here): {norm_action}")
                continue
            new_num = max(options_with_logits.keys()) + 1 if options_with_logits else 1
            options_with_logits[new_num] = {
                "action": grounded,
                "normalized_action": norm_action,
                "normalized_prob": 0,
            }
            existing_norms.add(norm_action)
            self.counters["n_memory_actions_injected"] += 1
            print(f"[Memory] Added historical action: {grounded}  (as {norm_action})")

        action_avg_rewards = {a: (sum(v) / len(v) if v else 0) for a, v in action_rewards.items()}

        all_rewards = [x for v in action_rewards.values() for x in v]
        count = len(all_rewards)
        overall_avg_reward = sum(all_rewards) / count if count else 0

        exploration_prob = self.calculate_exploration_probability(nearest, action_rewards)
        alpha = self.exploration_alpha
        new_actions_added = False
        for option_data in options_with_logits.values():
            norm = option_data.get("normalized_action")
            if norm not in action_avg_rewards:
                if random.random() < exploration_prob:
                    action_avg_rewards[norm] = overall_avg_reward + (alpha / count if count else alpha)
                else:
                    action_avg_rewards[norm] = 0
                new_actions_added = True
        if new_actions_added and action_avg_rewards:
            overall_avg_reward = sum(action_avg_rewards.values()) / len(action_avg_rewards)

        action_advantages = {a: v - overall_avg_reward for a, v in action_avg_rewards.items()}

        # ── PAPER EQ. 25 (replaces the max-positive normalization) ─────────
        candidates = {od.get("normalized_action") for od in options_with_logits.values()
                      if isinstance(od, dict)}
        normalized_advantages = paper_normalize(action_advantages, candidates) if action_advantages else {}
        # ───────────────────────────────────────────────────────────────────

        updated = {}
        if options_with_logits and normalized_advantages:
            episode_weight = 1.0
            if self.enable_cross_mem and self.cross_mem:
                ep = self.cross_mem.current_episode_number
                episode_weight = min(1.0 + (ep / 50.0) * 0.5, 1.5)

            print("\n=== Action Advantage Analysis & Logit Correction ===")
            print(f"Overall action average reward baseline: {overall_avg_reward:.4f}")

            for num, od in options_with_logits.items():
                if not isinstance(od, dict) or "action" not in od:
                    continue
                norm = od.get("normalized_action")
                adv = normalized_advantages.get(norm, 0)
                weighted_adv = adv * episode_weight
                p = od.get("normalized_prob", 0)

                if self.update_rule == "logprob":
                    base = math.log(max(p, 1e-6))
                else:
                    base = p
                corrected = base + weighted_adv

                print(f"  [{num}] {od['action']}  norm={norm}  "
                      f"p={p:.3f} base={base:.3f} adv={adv:+.3f} -> {corrected:.3f}")
                updated[num] = {
                    **od,
                    "normalized_advantage": adv,
                    "corrected_logprob": corrected,
                    "avg_reward": action_avg_rewards.get(norm, 0),
                }
        return updated

"""JitRL agent for ALFWorld.

The algorithm is unchanged from the paper's implementation:

    retrieve similar steps -> bucket discounted returns by action -> Q(a)
    V = pooled mean baseline;  A(a) = Q(a) - V;  one-sided max-normalisation
    score(a) = f(p_LLM(a)) + w * A_norm(a),   w = min(1 + ep/50*0.5, 1.5)
    a* = argmax score(a)

Port-specific changes, each of which the upstream code gets wrong on ALFWorld:

* **Action identity.** Upstream keys Q-values on raw action strings
  (jitrl_agent.py:262-283). ALFWorld actions carry instance ids, so every bucket
  would have n=1, `calculate_exploration_probability` skips actions with <2
  samples (:167-169), and exploration_prob pins at 0.8 forever -- at which point
  80% of unseen candidates get a *random* optimism bonus and the advantage term
  is noise, which is worse than no memory at all. We key on normalized actions.

* **NameError on LLM failure.** Upstream defines `options_with_logits` only
  inside the success branch but reads it unconditionally (:766-769), so one
  transient API error kills a whole stream.

* **<think> leakage.** Upstream's only cleanup is a markdown-fence strip
  (:657-671); a single unhandled think prefix makes every step of an episode
  return the default action, silently.

* **top_logprobs gating.** Upstream requests logprobs on every call (:646) even
  in verbalized mode, where they are never read.

* **logit mode robustness.** Upstream assumes the best_action digit is at
  `logprobs.content[-2]` (:723). We scan backwards for the first token that
  parses as an option index.

* **Victory detection.** Upstream keyword-matches the observation (:847-860).
  ALFWorld gives a real `won` flag.
"""

import math
import os
import random
import time

import numpy as np

from .action_utils import normalize_action, denormalize_action, repair_action
from .cross_episode_memory import CrossEpisodeMemory
from .openai_helpers import (
    chat_completion_with_retries,
    extract_json_from_response,
    response_text,
)
from .utils import generate_history_summary

SYS_PROMPT = """You are an embodied household agent operating in a simulated room. \
You are given a task goal and must complete it by manipulating objects and receptacles.

Every task decomposes into these subgoals:
  (1) NAVIGATE to a receptacle that plausibly holds the target object
  (2) OPEN it if it is openable and closed
  (3) TAKE the target object
  (4) TREAT it if the goal requires (clean with sinkbasin / heat with microwave /
      cool with fridge / examine under desklamp)
  (5) NAVIGATE to the destination receptacle
  (6) PUT the object there

Key facts about this environment:
- You cannot see the contents of a receptacle until you `go to` it (and `open` it if openable).
- Object and receptacle names carry instance numbers (e.g. `mug 1`, `desk 2`). These are \
REQUIRED - never strip or alter them.
- Receptacles you have already searched are listed under [SEARCHED]. Do NOT search them \
again unless you have a concrete new reason.
- Your current subgoal stage is given as [PHASE]. Choose actions that advance it."""

CONSTRAINT = """

**CRITICAL CONSTRAINT**: When REFERENCE ACTIONS are provided, you MUST ONLY choose actions \
from that list. Any action not in the REFERENCE ACTIONS list is INVALID and will fail. \
Do NOT create custom actions. The list is unordered - position doesn't indicate quality."""


class JitRLAgent:
    def __init__(self, args, guiding_prompt: str = None):
        self.args = args
        self.guiding_prompt = guiding_prompt or ""
        self.game_history = []
        self.counters = {}

        self.enable_cross_mem = getattr(args, "enable_cross_mem", True)
        self.exploration_rate = getattr(args, "exploration_rate", 0.65)
        self.exploration_alpha = getattr(args, "exploration_alpha", 1.0)
        self.update_rule = getattr(args, "update_rule", "prob")
        self.action_mode = getattr(args, "action_abstraction", "goal_slot")

        if self.enable_cross_mem:
            model_slug = args.llm_model.replace("/", "_").replace("\\", "_")
            self.cross_mem_dir = os.path.join(
                args.output_path, "alfworld", args.room_type, args.arm, model_slug)
            self.cross_mem = CrossEpisodeMemory(
                self.cross_mem_dir,
                gamma=args.gamma,
                llm_model=args.llm_model,
                eval_llm_model=args.eval_llm_model,
                eval_base_url=args.eval_base_url,
                eval_temperature=args.eval_temperature,
                eval_reasoning_effort=args.eval_reasoning_effort,
                failure_penalty=args.failure_penalty,
            )
            print(f"[JitRL] memory at {self.cross_mem_dir}: {self.cross_mem.get_stats()}")
        else:
            self.cross_mem_dir = None
            self.cross_mem = None

    # ── driver contract ────────────────────────────────────────────────────
    def start_episode(self):
        self.game_history = []
        self._tried_this_episode = set()
        self.counters = {
            "n_steps_with_retrieval": 0,
            "n_advantage_overrides": 0,
            "n_memory_actions_injected": 0,
            "n_json_parse_failures": 0,
            "n_action_repairs": 0,
            "n_conf_degenerate": 0,
            "n_logit_fallbacks": 0,
        }
        if self.enable_cross_mem and self.cross_mem:
            self.cross_mem.save_guiding_prompt(self.guiding_prompt)

    def end_episode(self, state, score, won=False):
        print(f"[JitRL] episode end: won={won} score={score}")
        if not self.enable_cross_mem or self.cross_mem is None:
            return
        if not self.game_history:
            return
        goal_text = self.game_history[-1].get("goal_text", "")
        self.cross_mem.add_episode(
            game_history=self.game_history,
            state=state,
            final_score=score,
            success=bool(won),          # real flag, not a keyword heuristic
            goal_text=goal_text,
        )

    def update_game_history_reward(self, reward, score):
        if self.game_history:
            self.game_history[-1]["reward"] = reward
            self.game_history[-1]["score"] = score

    # ── exploration schedule (unchanged from upstream) ─────────────────────
    def calculate_exploration_probability(self, nearest_trajectories, action_rewards):
        if not action_rewards:
            return 0.8

        action_scores = []
        for action, rewards in action_rewards.items():
            n_samples = len(rewards)
            if n_samples < 2:
                continue
            mean_reward = float(np.mean(rewards))
            std_reward = float(np.std(rewards))
            cv = std_reward / mean_reward if mean_reward > 0 else float("inf")

            performance_score = max(0.0, mean_reward)
            stability_score = 1.0 / (1.0 + cv)
            sample_confidence = max(n_samples / 4.0, 1)
            confidence = (performance_score * stability_score) ** self.exploration_rate * sample_confidence
            action_scores.append({
                "action": action, "confidence": confidence, "mean": mean_reward,
                "std": std_reward, "cv": cv, "n_samples": n_samples,
            })

        if not action_scores:
            print("[Exploration] No stable actions found -> High exploration (0.80)")
            return 0.8

        best = max(action_scores, key=lambda x: x["confidence"])
        prob = max(0.05, min(0.8, 1 / (1 + best["confidence"])))
        print(f"[Exploration] Best action '{best['action']}': mean={best['mean']:.2f}, "
              f"std={best['std']:.2f}, cv={best['cv']:.2f}, samples={best['n_samples']}, "
              f"confidence={best['confidence']:.2f} -> exploration_prob={prob:.3f}")
        return prob

    # ── advantage estimation + logit modulation ────────────────────────────
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

        # Bucket returns by NORMALIZED action -- this is the change that makes
        # Q-values transfer across episodes at all.
        action_rewards = {}
        for _sim, _dret, res in nearest:
            a = (res.get("action") or "").strip()
            if not a:
                continue
            action_rewards.setdefault(a, []).append(res.get("discounted_reward", 0))

        if not options_with_logits:
            options_with_logits = {}

        # Inject profitable historical actions as extra candidates, re-grounded
        # to something currently admissible. Upstream injected the raw string,
        # which in ALFWorld usually names an object that isn't here.
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

        if action_advantages:
            adv_values = list(action_advantages.values())
            positive = [a for a in adv_values if a > 0]
            if positive:
                m = max(positive)
                normalized_advantages = {a: v / m for a, v in action_advantages.items()}
            else:
                m = abs(min(adv_values))
                normalized_advantages = ({a: v / m for a, v in action_advantages.items()}
                                         if m > 0 else {a: 0 for a in action_advantages})
        else:
            normalized_advantages = {}

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
                    # The paper's KL closed form adds to a log-probability. Scales
                    # then match: log p in [-5, 0] vs w*A in [-1.5, 1.5].
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

    # ── prompting ──────────────────────────────────────────────────────────
    def get_prompts(self, state_node, info=None):
        info = info or {}
        current_inventory = info.get("inv")

        if self.game_history:
            summary, memory_text = generate_history_summary(
                self.game_history, state_node.state,
                llm_model=self.args.llm_model, temperature=0.0, max_tokens=1000,
                current_inventory=current_inventory, goal_text=state_node.goal_text,
                info=info, base_url=self.args.llm_base_url)
        else:
            summary = ""
            memory_text = info.get("prompt_block", "")
            if current_inventory:
                memory_text += f"\n\n[INVENTORY]\n{current_inventory}"

        sys_prompt = SYS_PROMPT
        if self.guiding_prompt:
            sys_prompt += f"\n\nFollow this guide: {self.guiding_prompt}"

        valid_actions = list(info.get("valid", [])) if self.args.use_valid_actions else []
        valid_text = ""
        if valid_actions:
            sys_prompt += CONSTRAINT
            prompt_seed = getattr(self.args, 'prompt_order_seed', None)
            shuffle_seed = time.time_ns() if prompt_seed is None else f'{prompt_seed}:{len(self.game_history)}'
            random.Random(shuffle_seed).shuffle(valid_actions)
            valid_text = (f"\nREFERENCE ACTIONS (ONLY VALID ACTIONS):\n{valid_actions}\n\n"
                          "**STRICT REQUIREMENT**: These are the ONLY valid actions for the "
                          "current state. You MUST select your actions EXCLUSIVELY from this "
                          "list, copied EXACTLY including instance numbers.\n")

        recent = ""
        for i, e in enumerate(self.game_history[-10:]):
            idx = max(0, len(self.game_history) - 10) + i
            recent += f"Step {idx} [{e.get('phase', '?')}]: {e.get('action', '')}\n"

        confidence_mode = self.args.confidence_mode
        opt_lines = []
        for i in range(1, self.args.top_actions + 1):
            opt_lines.append(f'    "reasoning{i}": "Why this action advances the current PHASE",')
            opt_lines.append(f'    "option{i}": "action copied exactly from REFERENCE ACTIONS",')
            if confidence_mode == "verbalized":
                opt_lines.append(f'    "confidence{i}": 80,')
        option_format = "\n".join(opt_lines)

        conf_rules = ""
        if confidence_mode == "verbalized":
            conf_rules = (
                f"\n- **CRITICAL**: confidence1 + ... + confidence{self.args.top_actions} MUST equal 100"
                "\n- The confidence values MUST have meaningful differences between them")

        user_prompt = f"""TASK GOAL: {state_node.goal_text}
{info.get('prompt_block', '')}

MEMORY / PROGRESS:
{summary}
{memory_text}

RECENT STEPS:
{recent or 'none'}

INVENTORY: {current_inventory}

CURRENT OBSERVATION: {state_node.state}

TASK:
1. Analyze progress: what has been achieved, what is the next objective for this [PHASE]?
2. **MANDATORY: select ONLY from the REFERENCE ACTIONS list below**
3. Propose {self.args.top_actions} different actions with reasoning

RESPONSE FORMAT (JSON):
{{
    "progress_analysis": "What you've achieved and current challenges",
    "next_objective": "Your next goal",
{option_format}
    "best_action": 1
}}

IMPORTANT:
- **ABSOLUTE REQUIREMENT**: every option MUST be copied EXACTLY from REFERENCE ACTIONS
- Do NOT create, modify, or combine actions; keep instance numbers intact
- Do NOT revisit receptacles listed under [SEARCHED] without a concrete new reason
- Don't repeat failed actions or create loops{conf_rules}

{valid_text}
"""
        return sys_prompt, user_prompt, memory_text

    def _response_format(self):
        confidence_mode = self.args.confidence_mode
        properties = {
            "progress_analysis": {"type": "string"},
            "next_objective": {"type": "string"},
        }
        required = ["progress_analysis", "next_objective"]
        choices = []
        for i in range(1, self.args.top_actions + 1):
            properties[f"reasoning{i}"] = {"type": "string"}
            properties[f"option{i}"] = {"type": "string"}
            required += [f"reasoning{i}", f"option{i}"]
            choices.append(i)
            if confidence_mode == "verbalized":
                # No minimum/maximum -- xgrammar rejects numeric ranges and the
                # silent backend fallback costs a per-request grammar compile.
                properties[f"confidence{i}"] = {"type": "integer"}
                required.append(f"confidence{i}")

        # enum, not a bounded number: in logit mode this restricts the sampler to
        # exactly K digit tokens, giving a clean distribution over option indices.
        properties["best_action"] = {"type": "integer", "enum": choices}
        required.append("best_action")

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "alfworld_action_choice",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    # ── action selection ───────────────────────────────────────────────────
    def _extract_logit_probs(self, res_obj, all_options):
        """Scan tokens backwards for the best_action digit.

        Upstream hardcodes `logprobs.content[-2]` (:723), which breaks on any
        trailing whitespace, a multi-digit number, or a different tokenizer.
        """
        try:
            content = res_obj.choices[0].logprobs.content
        except (AttributeError, IndexError, TypeError):
            return {}
        if not content:
            return {}

        for tok in reversed(content):
            try:
                num = int(tok.token.strip())
            except (ValueError, AttributeError, TypeError):
                continue
            if num not in all_options:
                continue
            out = {}
            for cand in (tok.top_logprobs or []):
                try:
                    n = int(cand.token.strip())
                except (ValueError, AttributeError, TypeError):
                    continue
                if n in all_options:
                    out[n] = math.exp(cand.logprob)
            if out:
                return out
        return {}

    def generate_action(self, state_node, info=None):
        info = info or {}
        sys_prompt, user_prompt, memory_text = self.get_prompts(state_node, info=info)
        confidence_mode = self.args.confidence_mode
        goal = info.get("goal")
        valid = info.get("valid", [])

        # Initialised before the branch: upstream's NameError (:766-769).
        options_with_logits = {}
        action_text = None
        full_response = ""

        res_obj = chat_completion_with_retries(
            model=self.args.llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=1500,
            temperature=self.args.llm_temperature,
            # Gated: upstream sent this unconditionally even in verbalized mode.
            top_logprobs=self.args.top_actions if confidence_mode == "logit" else 0,
            response_format=self._response_format(),
            base_url=self.args.llm_base_url,
            enable_thinking=False,
        )

        full_response = response_text(res_obj)
        json_response = extract_json_from_response(full_response)

        if not json_response:
            self.counters["n_json_parse_failures"] += 1
            print(f"[WARN] JSON parse failed ({self.counters['n_json_parse_failures']} this episode)")
        else:
            print(f"[ANALYSIS] {json_response.get('progress_analysis', '')[:160]}")
            print(f"[NEXT] {json_response.get('next_objective', '')[:160]}")

            all_options, all_confidences = {}, {}
            for i in range(1, self.args.top_actions + 1):
                opt = json_response.get(f"option{i}")
                if opt:
                    all_options[i] = opt
                if confidence_mode == "verbalized":
                    try:
                        all_confidences[i] = int(json_response.get(f"confidence{i}", 50))
                    except (TypeError, ValueError):
                        all_confidences[i] = 50

            if confidence_mode == "verbalized":
                confs = [all_confidences.get(i, 50) for i in all_options]
                if confs and (max(confs) - min(confs) < 5):
                    self.counters["n_conf_degenerate"] += 1
                    print(f"[CONF] degenerate distribution {confs} -- advantage will dominate")
                total = sum(confs) or 100
                for i, opt in all_options.items():
                    options_with_logits[i] = {
                        "action": opt,
                        "normalized_action": normalize_action(opt, goal, self.action_mode),
                        "normalized_prob": all_confidences.get(i, 50) / total,
                        "confidence": all_confidences.get(i, 50),
                    }
            else:
                probs = self._extract_logit_probs(res_obj, all_options)
                if not probs:
                    self.counters["n_logit_fallbacks"] += 1
                    print("[WARN] logit extraction failed; falling back to uniform prior")
                    probs = {i: 1.0 / max(1, len(all_options)) for i in all_options}
                for i, opt in all_options.items():
                    options_with_logits[i] = {
                        "action": opt,
                        "normalized_action": normalize_action(opt, goal, self.action_mode),
                        "normalized_prob": probs.get(i, 1e-6),
                    }

            best_idx = json_response.get("best_action")
            try:
                best_idx = int(best_idx)
            except (TypeError, ValueError):
                best_idx = None
            if best_idx in all_options:
                action_text = all_options[best_idx]
            elif all_options:
                action_text = all_options[min(all_options)]

        llm_choice = action_text

        updated = self.update_scores(
            state_node, options_with_logits,
            k=self.args.retrieval_top_k, r=self.args.retrieval_threshold,
            memory_text=memory_text, info=info)

        if updated:
            best_num, best_data = max(
                updated.items(), key=lambda x: x[1].get("corrected_logprob", float("-inf")))
            action_text = best_data["action"]
            if llm_choice and action_text != llm_choice:
                self.counters["n_advantage_overrides"] += 1
                print(f"[JITRL] advantage_override: {llm_choice} -> {action_text}")

        if action_text is None:
            action_text = valid[0] if valid else "look"

        ranked = [od["action"] for _, od in sorted(
            options_with_logits.items(),
            key=lambda x: -x[1].get("normalized_prob", 0))]
        action_text, repaired = repair_action(
            action_text, valid, goal, self.action_mode, fallback_ranked=ranked)
        if repaired:
            self.counters["n_action_repairs"] += 1
            print(f"[REPAIR] -> {action_text}")

        self._tried_this_episode.add(action_text)
        self._add_to_game_history(state_node, action_text, full_response, info)
        return action_text.strip(), full_response

    def _add_to_game_history(self, state_node, action, full_response, info):
        """Carry the retrieval keys on the history entry.

        Upstream stashed them in `step_context_cache` and read them back by step
        index during storage, so retrieval had to run exactly once per step in
        order or the metadata silently desynchronised. Keeping them here removes
        the coupling entirely.
        """
        self.game_history.append({
            "state": state_node.state,
            "action": action,
            "normalized_action": normalize_action(action, info.get("goal"), self.action_mode),
            "full_response": full_response,
            "reward": None,
            "score": None,
            "phase": info.get("phase"),
            "goal_text": info.get("goal_text", ""),
            "goal_key": info.get("goal_key", ""),
            "history_key": info.get("history_key", ""),
            "state_key": info.get("state_key", ""),
            "location_key": info.get("location_key", ""),
        })

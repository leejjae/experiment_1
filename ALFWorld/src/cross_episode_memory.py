"""Non-parametric cross-episode memory for ALFWorld.

Two structural changes from the Jericho original:

1. **No vectors.** FAISS was only a recall filter -- the final ranking is 100%
   Jaccard (Jericho/src/cross_episode_memory.py:428-432) -- and it required a
   second OpenAI key for ada-002 embeddings. Worse, `recall_size =
   min(max(k*10,100), ntotal)` means that late in a stream the Jaccard ranker
   would only ever see ~8% of the corpus, filtered by a *cosine* criterion that
   does not agree with it. With ~1,200 step records per room-type stream an
   exhaustive scan costs milliseconds. WebArena already runs a no-vector path
   (`encoder_available = False`).

2. **No positional step cache.** The original appends to `step_context_cache`
   inside retrieval and reads it back by step index during storage
   (:378 vs :251), so retrieval had to be called exactly once per step, in
   order, or the stored metadata silently desynchronised. Here the retrieval
   keys are carried on the game_history entries themselves, so storage cannot
   drift out of alignment with retrieval.

Retrieval scoring (see state_construction.py for how the keys are built):

    hard prefilter:  phase == phase  AND  location_key == location_key
    score = 0.25*J(goal_key,1) + 0.35*J(history_key,1) + 0.40*J(state_key,4)
"""

import json
import os
import pickle
from collections import Counter
from typing import Any, Dict, List, Optional

from .utils import evaluate_step_scores_with_llm


class CrossEpisodeMemory:

    def __init__(self, base_dir: str, gamma: float = 0.95, llm_model: str = "qwen3-32b",
                 eval_llm_model: str = None, eval_base_url: str = None,
                 eval_temperature: float = 0.3, eval_reasoning_effort: str = "medium",
                 similarity_weights=(0.25, 0.35, 0.40), failure_penalty: float = -10.0):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        self.gamma = gamma
        self.llm_model = llm_model
        self.eval_llm_model = eval_llm_model or llm_model
        self.eval_base_url = eval_base_url
        self.eval_temperature = eval_temperature
        self.eval_reasoning_effort = eval_reasoning_effort
        self.w_goal, self.w_history, self.w_state = similarity_weights
        self.failure_penalty = failure_penalty

        self.episodes_path = os.path.join(base_dir, "episodes.jsonl")
        self.prompt_history_path = os.path.join(base_dir, "prompt_history.jsonl")
        self.metadata_path = os.path.join(base_dir, "step_metadata.pkl")

        self.step_metadata: List[Dict[str, Any]] = []
        self._load_store()

        # Restore the counter from disk. The original discarded load_episodes()
        # (:571-572) so the episode_weight anneal reset to 1.0 on every restart
        # even though the memory itself persisted.
        self.current_episode_number = len(self.load_episodes())

        # Diagnostics, read by the agent for per-episode counters.
        self.last_similarity_samples: List[float] = []

    # ── persistence ────────────────────────────────────────────────────────
    def _load_store(self):
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "rb") as f:
                    self.step_metadata = pickle.load(f)
                print(f"[Memory] loaded {len(self.step_metadata)} step records from {self.metadata_path}")
            except Exception as e:
                print(f"[Memory] failed to load metadata ({e}); starting empty")
                self.step_metadata = []
        else:
            self.step_metadata = []

    def _save_store(self):
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.step_metadata, f)

    def clear_memory(self, save_to_disk: bool = True):
        self.step_metadata = []
        self.current_episode_number = 0
        if save_to_disk:
            for p in (self.metadata_path, self.episodes_path, self.prompt_history_path):
                if os.path.exists(p):
                    os.remove(p)

    # Kept under the upstream name so existing call sites keep working.
    clear_vector_database = clear_memory

    def _append_jsonl(self, path, obj):
        with open(path, "a") as f:
            f.write(json.dumps(obj) + "\n")

    def _load_jsonl(self, path):
        if not os.path.exists(path):
            return []
        out = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out

    def load_episodes(self):
        return self._load_jsonl(self.episodes_path)

    def get_stats(self):
        return {
            "n_steps": len(self.step_metadata),
            "n_episodes": self.current_episode_number,
            "base_dir": self.base_dir,
        }

    # ── similarity ─────────────────────────────────────────────────────────
    @staticmethod
    def _tokenize(text):
        return [t for t in (text or "").lower().replace("\n", " ").split()
                if t.isalpha() or t.isalnum() or "<" in t]

    @staticmethod
    def _ngrams(tokens, n):
        if len(tokens) < n:
            return tokens
        return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def _jaccard(self, a_tokens, b_tokens, ngram=1):
        a = Counter(self._ngrams(a_tokens, ngram))
        b = Counter(self._ngrams(b_tokens, ngram))
        if not a or not b:
            return 0.0
        inter = sum((a & b).values())
        union = sum((a | b).values())
        return inter / union if union else 0.0

    def _discounted_return(self, future_rewards):
        """Recomputed at retrieval time, so --gamma can be swept post hoc."""
        total = 0.0
        for u, r in enumerate(future_rewards or ()):
            total += (self.gamma ** u) * r
        return total

    # ── retrieval ──────────────────────────────────────────────────────────
    def retrieve_similar(self, game_history, current_state: str, current_summary: str = None,
                         k: int = 10, r: float = 0.5, use_vector: bool = True, info=None):
        """Return [(similarity, discounted_return, result_dict)], best first.

        result_dict carries `action` (the *normalized* action, which is what the
        agent aggregates Q-values on) and `discounted_reward`.
        """
        if not self.step_metadata or not info:
            return []

        q_phase = info.get("phase")
        q_location = info.get("location_key")
        q_goal = self._tokenize(info.get("goal_key", ""))
        q_hist = self._tokenize(info.get("history_key", ""))
        q_state = self._tokenize(info.get("state_key", ""))

        # Dynamic threshold: relaxes by up to 0.1 over the first 20 steps,
        # matching the upstream schedule (:357-368).
        step_count = len(game_history or ())
        threshold = r - 0.1 * min(step_count / 20.0, 1.0)

        scored = []
        for meta in self.step_metadata:
            if q_phase and meta.get("phase") != q_phase:
                continue
            if q_location and meta.get("location_key") != q_location:
                continue

            sim = (
                self.w_goal * self._jaccard(q_goal, self._tokenize(meta.get("goal_key", "")), 1)
                + self.w_history * self._jaccard(q_hist, self._tokenize(meta.get("history_key", "")), 1)
                # 1-gram, not the upstream 4-gram: state_key is four short
                # structured lines, so 4-grams are sparse enough that Jaccard
                # degenerates to 0-or-1. Measured on kitchen (scripts/
                # check_state_construction.py): 1-gram lifts mean similarity
                # over raw observations by +0.307 vs +0.227 for 4-gram.
                + self.w_state * self._jaccard(q_state, self._tokenize(meta.get("state_key", "")), 1)
            )
            scored.append((sim, meta))

        self.last_similarity_samples = sorted((s for s, _ in scored), reverse=True)[:50]
        if not scored:
            return []

        high = [(s, m) for s, m in scored if s > 0.98]
        mid = [(s, m) for s, m in scored if threshold < s <= 0.98]
        high.sort(key=lambda x: -x[0])
        mid.sort(key=lambda x: -x[0])
        selected = high + mid[:max(0, k - len(high))]

        results = []
        for sim, meta in selected:
            dret = self._discounted_return(meta.get("future_rewards", []))
            step_data = meta.get("step_data", {})
            results.append((sim, dret, {
                "action": step_data.get("normalized_action") or step_data.get("action", ""),
                "raw_action": step_data.get("action", ""),
                "discounted_reward": dret,
                "episode_number": meta.get("episode_number"),
                "step_num": step_data.get("step_num"),
                "phase": meta.get("phase"),
                "location_key": meta.get("location_key"),
                "episode_success": meta.get("episode_success"),
            }))
        results.sort(key=lambda x: (-x[0], -x[1]))
        return results

    # ── storage ────────────────────────────────────────────────────────────
    def add_episode(self, game_history, state: str, final_score: float = 0,
                    success: bool = False, goal_text: str = ""):
        self.current_episode_number += 1

        states = [e.get("state", "") for e in game_history]
        actions = [e.get("action", "") for e in game_history]
        rewards = [e.get("reward", 0) or 0 for e in game_history]
        scores = [e.get("score", 0) or 0 for e in game_history]

        print(f"[Memory] scoring {len(game_history)} steps with {self.eval_llm_model} ...")
        step_scores, step_reasonings = evaluate_step_scores_with_llm(
            game_history=game_history,
            state=state,
            final_score=final_score,
            success=success,
            goal_text=goal_text,
            llm_model=self.eval_llm_model,
            temperature=self.eval_temperature,
            base_url=self.eval_base_url,
            reasoning_effort=self.eval_reasoning_effort,
        )
        print(f"[Memory] step scores: {step_scores}")

        episode = {
            "episode_number": self.current_episode_number,
            "num_steps": len(states),
            "final_score": final_score,
            "success": success,
            "goal_text": goal_text,
            "steps": [],
        }

        for i in range(len(states)):
            entry = game_history[i]
            episode["steps"].append({
                "step_num": i,
                "state": states[i],
                "action": actions[i],
                "normalized_action": entry.get("normalized_action", actions[i]),
                "reward": rewards[i],
                "score": scores[i],
                "llm_step_score": step_scores[i] if i < len(step_scores) else 0,
                "llm_reasoning": step_reasonings[i] if i < len(step_reasonings) else "",
                "phase": entry.get("phase"),
            })

        if not success and episode["steps"]:
            episode["steps"][-1]["reward"] = self.failure_penalty

        # Reward-to-go. The env reward is scaled by 10 so a genuine success lands
        # on the same scale as the LLM's [-10, 10] step scores; the upstream
        # `llm_step_score if != 0 else reward` (:263) let a +1 success be masked
        # by any non-zero step score.
        steps = episode["steps"]
        for i, entry in enumerate(game_history):
            future = []
            for u in range(i, len(steps)):
                future.append(steps[u].get("llm_step_score", 0) + 10.0 * (steps[u].get("reward", 0) or 0))
            self.step_metadata.append({
                "episode_number": self.current_episode_number,
                "episode_success": success,
                "episode_final_score": final_score,
                "step_data": steps[i],
                "future_rewards": future,
                "goal_key": entry.get("goal_key", ""),
                "history_key": entry.get("history_key", ""),
                "state_key": entry.get("state_key", ""),
                "location_key": entry.get("location_key", ""),
                "phase": entry.get("phase"),
            })

        self._append_jsonl(self.episodes_path, episode)
        self._save_store()
        print(f"[Memory] stored episode {self.current_episode_number}: "
              f"{len(steps)} steps, total records {len(self.step_metadata)}")

    # ── guiding prompt (kept for interface parity; evolution stays off) ─────
    def save_guiding_prompt(self, guiding_prompt: str, episode_number: int = None):
        self._append_jsonl(self.prompt_history_path, {
            "episode_number": episode_number if episode_number is not None else self.current_episode_number,
            "guiding_prompt": guiding_prompt,
        })

    def load_latest_guiding_prompt(self) -> Optional[str]:
        hist = self._load_jsonl(self.prompt_history_path)
        return hist[-1].get("guiding_prompt") if hist else None

    def get_prompt_history(self):
        return self._load_jsonl(self.prompt_history_path)

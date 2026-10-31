"""JitRL (judge-only, scaled) with the WebArena upstream retrieval, k=3.

Only retrieval changes; rewards (judge-only, scaled), exploration and prompts are
those of experiments/alfworld_no_feedback/agent.py.

WebArena upstream (WebArena/memory_agents/utils/cross_episode_memory.py at
commit faa45f1, retrieve_similar_with_vector):
  1. hard filter: same normalized URL           -> ALFWorld: same location_key
  2. hard filter: task-goal 1-gram Jaccard > 0.27 on the raw instruction
                                                 -> ALFWorld: raw goal_text (concrete object names)
  3. similarity = 0.7 * history Jaccard(1-gram) + 0.3 * task Jaccard
     history = normalized actions so far        -> ALFWorld: normalized actions so far in the episode
  4. keep similarity >= r - 0.1*min(step/20, 1), sort by (similarity, return)
  5. top-k: upstream leaves `[:k]` commented out; we apply its --retrieval_top_k default 3.
Thresholds are upstream defaults: r=0.1, task 0.27, k=3. No phase filter (upstream has none).
"""
from collections import Counter

from experiments.alfworld_no_feedback.agent import JudgeOnlyAgent, JudgeOnlyMemory

TASK_SIMILARITY_THRESHOLD = 0.27
RETRIEVAL_TOP_K = 3
RETRIEVAL_THRESHOLD = 0.1
HISTORY_WEIGHT, TASK_WEIGHT = 0.7, 0.3


def tokens(text):
    # Same tokenizer as upstream CrossEpisodeMemory._tokenize.
    return [t for t in (text or '').lower().replace('\n', ' ').split() if t.isalpha() or t.isalnum()]


def jaccard1(a, b):
    # Upstream multiset Jaccard with ngram=1.
    ca, cb = Counter(a), Counter(b)
    if not ca or not cb:
        return 0.0
    return sum((ca & cb).values()) / sum((ca | cb).values())


class UpstreamRetrievalMemory(JudgeOnlyMemory):
    def add_episode(self, game_history, state, goal_text=''):
        before = len(self.step_metadata)
        super().add_episode(game_history, state, goal_text=goal_text)
        new = self.step_metadata[before:]
        assert len(new) == len(game_history)
        goal_tokens = tokens(goal_text)
        actions = [e.get('normalized_action', e.get('action', '')) for e in game_history]
        for i, meta in enumerate(new):
            meta['task_goal_tokens'] = goal_tokens
            meta['history_tokens'] = tokens(' '.join(actions[:i]))
        self._save_store()

    def retrieve_similar(self, game_history, current_state, current_summary=None,
                         k=RETRIEVAL_TOP_K, r=RETRIEVAL_THRESHOLD, use_vector=True, info=None):
        if not self.step_metadata or not info:
            return []
        location = info.get('location_key')
        goal_tokens = tokens(info.get('goal_text', ''))
        history_tokens = tokens(' '.join(e.get('normalized_action', e.get('action', ''))
                                         for e in (game_history or ())))
        step_count = len(game_history or ())
        threshold = r - 0.1 * min(step_count / 20.0, 1.0)

        scored = []
        for meta in self.step_metadata:
            if meta.get('location_key') != location:
                continue
            task_sim = jaccard1(goal_tokens, meta['task_goal_tokens'])
            if task_sim <= TASK_SIMILARITY_THRESHOLD:
                continue
            sim = HISTORY_WEIGHT * jaccard1(history_tokens, meta['history_tokens']) + TASK_WEIGHT * task_sim
            if sim < threshold:
                continue
            dret = self._discounted_return(meta.get('future_rewards', []))
            step_data = meta.get('step_data', {})
            scored.append((sim, dret, {
                'action': step_data.get('normalized_action') or step_data.get('action', ''),
                'raw_action': step_data.get('action', ''),
                'discounted_reward': dret,
                'episode_number': meta.get('episode_number'),
                'step_num': step_data.get('step_num'),
                'phase': meta.get('phase'),
                'location_key': meta.get('location_key'),
                'task_similarity': task_sim,
            }))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        self.last_similarity_samples = [s for s, _, _ in scored[:50]]
        return scored[:k]


class UpstreamRetrievalAgent(JudgeOnlyAgent):
    def __init__(self, args):
        args.retrieval_top_k = RETRIEVAL_TOP_K
        args.retrieval_threshold = RETRIEVAL_THRESHOLD
        super().__init__(args)
        if self.cross_mem is not None:
            self.cross_mem.__class__ = UpstreamRetrievalMemory
            for meta in self.cross_mem.step_metadata:
                assert 'task_goal_tokens' in meta, 'Memory written without upstream retrieval fields'

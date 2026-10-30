"""JitRL baseline using only LLM scores; evaluator outcomes stay outside memory."""
import math
from src.jitrl_agent import JitRLAgent
from src.cross_episode_memory import CrossEpisodeMemory
import src.cross_episode_memory as memory_module

ORACLE_KEYS = frozenset(('reward', 'score', 'success', 'won', 'final_score',
                         'outcome', 'environment_reward', 'episode_success', 'episode_final_score'))

def without_outcome(record):
    return {key: value for key, value in record.items() if key not in ORACLE_KEYS}

BASE_PROTOCOL = 'llm_judge_only_no_verifiable_feedback_v1'
SCALED_PROTOCOL = 'llm_judge_only_no_verifiable_feedback_scaled_0p1_v1'

class JudgeOnlyMemory(CrossEpisodeMemory):
    reward_scale = 1.0
    reward_protocol = BASE_PROTOCOL
    def add_episode(self, game_history, state, goal_text=''):
        assert all(not ORACLE_KEYS.intersection(entry) for entry in game_history)
        scores, reasons = memory_module.evaluate_step_scores_with_llm(
            game_history=game_history, state=state, goal_text=goal_text,
            llm_model=self.eval_llm_model, temperature=self.eval_temperature,
            base_url=self.eval_base_url, reasoning_effort=self.eval_reasoning_effort)
        if len(scores) != len(game_history) or len(reasons) != len(scores):
            raise ValueError('Incomplete judge scores; no fabricated fallback')
        if not all(math.isfinite(s) and -10 <= s <= 10 for s in scores):
            raise ValueError('Invalid judge score')
        raw_scores = list(scores)
        scores = [s * self.reward_scale for s in raw_scores]
        self.current_episode_number += 1
        steps = [dict(step_num=i, state=e.get('state', ''), action=e.get('action', ''),
                      normalized_action=e.get('normalized_action', e.get('action', '')),
                      llm_step_score=scores[i], llm_reasoning=reasons[i], phase=e.get('phase'),
                      raw_llm_step_score=raw_scores[i], reward_scale=self.reward_scale)
                 for i, e in enumerate(game_history)]
        for i, entry in enumerate(game_history):
            self.step_metadata.append(dict(episode_number=self.current_episode_number,
                step_data=steps[i], future_rewards=list(scores[i:]),
                **{k:entry.get(k, '') for k in ('goal_key','history_key','state_key','location_key','phase')}))
        self._append_jsonl(self.episodes_path, dict(episode_number=self.current_episode_number,
            num_steps=len(steps), goal_text=goal_text, steps=steps,
            reward_protocol=self.reward_protocol, reward_scale=self.reward_scale))
        self._save_store()

class JudgeOnlyAgent(JitRLAgent):
    no_verifiable_feedback = True

    def __init__(self, args):
        super().__init__(args)
        if self.cross_mem is not None:
            self.cross_mem.__class__ = JudgeOnlyMemory
            self.cross_mem.reward_scale = getattr(args, 'judge_reward_scale', 1.0)
            assert self.cross_mem.reward_scale in (1.0, 0.1)
            self.cross_mem.reward_protocol = SCALED_PROTOCOL if self.cross_mem.reward_scale == 0.1 else BASE_PROTOCOL
            for episode in self.cross_mem.load_episodes():
                assert episode.get('reward_protocol') == self.cross_mem.reward_protocol, 'Mixed reward protocols'
            # Reject old feedback-bearing memory instead of silently reusing it.
            for entry in self.cross_mem.step_metadata:
                assert not ORACLE_KEYS.intersection(entry)
                assert not ORACLE_KEYS.intersection(entry['step_data'])

    def generate_action(self, state_node, info=None):
        safe_info = without_outcome(info or {})
        result = super().generate_action(state_node, info=safe_info)
        self.game_history = [without_outcome(entry) for entry in self.game_history]
        return result

    def update_game_history_reward(self, reward, score):
        raise RuntimeError('Evaluator reward must never be passed to JudgeOnlyAgent')

    def end_episode(self, state):
        if self.enable_cross_mem and self.cross_mem is not None and self.game_history:
            self.cross_mem.add_episode(self.game_history, state,
                goal_text=self.game_history[-1].get('goal_text', ''))

"""ALFWorld environment wrapper satisfying the Jericho driver's env contract.

The JitRL agent and evaluator only consume:

    ob: str, info['score'|'valid'|'inv'|'look'], reward, done

Jericho got `look` and `inventory` for free by saving/restoring emulator state
(Jericho/src/env.py:56-64). TextWorld has no such facility, so we issue real
`look`/`inventory` steps as probes. They are pure observations -- they do not
mutate world state -- but the TextWorld gym wrapper counts every step against
`max_episode_steps`, which is why configs/base_config.yaml sets that to 1000
while we enforce the real budget ourselves via `step_limit`.

Supplying a real info['look'] matters: utils.generate_trajectory_context_for_vector
(Jericho/src/utils.py:673-687) takes a pure string-concat branch when `look` is
present, saving one LLM call per env step.
"""

import os
import re

from .state_construction import EpisodeState, GoalSpec

# Dropped from the candidate list: they are no-ops for task progress, and the
# shuffled list is pasted verbatim into the actor prompt (rooms expose 40-90
# admissible commands).
DROP_COMMANDS = {"help", "look", "inventory"}

_TASK_RE = re.compile(r"your task is to:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


class AlfWorldEnvFactory:
    """Builds AlfredTWEnv once per process.

    The AlfredTWEnv constructor walks the whole data directory (30-90s), so it
    must not be rebuilt per episode. `make()` pins a single game by shrinking
    `game_files` before init_env registers it with textworld.gym.
    """

    def __init__(self, config, split="eval_out_of_distribution"):
        from alfworld.agents.environment import get_environment

        self._tw = get_environment(config["env"]["type"])(config, train_eval=split)
        self.all_game_files = sorted(self._tw.game_files)

    def make(self, record, step_limit=40):
        # Plans store gamefiles relative to $ALFWORLD_DATA so they are portable across machines.
        self._tw.game_files = [os.path.join(os.environ.get("ALFWORLD_DATA", ""), record["gamefile"])]
        self._tw.num_games = 1
        gym_env = self._tw.init_env(batch_size=1)
        return AlfWorldEnv(gym_env, record, step_limit)


class AlfWorldEnv:
    def __init__(self, env, record, step_limit=40):
        self._env = env
        self.record = record
        self.step_limit = step_limit
        self.goal = GoalSpec.from_record(record)
        self.tracker = EpisodeState(self.goal)

        self.steps = 0
        self.score = 0.0
        self.done = False
        self.won = False
        self._last_look = ""
        self._last_inv = "You are not carrying anything."
        self.goal_text = ""

    # ── driver contract ────────────────────────────────────────────────────
    def reset(self):
        obs, infos = self._env.reset()
        ob = obs[0]
        self.steps = 0
        self.score = 0.0
        self.done = False
        self.won = False
        self.tracker = EpisodeState(self.goal)

        self.goal_text = self._parse_goal(ob)
        self.goal.goal_text = self.goal_text

        info = self._build_info(infos, ob, action=None, won=False, probe=True)
        return self._clean(ob), info

    def step(self, action):
        obs, scores, dones, infos = self._env.step([action])
        ob = obs[0]
        self.steps += 1

        won = bool(infos["won"][0])
        env_done = bool(dones[0])

        # Binary score: reward is +1.0 exactly once, on the winning step.
        new_score = 1.0 if won else 0.0
        reward = new_score - self.score
        self.score = new_score
        self.won = won

        done = env_done or won or (self.step_limit and self.steps >= self.step_limit)
        self.done = done

        info = self._build_info(infos, ob, action=action, won=won, probe=not done)
        return self._clean(ob), reward, done, info

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    # ── internals ──────────────────────────────────────────────────────────
    def _clean(self, ob):
        return (ob or "").strip()

    def _parse_goal(self, ob):
        m = _TASK_RE.search(ob or "")
        return m.group(1).strip() if m else ""

    def _probe(self):
        """Issue `look` and `inventory` as observation-only steps."""
        try:
            obs, _, _, _ = self._env.step(["look"])
            self._last_look = self._clean(obs[0])
        except Exception:
            pass
        try:
            obs, _, _, _ = self._env.step(["inventory"])
            self._last_inv = self._clean(obs[0])
        except Exception:
            pass
        return self._last_look, self._last_inv

    def _build_info(self, infos, ob, action, won, probe):
        # Capture admissible commands BEFORE probing -- the probe overwrites infos.
        valid = [c for c in infos["admissible_commands"][0] if c not in DROP_COMMANDS]
        gamefile = infos.get("extra.gamefile", [self.record["gamefile"]])[0]

        if probe:
            look, inv = self._probe()
        else:
            look, inv = self._last_look, self._last_inv

        # Exactly one tracker update per env step -- same discipline as
        # CrossEpisodeMemory.step_context_cache, which is indexed positionally.
        self.tracker.update(ob, action, inv, valid,
                            won if getattr(self, 'reveal_outcome_to_tracker', True) else False)

        return {
            "score": self.score,
            "valid": valid,
            "inv": inv,
            "look": look,
            "won": won,
            "goal": self.goal,
            "goal_text": self.goal_text,
            "phase": self.tracker.phase,
            "state_key": self.tracker.state_key(),
            "history_key": self.tracker.history_key(),
            "goal_key": self.tracker.goal_key(),
            "location_key": self.tracker.location_key(),
            "prompt_block": self.tracker.prompt_block(),
            "searched": sorted(self.tracker.searched),
            "gamefile": gamefile,
            "game_id": self.record["game_id"],
            "task_type": self.record["task_type"],
            "room_type": self.record["room_type"],
            "scene_num": self.record["scene_num"],
        }

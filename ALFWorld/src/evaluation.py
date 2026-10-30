"""Episode driver for the ALFWorld port.

Differences from the Jericho original (Jericho/src/evaluation.py):

* Iterates a room-type *game stream* instead of replaying one game N times.
  Episode i is a different ALFWorld game, which is what makes this a continual
  learning setting rather than a repeated-trial one.
* StateNode carries the task goal and subgoal phase. Jericho has no task goal at
  all, so there was no slot for it (WebArena solved the same problem by adding
  `instruction`, WebArena/memory_agents/jitrl_agent.py:21).
* Success comes from info['won'], not from keyword-matching the observation.
* The `library`-specific technician hack (:117-119) is gone.
* update_game_history_reward exceptions are logged instead of silently passed
  (:143-146) -- swallowing them is how memory corruption goes unnoticed.
* Appends one line per episode to episode_results.jsonl, which is the input to
  aggregate_results.py. The rolling summary json is kept but is not the source
  of truth.
"""

import json
import os
import random
import statistics
import time
import traceback

import numpy as np

from .env import AlfWorldEnvFactory
from .game_index import load_index, stream_for_room
from .jitrl_agent import JitRLAgent
from .naive_agent import NaiveAgent


class StateNode:
    def __init__(self, state, goal=None, goal_text="", phase="SEARCHING", reward=0.0):
        self.state = state
        self.goal = goal
        self.goal_text = goal_text
        self.phase = phase
        self.reward = reward
        self.response = ""


class AlfWorldEvaluator:

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.base_output_path = os.path.join(args.output_path, "alfworld", args.room_type)
        os.makedirs(self.base_output_path, exist_ok=True)

        if args.seed is None:
            args.seed = random.randint(0, 100000)
        np.random.seed(args.seed)
        random.seed(args.seed)

        index = load_index(args.game_index)
        self.stream = stream_for_room(index, args.room_type)
        if args.limit_episodes:
            self.stream = self.stream[:args.limit_episodes]

        self.model_slug = args.llm_model.replace("/", "_").replace("\\", "_")
        # Memory lives one level above the timestamp dir so it persists across
        # process restarts -- that is what makes the stream continual.
        self.memory_dir = os.path.join(self.base_output_path, args.arm, self.model_slug)
        os.makedirs(self.memory_dir, exist_ok=True)
        self.results_path = os.path.join(self.memory_dir, "episode_results.jsonl")

    def _done_game_ids(self):
        if not (self.args.resume and os.path.exists(self.results_path)):
            return set()
        done = set()
        with open(self.results_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "ok":
                        done.add(rec["game_id"])
                except Exception:
                    continue
        return done

    def run_evaluation(self):
        print("\n--- STARTING EVALUATION ---")
        print(f"Room type: {self.args.room_type}  |  arm: {self.args.arm}")
        print(f"Actor: {self.args.llm_model} @ {self.args.llm_base_url}")
        print(f"Evaluator: {self.args.eval_llm_model} @ {self.args.eval_base_url}")
        print(f"Episodes in stream: {len(self.stream)}")

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(self.memory_dir, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"Logs: {self.log_dir}")
        print(f"Memory: {self.memory_dir}")

        if self.args.agent_type == "jitrl":
            agent = JitRLAgent(self.args, guiding_prompt="")
        elif self.args.agent_type == "naive":
            agent = NaiveAgent(self.args, guiding_prompt="")
        else:
            raise ValueError(f"Unknown agent type: {self.args.agent_type}")

        factory = AlfWorldEnvFactory(self.config)
        skip = self._done_game_ids()
        if skip:
            print(f"Resuming: skipping {len(skip)} already-completed games")

        scores = []
        summary_data = {}
        for i, record in enumerate(self.stream):
            if record["game_id"] in skip:
                continue
            print(f"\n=== Episode {i + 1}/{len(self.stream)}: {record['game_id']} ===")
            t0 = time.time()
            try:
                result = self.run_game_episode(agent, factory, record, episode_num=i)
                result["status"] = "ok"
            except Exception as e:
                traceback.print_exc()
                result = {
                    "episode_index": i,
                    "game_id": record["game_id"],
                    "room_type": record["room_type"],
                    "task_type": record["task_type"],
                    "scene_num": record["scene_num"],
                    "arm": self.args.arm,
                    "won": False,
                    "score": 0.0,
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                }
            result["wall_seconds"] = round(time.time() - t0, 1)
            with open(self.results_path, "a") as f:
                f.write(json.dumps(result) + "\n")

            scores.append(result.get("score", 0.0))
            n_won = sum(1 for s in scores if s > 0)
            print(f"Episode {i + 1} -> won={result.get('won')} "
                  f"| running success {n_won}/{len(scores)} = {n_won / len(scores):.1%}")

            summary_data = {
                "individual_scores": scores,
                "success_rate": n_won / len(scores),
                "average_score": statistics.mean(scores),
                "stdev_score": statistics.stdev(scores) if len(scores) > 1 else 0,
                "n_episodes": len(scores),
                "timestamp": timestamp,
                "all_args": vars(self.args),
            }
            summary_path = os.path.join(
                self.log_dir, f"evaluation_summary_{self.args.arm}_{self.model_slug}_{timestamp}.json")
            with open(summary_path, "w") as f:
                json.dump(summary_data, f, indent=2)

        return summary_data

    def run_game_episode(self, agent, factory, record, episode_num=0):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(self.log_dir, f"episode_{episode_num:03d}_{timestamp}.txt")

        env = factory.make(record, step_limit=self.args.env_step_limit)
        ob, info = env.reset()
        score = info["score"]
        done = False
        step = 0

        with open(log_path, "w", encoding="utf-8") as out:
            out.write(f"Game: {record['game_id']}\n")
            out.write(f"Room: {record['room_type']}  Task: {record['task_type']}  "
                      f"Scene: {record['scene_num']}\n")
            out.write(f"Goal: {info['goal_text']}\n")
            out.write(f"GoalSpec: {info['goal']}\n")
            out.write(f"Model: {self.args.llm_model}  Arm: {self.args.arm}\n")

            agent.start_episode()
            out.write(f"Guiding prompt: {agent.guiding_prompt}\n")

            for step in range(self.args.env_step_limit):
                state_node = StateNode(
                    state=ob,
                    goal=info["goal"],
                    goal_text=info["goal_text"],
                    phase=info["phase"],
                )
                action, raw = agent.generate_action(state_node, info=info)

                out.write("==========\n")
                out.write(f"[STEP] {step}\n")
                out.write(f"[PHASE] {info['phase']}\n")
                out.write(f"[OBS] {ob}\n")
                out.write(f"[INV] {info.get('inv', 'N/A')}\n")
                out.write(f"[SEARCHED] {info.get('searched')}\n")
                if self.args.use_valid_actions:
                    out.write(f"[VALID_ACTIONS] {info.get('valid', [])}\n")
                out.write(f"[RAW_LLM_OUTPUT] {raw}\n")
                out.write(f"[CHOSEN_ACTION] {action}\n")

                prev_ob = ob
                ob, reward, done, info = env.step(action)
                score = info["score"]
                out.write(f"[REWARD] {reward}\n[CUM_REWARD] {score}\n")

                try:
                    agent.update_game_history_reward(reward, score)
                except Exception as e:
                    print(f"[WARN] update_game_history_reward failed: {e}", flush=True)
                    out.write(f"[WARN] update_game_history_reward failed: {e}\n")

                short = prev_ob[:70].replace("\n", " ").ljust(70)
                print(f"[STEP {step}] {short} || {action[:30].ljust(30)} || "
                      f"phase={info['phase']} r={reward}")

                if done:
                    out.write(f"\nEpisode finished at step {step}. won={info['won']}\n")
                    break

            if not done:
                out.write(f"\nStep limit ({self.args.env_step_limit}) reached. won={info['won']}\n")

            agent.end_episode(state=ob, score=score, won=info["won"])

        env.close()

        counters = getattr(agent, "counters", {}) or {}
        return {
            "episode_index": episode_num,
            "game_id": record["game_id"],
            "room_type": record["room_type"],
            "task_type": record["task_type"],
            "scene_num": record["scene_num"],
            "arm": self.args.arm,
            "won": bool(info["won"]),
            "score": float(score),
            "num_steps": step + 1,
            "final_phase": info["phase"],
            "log_file": log_path,
            **counters,
        }

"""Entry point: run one room-type continual stream for one arm.

    ALFWORLD_DATA=~/.cache/alfworld python main.py \
        --room_type kitchen --agent_type jitrl \
        --confidence_mode verbalized --update_rule prob
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.evaluation import AlfWorldEvaluator  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser()

    # environment / dataset
    p.add_argument("--alfworld_config", default="configs/base_config.yaml")
    p.add_argument("--game_index", default="games_unseen.json")
    p.add_argument("--room_type", required=True,
                   choices=["kitchen", "living_room", "bedroom", "bathroom"])
    p.add_argument("--output_path", default="output")
    p.add_argument("--env_step_limit", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit_episodes", type=int, default=0,
                   help="Cap the stream length (smoke tests). 0 = whole stream.")
    p.add_argument("--resume", action="store_true",
                   help="Skip games already recorded as ok in episode_results.jsonl")

    # agent
    p.add_argument("--agent_type", default="jitrl", choices=["jitrl", "naive"])
    p.add_argument("--arm", default=None,
                   help="Label for the output tree. Derived from the agent config if omitted.")
    p.add_argument("--top_actions", type=int, default=3)
    p.add_argument("--llm_temperature", type=float, default=0.7)
    p.add_argument("--confidence_mode", default="verbalized", choices=["verbalized", "logit"])
    p.add_argument("--update_rule", default="prob", choices=["prob", "logprob"])
    p.add_argument("--action_abstraction", default="goal_slot", choices=["instance", "goal_slot"])
    p.add_argument("--use_valid_actions", type=lambda s: s.lower() != "false", default=True)

    # models / endpoints
    p.add_argument("--llm_model", default="qwen3-32b")
    p.add_argument("--llm_base_url", default="http://localhost:8000/v1")
    p.add_argument("--eval_llm_model", default="gpt-oss-20b")
    p.add_argument("--eval_base_url", default="http://localhost:8001/v1")
    p.add_argument("--eval_temperature", type=float, default=0.3)
    p.add_argument("--eval_reasoning_effort", default="medium")

    # memory / RL
    p.add_argument("--enable_cross_mem", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--retrieval_top_k", type=int, default=10)
    p.add_argument("--retrieval_threshold", type=float, default=0.5)
    p.add_argument("--exploration_rate", type=float, default=0.65)
    p.add_argument("--exploration_alpha", type=float, default=1.0)
    p.add_argument("--failure_penalty", type=float, default=-10.0)
    p.add_argument("--max_memory", type=int, default=30)

    # Prompt evolution stays off: the upstream path hardcodes an Anthropic model
    # (Jericho/src/utils.py:986-997), which cannot run against a local server.
    p.add_argument("--update_guiding_prompt", action="store_true", default=False)
    p.add_argument("--use_history_prompt", action=argparse.BooleanOptionalAction, default=True)

    return p


def default_arm(args):
    if args.agent_type == "naive":
        return "naive"
    if not args.enable_cross_mem:
        return "jitrl_nomem"
    return f"jitrl_{args.confidence_mode}_{args.update_rule}"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = build_parser().parse_args()
    if not args.arm:
        args.arm = default_arm(args)

    here = os.path.dirname(os.path.abspath(__file__))
    for attr in ("alfworld_config", "game_index", "output_path"):
        val = getattr(args, attr)
        if not os.path.isabs(val):
            setattr(args, attr, os.path.join(here, val))

    if not os.environ.get("ALFWORLD_DATA"):
        print("ERROR: ALFWORLD_DATA is not set", file=sys.stderr)
        return 1

    config = load_config(args.alfworld_config)
    evaluator = AlfWorldEvaluator(args, config)
    summary = evaluator.run_evaluation()

    if summary:
        print(f"\n=== DONE: {args.room_type} / {args.arm} ===")
        print(f"success rate: {summary['success_rate']:.1%} over {summary['n_episodes']} episodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

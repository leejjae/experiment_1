"""One-off: enumerate the ALFWorld eval split and write games_unseen.json.

Run once after `alfworld-generate` has produced the game.tw-pddl files:

    ALFWORLD_DATA=~/.cache/alfworld \
      python build_game_index.py --config configs/base_config.yaml

The output is committed so every runner process agrees on the episode order.
"""

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.game_index import build_index, counts_by_room, counts_by_task  # noqa: E402


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--split", default="eval_out_of_distribution",
                        choices=["train", "eval_in_distribution", "eval_out_of_distribution"])
    parser.add_argument("--out", default="games_unseen.json")
    args = parser.parse_args()

    config = load_config(args.config)

    # Import here so a config error surfaces before the expensive directory walk.
    from alfworld.agents.environment import get_environment

    env = get_environment(config["env"]["type"])(config, train_eval=args.split)
    records = build_index(env.game_files)

    by_room = counts_by_room(records)
    by_task = counts_by_task(records)

    payload = {
        "split": args.split,
        "n_games": len(records),
        "counts_by_room": dict(by_room),
        "counts_by_task": dict(by_task),
        "games": records,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote {args.out}: {len(records)} games")
    print("  by room:", dict(by_room))
    print("  by task:", dict(by_task))

    if sum(by_room.values()) != len(records):
        print("WARNING: room buckets do not sum to the game count", file=sys.stderr)
        return 1
    missing = [r for r in ("kitchen", "living_room", "bedroom", "bathroom") if not by_room.get(r)]
    if missing:
        print(f"WARNING: empty room buckets: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

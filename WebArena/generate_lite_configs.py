#!/usr/bin/env python
"""Write config_files_lite/<task_id>.json for the WebArena-Lite set.

run.py reads a task's config from --config_dir to learn its sites and to run the
config-based evaluator; the environment itself gets the same configs through
webarena_lite_tasks.py. Both must come from the lite raw file, not the original
812-task one, because the lite set revises reference answers.

Usage (with env_setup.txt loaded so WA_* are set):
    python generate_lite_configs.py
    python generate_lite_configs.py --include-all-sites   # keep map/wikipedia too
"""

import argparse
import collections
import json
import os
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "config_files_generation" / "test_webarena_lite.raw.json"

# placeholder -> environment variable holding the live URL
URL_ENV = {
    "__GITLAB__": "WA_GITLAB",
    "__REDDIT__": "WA_REDDIT",
    "__SHOPPING__": "WA_SHOPPING",
    "__SHOPPING_ADMIN__": "WA_SHOPPING_ADMIN",
    "__WIKIPEDIA__": "WA_WIKIPEDIA",
    "__MAP__": "WA_MAP",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE / "config_files_lite"))
    ap.add_argument("--exclude-sites", default="map,wikipedia",
                    help="comma-separated sites to drop (we do not self-host these)")
    ap.add_argument("--include-all-sites", action="store_true")
    args = ap.parse_args()

    raw = RAW.read_text()
    missing = [p for p, e in URL_ENV.items() if not os.environ.get(e)]
    if missing:
        raise SystemExit(
            "These URL environment variables are unset: "
            + ", ".join(URL_ENV[p] for p in missing)
            + "\nLoad env_setup.txt first:  set -a && . ./env_setup.txt && set +a"
        )
    for placeholder, env_var in URL_ENV.items():
        raw = raw.replace(placeholder, os.environ[env_var])

    configs = json.loads(raw)
    excluded = set() if args.include_all_sites else {
        s.strip() for s in args.exclude_sites.split(",") if s.strip()
    }
    kept = [c for c in configs if not (set(c["sites"]) & excluded)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for conf in kept:
        with open(out_dir / f"{conf['task_id']}.json", "w") as f:
            json.dump(conf, f, indent=2)
    with open(out_dir / "test.json", "w") as f:
        json.dump(kept, f, indent=2)

    print(f"wrote {len(kept)} task configs to {out_dir}"
          + (f" (dropped {len(configs) - len(kept)} touching {sorted(excluded)})" if excluded else ""))
    for site, n in collections.Counter(c["sites"][0] for c in kept).most_common():
        print(f"  {site:16s} {n}")


if __name__ == "__main__":
    main()

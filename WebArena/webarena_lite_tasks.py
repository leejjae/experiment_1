"""Register the WebArena-Lite (165-task) evaluation set as BrowserGym tasks.

Why this module is needed: `browsergym.webarena` registers `webarena.<id>` against the
`webarena` package's own `test.raw.json`, i.e. the original 812-task benchmark. The
WebArena-Lite set renumbers its tasks 0..164 and, more importantly, *revises* them --
34 of the 134 non-map tasks carry corrected reference answers, and 3 carry a rewritten
intent. Running `webarena.<old_task_id>` would therefore grade a quarter of the suite
against ground truth that WebArena-Lite already superseded.

So we register a parallel `webarena_lite.<id>` task set that reads the lite configs and
otherwise reuses GenericWebArenaTask verbatim (setup, evaluation, teardown all inherited).

    import webarena_lite_tasks  # registers the gym ids as a side effect
    EnvArgs(task_name="walite.42", ...)

The prefix is "walite", not "webarena_lite", on purpose: browsergym's _get_env_name
eagerly imports browsergym.webarenalite and browsergym.webarena_verified for ANY task
name starting with "webarena", and neither is installed. Our tasks are registered by
importing this module, so we want that lazy-import branch skipped entirely.

Source of the configs: THUDM/VisualAgentBench, VAB-WebArena-Lite/new/test_webarena_lite.raw.json
"""

import json
import os
from pathlib import Path
from typing import Optional

from browsergym.core.registration import register_task
from browsergym.webarena.task import GenericWebArenaTask

LITE_CONFIG_PATH = Path(__file__).parent / "config_files_generation" / "test_webarena_lite.raw.json"

# See the module docstring: must not start with "webarena".
GYM_PREFIX = "walite"

URL_PATTERNS = {
    "__GITLAB__": "gitlab",
    "__REDDIT__": "reddit",
    "__SHOPPING__": "shopping",
    "__SHOPPING_ADMIN__": "shopping_admin",
    "__WIKIPEDIA__": "wikipedia",
    "__MAP__": "map",
}


def load_lite_configs(exclude_sites=()):
    """Raw (un-substituted) lite task configs, optionally dropping tasks that touch a site."""
    with open(LITE_CONFIG_PATH) as f:
        configs = json.load(f)
    if exclude_sites:
        excluded = set(exclude_sites)
        configs = [c for c in configs if not (set(c["sites"]) & excluded)]
    return configs


class WebArenaLiteTask(GenericWebArenaTask):
    """A WebArena task whose config comes from the WebArena-Lite set."""

    def __init__(self, seed: int, task_id: int, **kwargs):
        # Skip GenericWebArenaTask.__init__ (it hardcodes the 812-task config source) and
        # reproduce it here, differing only in where the configs are read from.
        super(GenericWebArenaTask, self).__init__(seed)

        from browsergym.webarena.instance import WebArenaInstance

        self.viewport = {"width": 1280, "height": 720}
        self.slow_mo = 1000  # ms
        self.timeout = 10000  # ms

        self.webarena_instance = WebArenaInstance()
        self.config_file: str = None
        self.with_na_hint = kwargs.get("with_na_hint", False)
        self.with_homepage_hint = kwargs.get("with_homepage_hint", False)

        all_configs_str = LITE_CONFIG_PATH.read_text()
        for pattern, url_key in URL_PATTERNS.items():
            all_configs_str = all_configs_str.replace(pattern, self.webarena_instance.urls[url_key])

        all_configs = json.loads(all_configs_str)
        task_configs = [conf for conf in all_configs if conf["task_id"] == task_id]
        if not task_configs:
            raise ValueError(
                f"Could not find any WebArena-Lite task config with task_id={task_id}."
            )

        self.task_id = task_id
        self.task_configs = task_configs


def register_all(exclude_sites=()):
    """Register walite.<id> for every lite task, returning the gym ids."""
    gym_ids = []
    for conf in load_lite_configs(exclude_sites=exclude_sites):
        gym_id = f"{GYM_PREFIX}.{conf['task_id']}"
        register_task(gym_id, WebArenaLiteTask, task_kwargs={"task_id": conf["task_id"]})
        gym_ids.append(gym_id)
    return gym_ids


# Sites we are not self-hosting are dropped at registration time so a typo in a task list
# surfaces as "unknown task" instead of a run that hangs against a dead URL.
# WEBARENA_LITE_EXCLUDE_SITES="" registers all 165.
_exclude = os.environ.get("WEBARENA_LITE_EXCLUDE_SITES", "map,wikipedia")
EXCLUDED_SITES = tuple(s.strip() for s in _exclude.split(",") if s.strip())

ALL_WEBARENA_LITE_TASK_IDS = register_all(exclude_sites=EXCLUDED_SITES)

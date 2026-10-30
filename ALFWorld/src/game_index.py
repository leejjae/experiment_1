"""Enumerate ALFWorld games and bucket them by room type.

ALFWorld's 120 rooms come from AI2-THOR floor plans, which are numbered by room
category. Every game lives at

    $ALFWORLD_DATA/json_2.1.1/<split>/<task_dir>/<trial_id>/game.tw-pddl

where ``<task_dir>`` looks like ``pick_and_place_simple-Book-None-Bed-308``: the
segment after the last hyphen is the AI2-THOR scene number.

We derive the scene number twice -- from the directory name and from
``traj_data.json`` -- and assert they agree, so a silently mis-bucketed game
fails loudly instead of quietly polluting a room-type stream.
"""

import json
import os
from collections import Counter

# AI2-THOR floor-plan numbering.
SCENE_RANGES = [
    (1, 30, "kitchen"),
    (201, 230, "living_room"),
    (301, 330, "bedroom"),
    (401, 430, "bathroom"),
]

ROOM_TYPES = [r for _, _, r in SCENE_RANGES]

# Which preprocessing step the goal requires, keyed by ALFWorld task type.
TREATMENT_BY_TASK = {
    "pick_and_place_simple": None,
    "pick_two_obj_and_place": None,
    "look_at_obj_in_light": "examine",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
}


class SceneMismatch(Exception):
    """Directory name and traj_data disagree about the scene number."""


def room_type_for_scene(scene_num):
    for lo, hi, room in SCENE_RANGES:
        if lo <= scene_num <= hi:
            return room
    raise ValueError(f"scene_num {scene_num} falls outside the known AI2-THOR ranges")


def _scene_from_traj(traj_data):
    scene = traj_data.get("scene", {})
    if "scene_num" in scene:
        return int(scene["scene_num"])
    # Older dumps only carry the floor plan string.
    floor_plan = scene.get("floor_plan", "")
    if floor_plan.startswith("FloorPlan"):
        return int(floor_plan[len("FloorPlan"):])
    raise KeyError("traj_data['scene'] has neither scene_num nor floor_plan")


def _scene_from_dirname(task_dir):
    return int(task_dir.rsplit("-", 1)[1])


def describe_game(gamefile):
    """Build the index record for one game.tw-pddl path.

    Raises SceneMismatch if the two independent scene-number derivations differ.
    """
    trial_dir = os.path.dirname(gamefile)
    task_dir = os.path.basename(os.path.dirname(trial_dir))

    with open(os.path.join(trial_dir, "traj_data.json")) as f:
        traj_data = json.load(f)

    scene = _scene_from_traj(traj_data)
    from_dir = _scene_from_dirname(task_dir)
    if scene != from_dir:
        raise SceneMismatch(f"{gamefile}: dirname says {from_dir}, traj_data says {scene}")

    task_type = traj_data["task_type"]
    pddl = traj_data["pddl_params"]

    return {
        "gamefile": gamefile,
        "game_id": os.path.join(task_dir, os.path.basename(trial_dir)),
        "scene_num": scene,
        "room_type": room_type_for_scene(scene),
        "task_type": task_type,
        "treatment": TREATMENT_BY_TASK.get(task_type),
        "pddl_params": pddl,
    }


def build_index(game_files):
    """Describe every game file, sorted for a deterministic stream order."""
    return [describe_game(g) for g in sorted(game_files)]


def group_by_room(records):
    grouped = {room: [] for room in ROOM_TYPES}
    for rec in records:
        grouped[rec["room_type"]].append(rec)
    return grouped


def counts_by_room(records):
    return Counter(rec["room_type"] for rec in records)


def counts_by_task(records):
    return Counter(rec["task_type"] for rec in records)


def load_index(path):
    with open(path) as f:
        return json.load(f)


def stream_for_room(index, room_type):
    """The ordered episode stream for one room type."""
    return [rec for rec in index["games"] if rec["room_type"] == room_type]

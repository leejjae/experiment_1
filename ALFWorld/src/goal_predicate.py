"""P1: task instruction -> goal predicate, by regex only (no LLM).

ALFWorld renders the goal sentence from a closed template set whenever
``env.goal_desc_human_anns_prob`` is 0.0, which configs/base_config.yaml pins.
A census over all 4027 game.tw-pddl files (train + valid_seen + valid_unseen)
found exactly twelve surface forms, all matched below.

This is deliberately *not* GoalSpec.from_record: that one reads
``traj_data['pddl_params']``, i.e. privileged ground truth. Anything claiming
"the goal was derived from the instruction" must go through here, and
from_record is then only admissible as the oracle to score this against.

Raise ``goal_desc_human_anns_prob`` above 0 and these templates stop covering
the data -- the turk annotations read like "Heating up a cup of coffee", which
names neither target nor destination. That regime needs an LLM (the P2 path).
"""

import re

# treatment -> the receptacle that performs it, mirroring
# state_construction.TREATMENT_TOOL.
TREATMENT_TOOL = {
    "clean": "sinkbasin",
    "heat": "microwave",
    "cool": "fridge",
    "examine": "desklamp",
}

# (pattern, treatment, n_required). Ordered longest-first so that
# "put a clean X in Y" is tried before "put a X in Y".
TEMPLATES = [
    (r"put a clean (\w+) in (\w+)",              "clean",   1),
    (r"clean some (\w+) and put it in (\w+)",    "clean",   1),
    (r"put a hot (\w+) in (\w+)",                "heat",    1),
    (r"heat some (\w+) and put it in (\w+)",     "heat",    1),
    (r"put a cool (\w+) in (\w+)",               "cool",    1),
    (r"cool some (\w+) and put it in (\w+)",     "cool",    1),
    (r"look at (\w+) under the (\w+)",           "examine", 1),
    (r"examine the (\w+) with the (\w+)",        "examine", 1),
    (r"find two (\w+) and put them in (\w+)",    None,      2),
    (r"put two (\w+) in (\w+)",                  None,      2),
    (r"put some (\w+) on (\w+)",                 None,      1),
    (r"put a (\w+) in (\w+)",                    None,      1),
]

_TASK_LINE = re.compile(r"your task is to:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


class GoalPredicate:
    """The goal as a conjunction of predicates over roles.

    ``conjuncts`` holds only what the PDDL ``(:goal ...)`` actually requires.
    ``holding`` is *not* among them -- it is a subgoal you necessarily pass
    through and then leave, so mixing it into the goal vector would break
    monotonicity.
    """

    def __init__(self, target, dest, treatment, n_required):
        self.target = target
        self.dest = dest
        self.treatment = treatment
        self.n_required = n_required

    @property
    def tool(self):
        return TREATMENT_TOOL.get(self.treatment)

    @property
    def conjuncts(self):
        c = []
        if self.treatment:
            c.append({"clean": "isClean", "heat": "isHot",
                      "cool": "isCool", "examine": "isExamined"}[self.treatment])
        c.append("inReceptacle" if self.n_required == 1 else f"inReceptacle x{self.n_required}")
        return c

    def as_dict(self):
        return {"target": self.target, "dest": self.dest,
                "treatment": self.treatment, "n_required": self.n_required}

    def __eq__(self, other):
        return isinstance(other, GoalPredicate) and self.as_dict() == other.as_dict()

    def __repr__(self):
        return (f"GoalPredicate(target={self.target!r}, dest={self.dest!r}, "
                f"treatment={self.treatment!r}, n={self.n_required})")


def extract_instruction(observation):
    """Pull the `Your task is to: ...` line out of a reset observation."""
    m = _TASK_LINE.search(observation or "")
    return m.group(1).strip() if m else ""


def strip_instruction(observation):
    """Remove the task sentence from an observation.

    Load-bearing for any downstream text matching: the reset observation
    contains the goal sentence, so a naive `if target in obs` test fires on the
    instruction itself and reports the target as *seen* at step 0.
    """
    return _TASK_LINE.sub("", observation or "").strip()


def parse_instruction(instruction):
    """Instruction text -> GoalPredicate, or None if no template matches."""
    s = (instruction or "").strip().rstrip(".").lower()
    for pattern, treatment, n in TEMPLATES:
        m = re.fullmatch(pattern, s)
        if m:
            return GoalPredicate(m.group(1), m.group(2), treatment, n)
    return None


def parse_observation(observation):
    """Convenience: reset observation -> GoalPredicate."""
    return parse_instruction(extract_instruction(observation))


def from_record(record):
    """P0 oracle: the same predicate read out of traj_data['pddl_params'].

    Only for scoring parse_instruction -- never as the agent's own goal source.
    """
    pddl = record.get("pddl_params", {}) or {}
    task_type = record.get("task_type")
    treatment = record.get("treatment")
    # look_at_obj_in_light has no parent_target -- nothing is ever put down.
    # The lamp lives in toggle_target, and that is what the instruction names,
    # so read it into the same slot or P1 disagrees on every examine task.
    dest = pddl.get("parent_target") or ""
    if treatment == "examine":
        dest = pddl.get("toggle_target") or dest
    return GoalPredicate(
        target=(pddl.get("object_target") or "").lower(),
        dest=dest.lower(),
        treatment=treatment,
        n_required=2 if task_type == "pick_two_obj_and_place" else 1,
    )

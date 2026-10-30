"""Goal-relative role labelling: raw observation -> abstract graph node.

The idea is to name every entity by the role it plays *in this goal* rather than
by its instance id, so that two episodes with different rooms, different target
objects and different instruction templates land on the same node sequence.
Roles are derived from the GoalPredicate, so nothing here is scene-specific:

    <target>            the goal object type
    <dest>              the goal receptacle type
    <tool>              the receptacle that performs the treatment
    <target_loc>        wherever the target was first observed
    unsearched(TYPE)    a receptacle of TYPE whose contents we have not seen
    searched(TYPE)      one we have

Two versions are kept on purpose. V0 reproduces the first prototype including
its two defects; V1 fixes them. Measuring both is how the cost of those defects
becomes a number instead of an opinion -- see scripts/measure_purity.py.

    V0 defect 1: the reset observation carries the task sentence, so
                 `target in obs` matched the *instruction* and KNOW latched to 1
                 at step 0 in every episode, making the field a constant.
    V0 defect 2: `holding` was packed into the goal bit-vector, so the final
                 place action read 110 -> 011 -- the vector is not monotone and
                 cannot be used to tell progress from regression.
"""

import re

from .goal_predicate import strip_instruction

# `a cabinet 4` / `a countertop 1` -- ALFWorld's uniform surface form for both
# receptacles and objects.
_ENTITY = re.compile(r"\ba ([a-z]+) (\d+)")
_GOTO = re.compile(r"go to ([a-z]+) (\d+)")

SEARCH_BUCKETS = ((0.0, "0"), (0.34, "low"), (0.67, "mid"), (1.01, "high"))


def _bucket(frac):
    for hi, name in SEARCH_BUCKETS:
        if frac <= hi:
            return name
    return "high"


def normalize_action(action, goal):
    """The label used to score every abstraction candidate. Held fixed.

    Instance ids are dropped and goal roles substituted, so `go to cabinet 3`
    and `go to cabinet 7` count as the same decision while `go to sinkbasin 1`
    (the tool for a clean task) counts as a different one.

    Substitution order is target, dest, tool: for `cool ... place in fridge`
    the destination and the tool are the same type, and calling it <dest> is
    the more informative of the two.
    """
    a = (action or "").strip().lower()
    if not a:
        return ""
    for pat, role in ((goal.target, "<target>"), (goal.dest, "<dest>"), (goal.tool, "<tool>")):
        if pat:
            a = re.sub(rf"\b{re.escape(pat)}\s+\d+\b", role, a)
    return re.sub(r"\b([a-z]+)\s+\d+\b", r"\1", a)


class _Base:
    def __init__(self, goal):
        self.goal = goal
        self.tool = goal.tool
        self.at = "room"
        self.at_type = "room"
        self.searched = set()      # receptacle instances whose contents we saw
        self.all_recep = set()     # every receptacle instance named at reset
        self.target_loc = None
        self.carrying = False
        self.treated = False
        self.n_placed = 0

    def role(self, rtype, instance=None):
        if rtype == "room":
            return "room"
        if rtype == self.goal.dest:
            return "<dest>"
        if self.tool and rtype == self.tool:
            return "<tool>"
        if instance and instance == self.target_loc:
            return "<target_loc>"
        if instance and instance not in self.searched:
            return f"unsearched({rtype})"
        return f"searched({rtype})"

    def search_frac(self):
        if not self.all_recep:
            return "0"
        return _bucket(len(self.searched & self.all_recep) / len(self.all_recep))

    def edge(self, action):
        """Role-normalized action, evaluated against the *pre-action* node."""
        m = _GOTO.match((action or "").strip().lower())
        if m:
            return f"go to {self.role(m.group(1), f'{m.group(1)} {m.group(2)}')}"
        return normalize_action(action, self.goal)


class RoleNodeV0(_Base):
    """First prototype, defects preserved. Not for use -- for measurement only."""

    def update(self, action, obs, inv=None):
        low = (obs or "").lower()
        if "middle of a room" in low:
            for m in _ENTITY.finditer(low):
                self.all_recep.add(m.group(1))          # types, not instances
        m = _GOTO.match((action or "").strip().lower())
        if m:
            self.at_type, self.at = m.group(1), f"{m.group(1)} {m.group(2)}"
            if "you see" in low:
                self.searched.add(self.at)
        # defect 1: `low` still contains the task sentence at reset.
        if self.goal.target in low and "you see" in low and not self.carrying:
            self.target_loc = self.at
        act = (action or "").lower()
        if re.match(rf"take {self.goal.target} ", act) and "you pick up" in low:
            self.carrying = True
        if re.match(rf"(clean|heat|cool) {self.goal.target} ", act):
            self.treated = True
        if "you turn on" in low and self.carrying:
            self.treated = True
        if re.match(rf"(move|put) {self.goal.target} ", act) and ("you move" in low or "you put" in low):
            self.n_placed += 1
            self.carrying = False

    def search_frac(self):
        # defect: compares instance names against a set of types, so the
        # intersection is always empty -- kept as-is deliberately.
        if not self.all_recep:
            return "0"
        return _bucket(len(self.searched) / len(self.all_recep))

    def key(self):
        # defect 2: holding sits in the goal vector.
        bits = f"{int(self.carrying)}{int(self.treated)}{int(self.n_placed >= self.goal.n_required)}"
        return (f"G={bits} AT={self.role(self.at_type, self.at)} "
                f"CARRY={'<target>' if self.carrying else 'nothing'} "
                f"KNOW={int(self.target_loc is not None)} SEARCH={self.search_frac()}")


class RoleNodeV1(_Base):
    """Task sentence stripped; goal conjuncts and the holding subgoal separated."""

    def update(self, action, obs, inv=None):
        raw = obs or ""
        low = strip_instruction(raw).lower()        # fix 1

        if "middle of a room" in low:
            for m in _ENTITY.finditer(low):
                self.all_recep.add(f"{m.group(1)} {m.group(2)}")   # instances

        act = (action or "").strip().lower()
        m = _GOTO.match(act)
        if m:
            self.at_type, self.at = m.group(1), f"{m.group(1)} {m.group(2)}"
            if "you see" in low:
                self.searched.add(self.at)

        # Target sighted: require the entity form `a <target> <id>` in a
        # listing, not a bare substring anywhere in the text.
        if self.target_loc is None and "you see" in low:
            if re.search(rf"\ba {re.escape(self.goal.target)} \d+", low):
                self.target_loc = self.at

        if re.match(rf"take {re.escape(self.goal.target)} ", act) and "you pick up" in low:
            self.carrying = True
        if re.match(rf"(clean|heat|cool) {re.escape(self.goal.target)} ", act) and "you " in low:
            self.treated = True
        if self.goal.treatment == "examine" and "you turn on" in low and self.carrying:
            self.treated = True
        if re.match(rf"(move|put) {re.escape(self.goal.target)} ", act) and ("you move" in low or "you put" in low):
            self.n_placed += 1
            self.carrying = False

    def key(self):
        # fix 2: GOAL holds only the PDDL conjuncts and is monotone; the
        # holding subgoal gets its own axis.
        goal_bits = "".join([
            str(int(self.treated)) if self.goal.treatment else "",
            str(int(self.n_placed >= self.goal.n_required)),
        ])
        return (f"GOAL={goal_bits} HOLD={int(self.carrying)} "
                f"AT={self.role(self.at_type, self.at)} "
                f"KNOW={int(self.target_loc is not None)} "
                f"SEARCH={self.search_frac()}")


VERSIONS = {"role_v0": RoleNodeV0, "role_v1": RoleNodeV1}

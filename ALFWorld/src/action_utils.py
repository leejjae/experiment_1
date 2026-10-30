"""Action normalization for ALFWorld, patterned on WebArena's normalize/denormalize pair.

JitRL aggregates returns by *raw string equality* on actions
(jitrl_agent.py:262-283, :312, :366). ALFWorld commands carry instance ids --
`go to drawer 3`, `put mug 1 in/on desk 1` -- so raw keys never repeat across
episodes. Every Q-bucket ends up with n=1, and
`calculate_exploration_probability` skips actions with fewer than 2 samples
(jitrl_agent.py:167-169), leaving `exploration_prob` pinned at 0.8 forever. At
that point 80% of unseen candidates get a random optimism bonus, so the
advantage term becomes noise rather than signal -- worse than having no memory.

Two abstraction levels:

  instance   strip instance ids only. Faithful to the original algorithm, but
             Q-values only transfer between episodes whose goal object matches.
  goal_slot  additionally replace the goal's target object and destination
             receptacle with <target>/<dest>. Default: each room-type stream is
             ~33 episodes of *different* tasks, so without slot abstraction the
             memory learns "take apple from countertop" and can never apply it
             to "take mug from countertop".
"""

import re

_ID = re.compile(r"\s+\d+(?=\b)")


def strip_ids(action):
    return _ID.sub("", (action or "").strip().lower())


def normalize_action(action, goal=None, mode="goal_slot"):
    """`put mug 1 in/on desk 1` -> `put <target> in/on <dest>` (goal_slot mode)."""
    n = strip_ids(action)
    if mode == "goal_slot" and goal is not None:
        # Longest first so a target that is a substring of the dest is not
        # clobbered by the shorter replacement.
        subs = []
        if getattr(goal, "target", ""):
            subs.append((goal.target, "<target>"))
        if getattr(goal, "dest", ""):
            subs.append((goal.dest, "<dest>"))
        for name, slot in sorted(subs, key=lambda kv: -len(kv[0])):
            n = re.sub(rf"\b{re.escape(name)}\b", slot, n)
    return n


def denormalize_action(normalized, admissible, goal=None, mode="goal_slot", tried=()):
    """Re-ground a normalized key to a currently-admissible command.

    Returns None when the abstract action has no instantiation here -- the
    caller should drop the candidate rather than substitute something else.
    """
    buckets = {}
    for cmd in admissible or ():
        buckets.setdefault(normalize_action(cmd, goal, mode), []).append(cmd)
    cands = buckets.get(normalized)
    if not cands:
        return None
    for c in cands:
        if c not in tried:
            return c
    return cands[0]


def repair_action(action, admissible, goal=None, mode="goal_slot", fallback_ranked=()):
    """Coerce a proposed action into the admissible set.

    Order: exact match -> re-ground via normalization -> highest-ranked
    admissible fallback -> first admissible command. Never returns `look`,
    which AlfWorldEnv filters out of `valid` anyway.

    Returns (action, repaired: bool).
    """
    admissible = list(admissible or ())
    if not admissible:
        return action, False
    if action in admissible:
        return action, False

    grounded = denormalize_action(normalize_action(action, goal, mode), admissible, goal, mode)
    if grounded:
        return grounded, True

    for cand in fallback_ranked:
        if cand in admissible:
            return cand, True

    return admissible[0], True

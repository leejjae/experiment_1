"""State Construction for ALFWorld -- the port of the paper's Section 4.1.

The paper's principle is that "functionally equivalent states" must map to
similar representations so retrieval can find them; raw observations are too
noisy. Jericho realises this with progress milestones and a pruned location
trajectory, both keyed on *score increases*.

ALFWorld has no dense score -- reward is +1 exactly once, at success -- so the
Jericho rules degenerate: [PROGRESS] would always read "No Progress" and the
whole location trajectory would be pruned as "movement without score increase".

We therefore replace "score increase" with "subgoal phase transition". Phases
are inferred with rules from the observation stream and the goal spec (which
comes from traj_data's pddl_params, not from parsing the goal sentence), so this
module makes zero LLM calls and is fully deterministic.

The three retrieval keys produced here are what CrossEpisodeMemory scores:

    goal_key     weight 0.25, 1-gram Jaccard
    history_key  weight 0.35, 1-gram Jaccard
    state_key    weight 0.40, 4-gram Jaccard

Instance ids are stripped and the goal's target/destination are replaced by
<target>/<dest> slots, so "carrying the mug at the countertop" in one episode and
"carrying the plate at the countertop" in another collapse to the same string.
"""

import re

PHASES = ["SEARCHING", "LOCATED", "HOLDING", "TREATED", "AT_DEST", "DONE"]

# Receptacle that performs each treatment, used to recognise completion.
TREATMENT_TOOL = {
    "clean": "sinkbasin",
    "heat": "microwave",
    "cool": "fridge",
    "examine": "desklamp",
}

_ID = re.compile(r"\s+\d+(?=\b)")
_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")


def strip_ids(text):
    """`countertop 1` -> `countertop`; also lowercases."""
    return _ID.sub("", (text or "").strip().lower())


def normalize_name(name):
    """ALFRED's CamelCase entity names -> the env's lowercase spelling.

    `SinkBasin` -> `sinkbasin`, `AlarmClock` -> `alarmclock`. ALFWorld's text
    surface form is simply the lowercased concatenation, so we do not insert
    separators.
    """
    return (name or "").strip().lower()


class GoalSpec:
    """The task goal, taken from traj_data['pddl_params'] -- never parsed from text."""

    def __init__(self, target, dest, mrecep=None, toggle=None,
                 treatment=None, n_required=1, task_type=None, goal_text=""):
        self.target = normalize_name(target)
        self.dest = normalize_name(dest)
        self.mrecep = normalize_name(mrecep)
        self.toggle = normalize_name(toggle)
        self.treatment = treatment
        self.n_required = n_required
        self.task_type = task_type
        self.goal_text = goal_text

    @classmethod
    def from_record(cls, rec):
        pddl = rec.get("pddl_params", {}) or {}
        task_type = rec.get("task_type")
        return cls(
            target=pddl.get("object_target", ""),
            dest=pddl.get("parent_target", ""),
            mrecep=pddl.get("mrecep_target", ""),
            toggle=pddl.get("toggle_target", ""),
            treatment=rec.get("treatment"),
            n_required=2 if task_type == "pick_two_obj_and_place" else 1,
            task_type=task_type,
        )

    @property
    def treatment_tool(self):
        return TREATMENT_TOOL.get(self.treatment)

    def __repr__(self):
        return (f"GoalSpec(target={self.target!r}, dest={self.dest!r}, "
                f"treatment={self.treatment!r}, n={self.n_required})")


class EpisodeState:
    """Cumulative tracker for one episode. `update()` runs exactly once per env step.

    The once-per-step discipline mirrors CrossEpisodeMemory.step_context_cache,
    which is indexed positionally by step; calling update() twice for one step
    would silently desynchronise the stored metadata.
    """

    def __init__(self, goal):
        self.goal = goal
        self.at = "room"
        self.visited = []          # first visits only, in order
        self.searched = set()      # receptacles whose contents we have observed
        self.visible = set()       # receptacles visible from the current spot
        self.seen_objects = set()  # object types observed anywhere so far
        self.carrying = None
        self.treated = False
        self.n_placed = 0
        self.milestones = []
        self.phase = "SEARCHING"
        self.steps = 0

    # ── observation parsing ────────────────────────────────────────────────
    def _parse_receptacles(self, text):
        """Receptacle types named in a `you see a cabinet 4, a cabinet 3, ...` list."""
        found = set()
        for m in re.finditer(r"\ba ([a-z]+) \d+", text):
            found.add(m.group(1))
        return found

    def _parse_objects(self, text):
        """Object types named in the text (same surface form as receptacles)."""
        return self._parse_receptacles(text)

    def _update_location(self, action, obs):
        if not action:
            return
        m = re.match(r"go to (.+)", action.strip().lower())
        if not m:
            return
        loc = strip_ids(m.group(1))
        self.at = loc
        if loc not in self.visited:
            self.visited.append(loc)

    def _update_carrying(self, inv):
        text = (inv or "").lower()
        if "not carrying" in text or "nothing" in text:
            self.carrying = None
            return
        m = re.search(r"carrying:\s*a ([a-z]+) \d+", text)
        self.carrying = m.group(1) if m else self.carrying

    def _update_treated(self, obs, action):
        tool = self.goal.treatment_tool
        if not tool or self.treated:
            return
        low = (obs or "").lower()
        if self.goal.treatment == "examine":
            # look_at_obj_in_light completes by turning the lamp on while holding
            # the target.
            if "turn on" in low or "you turn on" in low:
                if self.carrying == self.goal.target:
                    self.treated = True
            return
        verb = {"clean": "clean", "heat": "heat", "cool": "cool"}[self.goal.treatment]
        if re.search(rf"you {verb} the {re.escape(self.goal.target)}", low):
            self.treated = True

    def update(self, obs, action, inv, admissible, won):
        """Fold one env step into the tracker."""
        self.steps += 1
        obs = obs or ""
        low = obs.lower()

        self._update_location(action, obs)
        self._update_carrying(inv)

        if "you see" in low or "in it, you see" in low:
            self.searched.add(self.at)
            self.visible |= self._parse_receptacles(low)
            self.seen_objects |= self._parse_objects(low)
        if "you see nothing" in low or "is empty" in low:
            self.searched.add(self.at)

        self._update_treated(obs, action)

        if action and re.match(r"put .+ (in|on|in/on) .+", action.strip().lower()):
            if "you put" in low:
                self.n_placed += 1

        new_phase = self.infer_phase(won)
        if new_phase != self.phase:
            self.milestones.append(self._milestone(self.phase, new_phase))
            self.phase = new_phase
        return self.phase

    # ── phase inference ────────────────────────────────────────────────────
    def infer_phase(self, won):
        if won:
            return "DONE"
        holding = self.carrying is not None and self.carrying == self.goal.target
        needs_treatment = self.goal.treatment is not None
        if holding and (self.treated or not needs_treatment):
            if self.at == self.goal.dest:
                return "AT_DEST"
            return "TREATED"
        if holding:
            return "HOLDING"
        if self.goal.target in self.seen_objects:
            return "LOCATED"
        return "SEARCHING"

    def _milestone(self, old, new):
        verb = {
            "LOCATED": "locate",
            "HOLDING": "take",
            "TREATED": (self.goal.treatment or "ready"),
            "AT_DEST": "reach",
            "DONE": "place",
        }.get(new, new.lower())
        where = f"@{self.at}" if new in ("LOCATED", "AT_DEST") else ""
        return f"M{len(self.milestones) + 1}: {verb}-><target>{where}"

    # ── retrieval keys ─────────────────────────────────────────────────────
    def _slot(self, name):
        if not name:
            return "nothing"
        if name == self.goal.target:
            return "<target>"
        if name == self.goal.dest:
            return "<dest>"
        return name

    def _slot_set(self, names):
        return sorted(self._slot(n) for n in names)

    def state_key(self):
        return (
            f"[PHASE] {self.phase}\n"
            f"[AT] {self._slot(self.at)}\n"
            f"[VISIBLE] {', '.join(self._slot_set(self.visible))}\n"
            f"[CARRY] {self._slot(self.carrying)}"
        )

    def history_key(self):
        return (
            f"[MILESTONES] {'  '.join(self.milestones) if self.milestones else 'none'}\n"
            f"[VISITED] {'->'.join(self._slot(v) for v in self.visited) or 'none'}\n"
            f"[SEARCHED] {', '.join(self._slot_set(self.searched)) or 'none'}\n"
            f"[INVENTORY] {self._slot(self.carrying)}"
        )

    def goal_key(self):
        return (
            f"[TASK] {self.goal.treatment or 'none'} <target> -> <dest>\n"
            f"[TYPE] {self.goal.task_type}\n"
            f"[N] {self.goal.n_required}"
        )

    def location_key(self):
        """Discrete prefilter for retrieval -- the WebArena normalized_url analogue."""
        return self._slot(self.at)

    def prompt_block(self):
        """The header injected into the actor prompt."""
        return (
            f"[PHASE] {self.phase}\n"
            f"[AT] {self.at}\n"
            f"[SEARCHED] {', '.join(sorted(self.searched)) or 'none'}"
        )

"""LLM utilities for the ALFWorld port.

Only two functions from the 1150-line Jericho original survive, both with
rewritten prompts:

* `evaluate_step_scores_with_llm` -- the paper's Step-wise Reward Evaluation
  prompt (Appendix G.2.2). This is the ONLY dense signal in ALFWorld, whose env
  reward is +1 exactly once. The upstream prompt says "calibrate based on
  rewards shown in trajectory"; with an all-zero trajectory there is nothing to
  calibrate against, the model emits near-uniform scores, and the advantage term
  collapses. Replaced with an absolute subgoal rubric.

* `generate_history_summary` -- the History Summary prompt (Appendix G.2.4).
  Upstream defines a milestone as a *score increase* and prunes locations that
  moved "without score increase"; in ALFWorld that makes [PROGRESS] permanently
  "No Progress" and deletes the entire location trajectory. Redefined around
  subgoal phase transitions.

The State Summarization prompt (G.2.3) is deleted outright: it keys on a
`<<location>>` marker and exists to recover environment info when the current
state lacks it. We always supply info['look'], and state_construction.py builds
the retrieval keys deterministically -- saving one LLM call per env step.

The original file is kept alongside as utils_jericho_original.py.bak.
"""

from .openai_helpers import (
    TokenLimitExceededError,
    chat_completion_with_retries,
    extract_json_from_response,
    response_text,
)

MAX_OBS_CHARS = 400


def _truncate(text, n=MAX_OBS_CHARS):
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + " ..."


# ── G.2.2 Step-wise Reward Evaluation ──────────────────────────────────────
STEP_SCORE_SYS_PROMPT = """You are scoring the actions of an embodied household agent to build \
training data for future episodes.

Every ALFWorld task decomposes into:
  (1) NAVIGATE to a receptacle that plausibly holds the target object
  (2) OPEN it if openable and closed
  (3) TAKE the target object
  (4) TREAT it if required (clean with sinkbasin / heat with microwave /
      cool with fridge / examine under desklamp)
  (5) NAVIGATE to the destination receptacle
  (6) PUT the object there

Score each action on this ABSOLUTE scale. Do NOT calibrate to the rewards shown in the \
trajectory -- the environment reward is sparse and is 0 almost everywhere.

  +10  completed the final required subgoal (task success)
   +6  completed an intermediate subgoal (took the target, finished the treatment,
       reached the destination while carrying the target)
   +3  first visit to a receptacle that plausibly holds the target,
       or opening a closed receptacle that had not been checked
    0  neutral / purely informational, or reasonable but unproductive search
       of a new receptacle
   -3  revisited a receptacle already searched this episode with no new reason
   -6  a loop (A -> B -> A) or repeating an action that already failed
  -10  discarded or misplaced the target object, or burned many steps doing nothing

Use only these seven scores: -10, -6, -3, 0, +3, +6, +10.

The [PHASE] of each step is given alongside the trajectory. An action that causes a PHASE \
transition is by definition a subgoal completion and should score +6 or better.

Judge by the FULL consequence chain, not the immediate observation."""

STEP_SCORE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "step_analysis_scores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "step_analysis": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "integer"},
                            "action": {"type": "string"},
                            "detailed_reasoning": {"type": "string"},
                            # No minimum/maximum: xgrammar rejects numeric range
                            # constraints, and the silent backend fallback costs
                            # a per-request grammar compile. Clamped in Python.
                            "score": {"type": "number"},
                        },
                        "required": ["step", "action", "detailed_reasoning", "score"],
                        "additionalProperties": False,
                    },
                },
                "overall_assessment": {"type": "string"},
            },
            "required": ["step_analysis", "overall_assessment"],
            "additionalProperties": False,
        },
    },
}


def evaluate_step_scores_with_llm(game_history, state, final_score, success,
                                  goal_text="", llm_model="gpt-oss-20b",
                                  temperature=0.3, base_url=None,
                                  reasoning_effort="medium"):
    """Hindsight credit assignment over a finished trajectory.

    Returns (scores, reasonings), both aligned with game_history. Falls back to
    raw env rewards on failure -- which in ALFWorld means all zeros, so the
    caller must check for that (see the P6 gate).
    """
    if not game_history:
        return [], []

    fallback = ([e.get("reward", 0) or 0 for e in game_history],
                ["No reasoning available"] * len(game_history))

    lines = []
    for i, entry in enumerate(game_history):
        lines.append(f"Step {i}: [PHASE] {entry.get('phase', '?')}")
        lines.append(f"  State: {_truncate(entry.get('state', ''))}")
        if entry.get("action"):
            lines.append(f"  Action: {entry.get('action')}")
        lines.append(f"  Reward: {entry.get('reward', 0)}")
    lines.append(f"Step {len(game_history)}: State: {_truncate(state)}")
    trajectory_text = "\n".join(lines)

    user_prompt = f"""Score each action in this episode.

TASK GOAL: {goal_text}
Final result: {"SUCCESS" if success else "FAILURE"}

Trajectory:
{trajectory_text}

Return one entry per step (steps 0..{len(game_history) - 1}), with
`detailed_reasoning` as ONE sentence of at most 25 words."""

    try:
        res = chat_completion_with_retries(
            model=llm_model,
            sys_prompt=STEP_SCORE_SYS_PROMPT,
            prompt=user_prompt,
            max_tokens=6000,
            temperature=temperature,
            response_format=STEP_SCORE_SCHEMA,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            enable_thinking=None,      # evaluator keeps its native reasoning
        )
    except TokenLimitExceededError as e:
        print(f"[StepScore] context overflow: {e}")
        return fallback
    except Exception as e:
        print(f"[StepScore] call failed: {e}")
        return fallback

    parsed = extract_json_from_response(response_text(res))
    if not parsed or "step_analysis" not in parsed:
        print("[StepScore] unparseable response; falling back to env rewards")
        return fallback

    scores = [0] * len(game_history)
    reasonings = ["No reasoning available"] * len(game_history)
    for item in parsed["step_analysis"]:
        try:
            idx = int(item["step"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= idx < len(game_history):
            try:
                scores[idx] = max(-10.0, min(10.0, float(item.get("score", 0))))
            except (TypeError, ValueError):
                scores[idx] = 0
            reasonings[idx] = item.get("detailed_reasoning", "")
    return scores, reasonings


# ── G.2.4 History Summary ──────────────────────────────────────────────────
HISTORY_SYS_PROMPT = """You are analyzing the trajectory of an embodied household agent and \
producing a compact, distinctive summary.

Output format:

[SUMMARY]: A 2-3 sentence natural-language summary covering the task goal, what has been
   accomplished, and the current situation.

[PROGRESS]: List milestones in the format "M#: <verb>-><result>"
   - A milestone is a SUBGOAL PHASE TRANSITION, not a score change.
     Phases: SEARCHING -> LOCATED -> HOLDING -> TREATED -> AT_DEST -> DONE
   - If NO phase transition has occurred, output ONLY "No Progress"

[VISITED]: Receptacles visited, in order: "Visited: A->B->C"
   - Strip instance numbers (countertop 1 -> countertop)
   - REMOVE revisits: if a receptacle was already searched and visiting it again revealed
     nothing new, drop that visit
   - Example: "fridge->cabinet->fridge->cabinet->countertop" -> "fridge->cabinet->countertop"

[SEARCHED]: Comma-separated set of receptacles whose contents have been observed.

Examples:
BAD SUMMARY:  "The agent has been exploring the room."
GOOD SUMMARY: "The goal is to clean a mug and put it on the desk. The agent searched the
               fridge and cabinet, found the mug on the countertop, and is now carrying it
               toward the sinkbasin to clean it."

BAD:  "M1: Went to the countertop and picked up the mug"
GOOD: "M2: take-><target>"

BAD:  "Visited: fridge 1->cabinet 2->fridge 1->cabinet 2->countertop 1"
GOOD: "Visited: fridge->cabinet->countertop\""""


def generate_history_summary(game_history, current_state=None, llm_model="qwen3-32b",
                             temperature=0.0, max_tokens=1000, current_inventory=None,
                             goal_text="", info=None, base_url=None):
    """Returns (summary_text, structured_summary).

    structured_summary becomes `memory_text`, which is shown to the actor.
    Retrieval keys come from state_construction.py, not from this text.
    """
    if not game_history and not current_state:
        return "", ""

    lines = []
    for i, entry in enumerate(game_history):
        lines.append(f"Step {i} [PHASE {entry.get('phase', '?')}]: "
                     f"{_truncate(entry.get('state', ''), 250)}")
        if entry.get("action"):
            lines.append(f"  Action: {entry['action']}")
    if current_state:
        lines.append(f"Step {len(game_history)} (Current): {_truncate(current_state, 250)}")
    trajectory_text = "\n".join(lines)

    user_prompt = f"""TASK GOAL: {goal_text}

Trajectory:
{trajectory_text}

Produce the summary in the required format ([SUMMARY], [PROGRESS], [VISITED], [SEARCHED])."""

    try:
        text = response_text(chat_completion_with_retries(
            model=llm_model,
            sys_prompt=HISTORY_SYS_PROMPT,
            prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            base_url=base_url,
            enable_thinking=False,
        ))
    except Exception as e:
        print(f"[HistorySummary] call failed: {e}")
        text = ""

    if not text:
        structured = info.get("prompt_block", "") if info else ""
        if current_inventory:
            structured += f"\n[INVENTORY]\n{current_inventory}"
        return "", structured

    summary_text = ""
    if "[SUMMARY]" in text:
        start = text.find("[SUMMARY]") + len("[SUMMARY]")
        nxt = text.find("[PROGRESS]", start)
        if nxt == -1:
            nxt = text.find("[VISITED]", start)
        summary_text = (text[start:nxt] if nxt != -1 else text[start:]).strip()

    structured = text
    p = text.find("[PROGRESS]")
    if p != -1:
        structured = text[p:]
    if current_inventory:
        structured += f"\n\n[INVENTORY]\n{current_inventory}"
    return summary_text, structured

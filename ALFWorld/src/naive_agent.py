"""Memory-free baseline.

Rewritten from the Jericho version, whose prompt was text-adventure specific
("say X", "ask X about Y", "any such X or Y MUST be a *SINGLE WORD*"). Kept
verbatim it would be a strawman: ALFWorld commands *require* instance numbers,
so a single-word rule guarantees invalid actions.

For a fair comparison this agent gets exactly the information JitRL gets, minus
the cross-episode memory: the task goal, the subgoal phase, the searched-set,
the shuffled admissible-command list, the inventory, and the last ~10 steps of
*within-episode* history. Withholding within-episode history would confound
"memory" with "context" and inflate the JitRL delta.
"""

import random
import time

from .action_utils import repair_action
from .openai_helpers import chat_completion_with_retries, response_text

SYS_PROMPT = """You are an embodied household agent operating in a simulated room. \
You are given a task goal and must complete it by manipulating objects and receptacles.

Every task decomposes into these subgoals:
  (1) NAVIGATE to a receptacle that plausibly holds the target object
  (2) OPEN it if it is openable and closed
  (3) TAKE the target object
  (4) TREAT it if the goal requires (clean with sinkbasin / heat with microwave /
      cool with fridge / examine under desklamp)
  (5) NAVIGATE to the destination receptacle
  (6) PUT the object there

Key facts about this environment:
- You cannot see the contents of a receptacle until you `go to` it (and `open` it if openable).
- Object and receptacle names carry instance numbers (e.g. `mug 1`, `desk 2`). These are \
REQUIRED - never strip or alter them.
- Receptacles you have already searched are listed under [SEARCHED]. Do NOT search them \
again unless you have a concrete new reason.
- Your current subgoal stage is given as [PHASE]. Choose actions that advance it.

**CRITICAL CONSTRAINT**: When REFERENCE ACTIONS are provided, you MUST ONLY choose actions \
from that list. Any action not in the REFERENCE ACTIONS list is INVALID and will fail. \
Do NOT create custom actions. The list is unordered - position doesn't indicate quality."""


class NaiveAgent:
    def __init__(self, args, guiding_prompt: str = None):
        self.args = args
        self.guiding_prompt = guiding_prompt or ""
        self.game_history = []
        self.counters = {}

    # ── driver contract ────────────────────────────────────────────────────
    def start_episode(self):
        self.game_history = []
        self.counters = {
            "n_json_parse_failures": 0,
            "n_action_repairs": 0,
        }

    def end_episode(self, state, score, won=False):
        print(f"[naive] episode end: won={won} score={score}")

    def update_game_history_reward(self, reward, score):
        if self.game_history:
            self.game_history[-1]["reward"] = reward
            self.game_history[-1]["score"] = score

    def generate_action(self, state_node, info=None):
        sys_prompt, user_prompt = self.get_prompts(state_node, info)

        res = chat_completion_with_retries(
            model=self.args.llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=300,
            temperature=self.args.llm_temperature,
            base_url=self.args.llm_base_url,
            enable_thinking=False,
        )
        full_response = response_text(res)
        if not full_response:
            self.counters["n_json_parse_failures"] += 1
            action_text = None
        else:
            action_text = self._parse_llm_response(full_response)

        valid = (info or {}).get("valid", [])
        goal = (info or {}).get("goal")
        if action_text is None:
            action_text = valid[0] if valid else "look"
            repaired = True
        else:
            action_text, repaired = repair_action(
                action_text, valid, goal, mode=self.args.action_abstraction)
        if repaired:
            self.counters["n_action_repairs"] += 1
            print(f"[REPAIR] -> {action_text}")

        self.game_history.append({
            "state": state_node.state, "action": action_text,
            "reward": None, "score": None,
        })
        return action_text.strip(), full_response

    # ── prompting ──────────────────────────────────────────────────────────
    def get_prompts(self, state_node, info=None):
        info = info or {}
        sys_prompt = SYS_PROMPT
        if self.guiding_prompt:
            sys_prompt += f"\n\nFollow this guide: {self.guiding_prompt}"

        valid = list(info.get("valid", []))
        if valid:
            random.Random(time.time_ns()).shuffle(valid)

        recent = ""
        for i, entry in enumerate(self.game_history[-10:]):
            idx = max(0, len(self.game_history) - 10) + i
            recent += f"Step {idx}: {entry.get('action', '')}\n"

        user_prompt = f"""TASK GOAL: {state_node.goal_text}
{info.get('prompt_block', '')}

RECENT ACTIONS:
{recent or 'none'}

INVENTORY: {info.get('inv', 'unknown')}

CURRENT OBSERVATION:
{state_node.state}

REFERENCE ACTIONS (ONLY VALID ACTIONS):
{valid}

**STRICT REQUIREMENT**: These are the ONLY valid actions for the current state. You MUST \
select your action EXCLUSIVELY from this list, copied EXACTLY including its instance number.

Your response MUST strictly follow this format and include nothing else:
REASONING: [1-2 sentences on why this action advances the current [PHASE]]
ACTION: [one action copied exactly from REFERENCE ACTIONS]

For example:
REASONING: The mug is likely in a cabinet I have not searched yet.
ACTION: go to cabinet 3
"""
        return sys_prompt, user_prompt

    def _parse_llm_response(self, full_response):
        if not full_response or not isinstance(full_response, str):
            return None
        action_text = None
        for line in full_response.strip().split("\n"):
            if line.upper().strip().startswith("ACTION:"):
                action_text = line.split(":", 1)[1].strip()
        return action_text

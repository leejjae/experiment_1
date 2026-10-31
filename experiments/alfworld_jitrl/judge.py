"""Judge-only step scoring for JitRL memory (no environment outcome).

Replaces src.cross_episode_memory.evaluate_step_scores_with_llm with the version
the reported judge-only runs used: the same system prompt and seven-level rubric
as ALFWorld/src/utils.py, but the trajectory carries no reward and no
"Final result: SUCCESS/FAILURE" line. Incomplete responses are retried with
larger token budgets and a per-step formatting correction; there is no fallback
score. Every judge response is written to the audit directory.

Model calls (actor and judge) raise instead of degrading silently, so a failed
request stops the stream and the launcher restarts it.
"""
import copy
import json
import math
import re
import time

import src.cross_episode_memory as memory_module
import src.jitrl_agent as agent_module
import src.utils as utils

ALLOWED_SCORES = (-10, -6, -3, 0, 3, 6, 10)
OUTCOME_KEYS = {'reward', 'score', 'success', 'won', 'outcome', 'environment_reward'}


def decode_scores(content, expected):
    """Read complete step-score arrays without inventing truncated score entries."""
    try:
        items = json.loads(content)['step_analysis']
    except (ValueError, KeyError, TypeError):
        # Only the leading array can be decoded on its own; nothing is added to it.
        match = re.match(r'^\s*\{\s*"step_analysis"\s*:\s*', content)
        if not match:
            raise ValueError('No complete leading step_analysis array')
        items, _ = json.JSONDecoder().raw_decode(content, match.end())
    if not isinstance(items, list) or len(items) != expected:
        raise ValueError('Incomplete score array')
    if any(not isinstance(item, dict) or type(item.get('step')) is not int for item in items):
        raise ValueError('Invalid step index type')
    if sorted(item['step'] for item in items) != list(range(expected)):
        raise ValueError('Missing/duplicate step indices')
    for item in items:
        if type(item.get('score')) not in (int, float) or not math.isfinite(item['score']):
            raise ValueError('Invalid or non-finite step score')
        if not isinstance(item.get('detailed_reasoning'), str) or not isinstance(item.get('action'), str):
            raise ValueError('Incomplete step fields')
    return sorted(items, key=lambda item: item['step'])


def install(audit_dir):
    """Patch the model calls and the step scorer. Returns the list that collects call errors."""
    original = agent_module.chat_completion_with_retries
    errors = []

    def strict(**kwargs):
        kwargs.setdefault('timeout', 120)
        kwargs.setdefault('max_retries', 3)
        try:
            result = original(**kwargs)
            if not result or not getattr(result, 'choices', None):
                raise RuntimeError('Model API returned no choices after retries')
            return result
        except Exception as exc:
            errors.append(str(exc))
            raise

    agent_module.chat_completion_with_retries = strict
    utils.chat_completion_with_retries = strict

    def judge_scores(**kwargs):
        history = kwargs['game_history']
        assert not {'success', 'final_score', 'won', 'outcome'}.intersection(kwargs)
        assert all(not OUTCOME_KEYS.intersection(e) for e in history)
        lines = []
        for i, entry in enumerate(history):
            lines.append(f"Step {i}: [PHASE] {entry.get('phase', '?')}")
            lines.append(f"  State: {utils._truncate(entry.get('state', ''))}")
            if entry.get('action'):
                lines.append(f"  Action: {entry['action']}")
        lines.append(f"Step {len(history)}: State: {utils._truncate(kwargs['state'])}")
        trajectory = '\n'.join(lines)
        prompt = f"""Score each action in this episode.

TASK GOAL: {kwargs.get('goal_text', '')}


Trajectory:
{trajectory}

Return one entry per step (steps 0..{len(history) - 1}), with
`detailed_reasoning` as ONE sentence of at most 25 words."""
        for attempt, budget in enumerate((6000, 12000, 18000), 1):
            response_format = utils.STEP_SCORE_SCHEMA
            request_prompt = prompt
            if attempt > 1:
                # Repeated identical actions remain separate executed steps.
                response_format = copy.deepcopy(utils.STEP_SCORE_SCHEMA)
                array = response_format['json_schema']['schema']['properties']['step_analysis']
                array['minItems'] = array['maxItems'] = len(history)
                request_prompt += ('\nFormatting correction: Return exactly one row for EVERY indexed action below. '
                                   'Repeated identical actions are distinct executions; do not merge them. '
                                   'Keep each action at its original step index.\n' +
                                   '\n'.join(f"{i}: {entry.get('action', '')}" for i, entry in enumerate(history)))
            response = original(model=kwargs['llm_model'], sys_prompt=utils.STEP_SCORE_SYS_PROMPT,
                                prompt=request_prompt, max_tokens=budget, temperature=kwargs['temperature'],
                                response_format=response_format, base_url=kwargs['base_url'],
                                reasoning_effort=kwargs['reasoning_effort'], enable_thinking=None,
                                timeout=180, max_retries=3)
            content = utils.response_text(response)
            choice = response.choices[0] if getattr(response, 'choices', None) else None
            audit_dir.mkdir(parents=True, exist_ok=True)
            (audit_dir / f'judge_{time.time_ns()}.json').write_text(json.dumps(dict(
                attempt=attempt, max_tokens=budget, finish_reason=choice.finish_reason if choice else None,
                steps=len(history), prompt=request_prompt, content=content), ensure_ascii=False) + '\n')
            try:
                items = decode_scores(content, len(history))
                if attempt > 1 and any(item['action'].strip().casefold() != entry.get('action', '').strip().casefold()
                                       for item, entry in zip(items, history)):
                    raise ValueError('Retry action rows do not match the executed actions')
                if any(float(item['score']) not in ALLOWED_SCORES for item in items):
                    raise ValueError('Judge score outside the seven-level rubric')
                return [float(item['score']) for item in items], [item['detailed_reasoning'] for item in items]
            except (KeyError, TypeError, ValueError) as exc:
                print(f'[Judge] attempt {attempt} incomplete ({exc}); retrying with {budget} -> larger budget', flush=True)
        raise RuntimeError('Judge failed after 6000/12000/18000 token budgets; no fallback')

    memory_module.evaluate_step_scores_with_llm = judge_scores
    return errors

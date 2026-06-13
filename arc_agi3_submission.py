"""
ARC-AGI-3 Kaggle Submission
============================
Multi-Agent LLM Collaboration Framework
----------------------------------------
Architecture:
  - Explorer Agent    : issues actions, reports raw observations
  - Reasoner Agent    : builds / updates a world model from observations
  - Planner Agent     : produces an action plan given the world model
  - Critic Agent      : validates the plan; requests revision when uncertain
  - Summarizer        : compresses history to stay within context limits

Environment format (from ARC-AGI-3 paper):
  - 64x64 grid, 16 colours per cell
  - Action space: up to 5 key actions + Undo + coordinate click
  - Agent is NEVER told the objective or controls — must infer both
  - Scored by RHAE (Relative Human Action Efficiency): fewer actions = higher score

Kaggle constraints:
  - NO internet access during evaluation
  - Must run on the provided Kaggle hardware (RTX 6000 / g4-standard-48)
  - Must open-source (MIT / CC0) to be prize-eligible

NOTE: Replace the stub `call_local_model()` with your offline model inference.
      Kaggle disallows internet, so no external API calls are permitted during scoring.
"""

import os
import json
import time
import base64
import pathlib
import traceback
from typing import Any

# ── 1. ENVIRONMENT INTERFACE ──────────────────────────────────────────────────
# The ARC-AGI-3 Kaggle environment is accessed via the `arcagi3` module that
# Kaggle injects at runtime. We provide a thin wrapper and a local mock for
# offline development / debugging.

try:
    import arcagi3  # Kaggle runtime injects this
    KAGGLE_ENV = True
except ImportError:
    KAGGLE_ENV = False
    print("[WARN] arcagi3 not found — running in local mock mode.")


class EnvironmentWrapper:
    """Thin wrapper around the ARC-AGI-3 Kaggle environment API."""

    def __init__(self):
        if KAGGLE_ENV:
            self.env = arcagi3.make()
        else:
            self.env = _MockEnv()

    def reset(self, game_id: str):
        """Reset to the first level of a game and return the initial frame."""
        obs = self.env.reset(game_id)
        return self._parse_obs(obs)

    def step(self, action: str):
        """Take one action. Returns (frame, done, info)."""
        obs, done, info = self.env.step(action)
        return self._parse_obs(obs), done, info

    def _parse_obs(self, obs) -> dict:
        """Normalise observation to a dict with 'frame' (list[list[int]])."""
        if isinstance(obs, dict):
            return obs
        # Raw numpy / list grid
        if hasattr(obs, "tolist"):
            obs = obs.tolist()
        return {"frame": obs}

    def list_games(self):
        if KAGGLE_ENV:
            return arcagi3.list_games()
        return self.env.list_games()


# ── MOCK ENV (local dev only) ─────────────────────────────────────────────────
class _MockEnv:
    """Tiny mock so the code can be tested without Kaggle runtime."""
    _games = ["MOCK_GAME_01", "MOCK_GAME_02"]
    _actions = ["up", "down", "left", "right", "action1", "undo"]

    def reset(self, game_id):
        self._step = 0
        self._game = game_id
        return {"frame": [[0]*64 for _ in range(64)], "level": 1, "available_actions": self._actions}

    def step(self, action):
        self._step += 1
        done = self._step >= 200  # mock: episode ends after 200 steps
        obs = {"frame": [[0]*64 for _ in range(64)], "level": 1, "available_actions": self._actions}
        info = {"score": 0.0, "level_complete": False, "game_complete": done}
        return obs, done, info

    def list_games(self):
        return self._games


# ── 2. LOCAL MODEL INFERENCE ──────────────────────────────────────────────────
# Kaggle blocks internet. You MUST bundle your model weights in the notebook
# dataset or use a Kaggle-hosted model (e.g. via kaggle_secrets + local copy).
#
# Options:
#   A) Use a Kaggle Dataset containing quantised model weights (e.g. GGUF/llama.cpp)
#   B) Use a smaller open model (Mistral-7B, Phi-3-mini, etc.) already on Kaggle Models
#   C) For leaderboard-only (non-prize) runs: use the arcprize.org API during dev,
#      then swap to a local model for the final Kaggle submission.

MODEL_PATH = "/kaggle/input/your-model-dataset/model.gguf"   # CHANGE THIS

def call_local_model(system_prompt: str, messages: list[dict], max_tokens: int = 512) -> str:
    """
    Call your bundled offline model. Swap the body here with your inference code.

    Args:
        system_prompt: Role-specific system prompt for this agent.
        messages: List of {"role": "user"|"assistant", "content": str} dicts.
        max_tokens: Max tokens to generate.

    Returns:
        Generated text string.
    """
    # ── STUB — replace with real inference ────────────────────────────────────
    # Example using llama-cpp-python (install via Kaggle dataset):
    #
    #   from llama_cpp import Llama
    #   llm = Llama(model_path=MODEL_PATH, n_ctx=8192, n_gpu_layers=-1)
    #   prompt = build_prompt(system_prompt, messages)
    #   out = llm(prompt, max_tokens=max_tokens, stop=["</s>"])
    #   return out["choices"][0]["text"].strip()
    #
    # ─────────────────────────────────────────────────────────────────────────
    return "[STUB] model response — replace call_local_model() with real inference."


# ── 3. FRAME UTILITIES ────────────────────────────────────────────────────────

def frame_to_text(frame: list[list[int]]) -> str:
    """Compact ASCII representation of a 64x64 frame (hex digits 0-f)."""
    HEX = "0123456789abcdef"
    rows = []
    for row in frame:
        rows.append("".join(HEX[min(c, 15)] for c in row))
    return "\n".join(rows)


def diff_frames(prev: list[list[int]], curr: list[list[int]]) -> str:
    """Return a compact description of cells that changed between frames."""
    changes = []
    for r in range(len(curr)):
        for c in range(len(curr[r])):
            if curr[r][c] != prev[r][c]:
                changes.append(f"({r},{c}): {prev[r][c]}->{curr[r][c]}")
            if len(changes) > 100:  # cap verbosity
                changes.append("... (more)")
                return ", ".join(changes)
    return ", ".join(changes) if changes else "no change"


# ── 4. AGENT DEFINITIONS ──────────────────────────────────────────────────────

EXPLORER_SYSTEM = """You are the Explorer Agent in a team solving a novel interactive game.
You receive the current game frame (a 64x64 grid of colour indices 0-15) and choose ONE action.
Available actions will be listed. You have NO prior knowledge of this game's rules or goal.
Output ONLY the action name (e.g. "up", "left", "action1", "click 32 40"). Nothing else."""

REASONER_SYSTEM = """You are the Reasoner Agent. You receive a history of frames and actions.
Your job: update the shared world model with new observations.
Output a concise JSON object with these keys:
  - "entities": list of observed objects and their properties
  - "rules": list of inferred mechanics
  - "goal_hypothesis": your best guess at the win condition
  - "confidence": 0.0-1.0
Keep it short. No prose outside the JSON."""

PLANNER_SYSTEM = """You are the Planner Agent. Given a world model, output a JSON list of
up to 5 recommended actions in order, e.g. ["up", "up", "action1", "left", "click 10 20"].
Explain each briefly with a "rationale" key. Output ONLY valid JSON: {"plan": [...], "rationale": "..."}"""

CRITIC_SYSTEM = """You are the Critic Agent. Review the proposed plan and world model.
Output JSON: {"approve": true/false, "issues": ["...", ...], "suggestion": "..."}
Approve if the plan is sensible given the evidence. Reject if there are obvious flaws."""

SUMMARIZER_SYSTEM = """You are a Summarizer. Compress the agent history into a short state summary.
Preserve: key mechanics discovered, current goal hypothesis, last 3 actions and outcomes.
Output plain text, max 300 words."""


def explorer_turn(messages: list[dict], available_actions: list[str]) -> str:
    """Ask Explorer to choose a single action."""
    action_hint = f"\nAvailable actions: {', '.join(available_actions)}"
    msgs = messages + [{"role": "user", "content": action_hint}]
    raw = call_local_model(EXPLORER_SYSTEM, msgs, max_tokens=16)
    # Clean: take the first token / word as the action
    action = raw.strip().split()[0] if raw.strip() else available_actions[0]
    # Handle click actions: "click" may be followed by coords
    if raw.strip().lower().startswith("click"):
        parts = raw.strip().split()
        if len(parts) >= 3:
            action = f"click {parts[1]} {parts[2]}"
    return action


def reasoner_turn(messages: list[dict]) -> dict:
    raw = call_local_model(REASONER_SYSTEM, messages, max_tokens=512)
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {"entities": [], "rules": [], "goal_hypothesis": "unknown", "confidence": 0.0}


def planner_turn(world_model: dict, messages: list[dict]) -> dict:
    prompt = f"World model:\n{json.dumps(world_model, indent=2)}\n\nPropose the next actions."
    msgs = messages + [{"role": "user", "content": prompt}]
    raw = call_local_model(PLANNER_SYSTEM, msgs, max_tokens=256)
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {"plan": [], "rationale": "parse error"}


def critic_turn(world_model: dict, plan: dict, messages: list[dict]) -> dict:
    prompt = (
        f"World model:\n{json.dumps(world_model, indent=2)}\n\n"
        f"Proposed plan:\n{json.dumps(plan, indent=2)}\n\nApprove or reject."
    )
    msgs = messages + [{"role": "user", "content": prompt}]
    raw = call_local_model(CRITIC_SYSTEM, msgs, max_tokens=256)
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {"approve": True, "issues": [], "suggestion": ""}


def summarize_history(messages: list[dict]) -> str:
    raw = call_local_model(SUMMARIZER_SYSTEM, messages, max_tokens=400)
    return raw.strip()


# ── 5. MULTI-AGENT COLLABORATION LOOP ────────────────────────────────────────

MAX_ACTIONS_PER_LEVEL = 500          # hard cap per level
CONTEXT_COMPRESS_EVERY = 30          # compress history every N actions
MAX_CRITIC_RETRIES = 3               # max plan revision rounds


def run_game(env: EnvironmentWrapper, game_id: str) -> dict:
    """
    Run the full multi-agent loop on a single game.
    Returns a results dict with per-level action counts and outcomes.
    """
    print(f"\n{'='*60}")
    print(f"Game: {game_id}")
    print(f"{'='*60}")

    obs = env.reset(game_id)
    done = False
    total_actions = 0
    level_results = []
    level = 1

    # Shared conversation history (compressed periodically)
    shared_messages: list[dict] = []
    world_model: dict = {"entities": [], "rules": [], "goal_hypothesis": "unknown", "confidence": 0.0}
    prev_frame = obs.get("frame", [])

    while not done:
        frame = obs.get("frame", [])
        available_actions = obs.get("available_actions", ["up", "down", "left", "right", "action1", "undo"])

        # ── Observation message ────────────────────────────────────────────────
        frame_text = frame_to_text(frame)
        diff_text = diff_frames(prev_frame, frame) if prev_frame else "initial frame"
        obs_msg = f"[Level {level}] Frame changes: {diff_text}\nFull frame (hex):\n{frame_text}"
        shared_messages.append({"role": "user", "content": obs_msg})
        prev_frame = frame

        # ── Reasoner: update world model ───────────────────────────────────────
        world_model = reasoner_turn(shared_messages)
        shared_messages.append({"role": "assistant", "content": f"[Reasoner] {json.dumps(world_model)}"})

        # ── Planner → Critic loop ──────────────────────────────────────────────
        plan = planner_turn(world_model, shared_messages)
        for retry in range(MAX_CRITIC_RETRIES):
            critique = critic_turn(world_model, plan, shared_messages)
            if critique.get("approve", True):
                break
            # Planner revises given critic feedback
            revision_msg = f"Critic rejected: {critique.get('issues', [])}. Suggestion: {critique.get('suggestion', '')}. Revise."
            shared_messages.append({"role": "user", "content": revision_msg})
            plan = planner_turn(world_model, shared_messages)

        approved_actions: list[str] = plan.get("plan", [])
        shared_messages.append({"role": "assistant", "content": f"[Planner] Plan: {approved_actions}"})

        # ── Execute plan, or fall back to Explorer for a single action ─────────
        if approved_actions:
            actions_to_take = approved_actions
        else:
            # Explorer picks a single exploratory action
            actions_to_take = [explorer_turn(shared_messages, available_actions)]

        for action in actions_to_take:
            if total_actions >= MAX_ACTIONS_PER_LEVEL:
                print(f"  [LIMIT] Hit max actions ({MAX_ACTIONS_PER_LEVEL}), ending.")
                done = True
                break

            print(f"  Action #{total_actions+1}: {action}")
            obs, done, info = env.step(action)
            total_actions += 1

            if info.get("level_complete", False):
                print(f"  ✓ Level {level} complete in {total_actions} actions.")
                level_results.append({"level": level, "actions": total_actions, "complete": True})
                level += 1
                prev_frame = []

            if done:
                break

        # ── Periodic context compression ───────────────────────────────────────
        if total_actions % CONTEXT_COMPRESS_EVERY == 0 and total_actions > 0:
            summary = summarize_history(shared_messages)
            shared_messages = [{"role": "user", "content": f"[Summary]\n{summary}"}]
            print(f"  [Context compressed at action {total_actions}]")

    if not level_results or level_results[-1]["level"] < level:
        level_results.append({"level": level, "actions": total_actions, "complete": False})

    game_complete = info.get("game_complete", done)
    print(f"Game {game_id}: {'COMPLETE' if game_complete else 'incomplete'} | Total actions: {total_actions}")
    return {"game_id": game_id, "complete": game_complete, "total_actions": total_actions, "levels": level_results}


# ── 6. MAIN ENTRY POINT ───────────────────────────────────────────────────────

def main():
    env = EnvironmentWrapper()
    games = env.list_games()
    print(f"Found {len(games)} games: {games}")

    all_results = []
    for game_id in games:
        try:
            result = run_game(env, game_id)
            all_results.append(result)
        except Exception as e:
            print(f"[ERROR] Game {game_id} failed: {e}")
            traceback.print_exc()
            all_results.append({"game_id": game_id, "complete": False, "total_actions": 0, "levels": [], "error": str(e)})

    # ── Summary ───────────────────────────────────────────────────────────────
    completed = sum(1 for r in all_results if r["complete"])
    print(f"\n{'='*60}")
    print(f"FINAL: {completed}/{len(all_results)} games completed")
    print(f"{'='*60}")

    # Save results (Kaggle reads submission from the run output)
    out_path = pathlib.Path("/kaggle/working/submission_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()

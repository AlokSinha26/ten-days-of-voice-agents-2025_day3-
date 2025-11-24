"""
agent_day3_wellness.py

Daily Wellness Voice Companion — ready-to-drop-in agent file.
- Clean, minimal, configurable
- JSON persistence to `wellness_log.json` (one list of entries)
- Reads last entry and references it in the agent instructions
- Function tools to record mood/energy, objectives, and complete/save
- Graceful fallbacks if `livekit` or plugins are missing so you can test locally

Original user file path (if you want to map local file): /mnt/data/b18d8549-8c29-4c83-821c-28dd199ee2c1.png
"""

import logging
import json
import os
import asyncio
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# Optional imports with graceful fallbacks for local testing
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*a, **k):
        return None

try:
    from pydantic import Field
except Exception:
    def Field(*a, **k):
        return None

# Try importing livekit; if not installed, provide lightweight stand-ins
try:
    from livekit.agents import (
        Agent,
        AgentSession,
        JobContext,
        JobProcess,
        RoomInputOptions,
        WorkerOptions,
        cli,
        metrics,
        MetricsCollectedEvent,
        RunContext,
        function_tool,
    )
    LIVEKIT_AVAILABLE = True
except Exception:
    LIVEKIT_AVAILABLE = False

    # Minimal stubs so the file can be imported for editing/testing
    class Agent:
        def __init__(self, *a, **k):
            pass

    class AgentSession:
        def __init__(self, *a, **k):
            pass
        async def start(self, *a, **k):
            return None
        def on(self, *a, **k):
            def _decor(f):
                return f
            return _decor

    class JobContext:
        pass

    class JobProcess:
        pass

    class RoomInputOptions:
        def __init__(self, *a, **k):
            pass

    class WorkerOptions:
        def __init__(self, *a, **k):
            pass

    class cli:
        @staticmethod
        def run_app(*a, **k):
            raise RuntimeError("livekit.cli.run_app is not available in this environment")

    def function_tool(f):
        # no-op decorator fallback
        return f

    class metrics:
        class UsageCollector:
            def collect(self, *a, **k):
                pass

# Try to import plugins
try:
    from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
except Exception:
    murf = silero = google = deepgram = noise_cancellation = None
    MultilingualModel = None

# Setup logging
logger = logging.getLogger("wellness_agent")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)

load_dotenv()

# ----------------------------
# Config
# ----------------------------
CONFIG = {
    "WELLNESS_LOG_FILENAME": os.getenv("WELLNESS_LOG_FILENAME", "wellness_log.json"),
    "TTS_VOICE": os.getenv("TTS_VOICE", "en-US-natalie"),
    "LLM_MODEL": os.getenv("LLM_MODEL", "gemini-2.5-flash"),
    "STT_MODEL": os.getenv("STT_MODEL", "nova-3"),
}

# ----------------------------
# Data models
# ----------------------------
@dataclass
class CheckInState:
    mood: Optional[str] = None
    energy: Optional[str] = None
    objectives: List[str] = field(default_factory=list)
    advice_given: Optional[str] = None

    def is_complete(self) -> bool:
        return bool(self.mood and self.energy and len(self.objectives) > 0)

    def to_record(self) -> dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "mood": self.mood,
            "energy": self.energy,
            "objectives": self.objectives,
            "advice": self.advice_given,
        }

@dataclass
class Userdata:
    current_checkin: CheckInState
    history_summary: str = ""
    session_start: datetime = field(default_factory=datetime.now)

# ----------------------------
# Persistence helpers
# ----------------------------

def get_log_path() -> str:
    # Default: save in project root (one level up from this file)
    base_dir = os.path.dirname(__file__)
    backend_dir = os.path.abspath(os.path.join(base_dir, ".."))
    path = os.path.join(backend_dir, CONFIG["WELLNESS_LOG_FILENAME"])
    return path


def load_history() -> List[dict]:
    path = get_log_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            # If older schema saved an object, try to extract 'entries' key
            if isinstance(data, dict) and "entries" in data:
                return data["entries"]
            return []
    except Exception as e:
        logger.warning("Failed to load history (%s): %s", path, e)
        return []


def save_checkin(entry: CheckInState) -> str:
    path = get_log_path()
    history = load_history()
    record = entry.to_record()
    history.append(record)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
        logger.info("Saved check-in to %s", path)
        return path
    except Exception as e:
        logger.exception("Failed to save check-in: %s", e)
        raise

# ----------------------------
# Agent function tools
# ----------------------------

@function_tool
async def record_mood_and_energy(
    ctx: "RunContext[Userdata]",
    mood: str,
    energy: str,
) -> str:
    """Record mood and energy into the session userdata."""
    ctx.userdata.current_checkin.mood = (mood or "").strip()
    ctx.userdata.current_checkin.energy = (energy or "").strip()
    logger.info("Recorded mood='%s' energy='%s'", mood, energy)
    return f"Noted — you're feeling {ctx.userdata.current_checkin.mood} with {ctx.userdata.current_checkin.energy} energy."


@function_tool
async def record_objectives(
    ctx: "RunContext[Userdata]",
    objectives: List[str],
) -> str:
    # sanitize list
    cleaned = [o.strip() for o in objectives if o and o.strip()]
    ctx.userdata.current_checkin.objectives = cleaned[:3]
    logger.info("Recorded objectives: %s", ctx.userdata.current_checkin.objectives)
    return f"Goals recorded: {', '.join(ctx.userdata.current_checkin.objectives)}"


@function_tool
async def complete_checkin(
    ctx: "RunContext[Userdata]",
    final_advice_summary: str,
) -> str:
    state = ctx.userdata.current_checkin
    state.advice_given = (final_advice_summary or "").strip()

    if not state.is_complete():
        missing = []
        if not state.mood:
            missing.append("mood")
        if not state.energy:
            missing.append("energy")
        if not state.objectives:
            missing.append("objectives")
        return "I still need: " + ", ".join(missing)

    try:
        path = save_checkin(state)
        summary = (
            f"You reported feeling {state.mood} with {state.energy} energy. "
            f"Today's goals: {', '.join(state.objectives)}. Advice: {state.advice_given}"
        )
        logger.info("Check-in complete and saved: %s", path)
        return (
            f"Check-in complete. {summary} \n\nI've saved this entry to your wellness log. Have a good day!"
        )
    except Exception:
        return "Check-in recorded in session but failed to save to disk."

# ----------------------------
# Agent class
# ----------------------------

class WellnessAgent(Agent):
    def __init__(self, history_context: str = ""):
        instructions = f"""
You are a calm, grounded, non-clinical wellness companion.
Reference the short history context below when appropriate.

HISTORY SUMMARY:\n{history_context}\n
Session goals:
1) Ask about mood and energy (one question at a time).
2) Ask for 1-3 simple objectives for the day.
3) Offer a brief, practical, non-medical suggestion.
4) Recap and ask for confirmation before saving.

Safety: Do NOT provide medical advice, diagnosis, or crisis counseling. If a user indicates self-harm or severe risk, advise them to seek emergency or professional help immediately.

Use the provided tools to store each piece of data as the user responds.
"""
        super().__init__(instructions=instructions, tools=[record_mood_and_energy, record_objectives, complete_checkin])

# ----------------------------
# Prewarm and entrypoint
# ----------------------------

def prewarm(proc: "JobProcess"):
    # Try to load VAD if available
    try:
        if silero is not None and hasattr(silero, "VAD"):
            proc.userdata["vad"] = silero.VAD.load()
            logger.info("Silero VAD loaded in prewarm.")
    except Exception:
        logger.warning("Prewarm: VAD not available or failed to load.")


async def entrypoint(ctx: "JobContext"):
    # Log minimal context
    try:
        ctx.log_context_fields = {"room": getattr(ctx.room, "name", "unknown")}
    except Exception:
        pass

    logger.info("Starting Wellness Session...")

    # Load history and prepare short summary
    history = load_history()
    if history:
        last = history[-1]
        last_ts = last.get("timestamp", "unknown")
        last_mood = last.get("mood", "unknown")
        last_energy = last.get("energy", "unknown")
        last_obj = last.get("objectives", [])
        history_summary = (
            f"Last check-in on {last_ts}: mood={last_mood}, energy={last_energy}, goals={', '.join(last_obj)}"
        )
        logger.info("History summary: %s", history_summary)
    else:
        history_summary = "No previous history found."
        logger.info("No history found.")

    # Initialize userdata
    userdata = Userdata(current_checkin=CheckInState(), history_summary=history_summary)

    # Create session (if livekit available)
    if LIVEKIT_AVAILABLE and deepgram is not None and google is not None and murf is not None:
        session = AgentSession(
            stt=deepgram.STT(model=CONFIG["STT_MODEL"]),
            llm=google.LLM(model=CONFIG["LLM_MODEL"]),
            tts=murf.TTS(voice=CONFIG["TTS_VOICE"], style="Conversation", text_pacing=True),
            turn_detection=(MultilingualModel() if MultilingualModel is not None else None),
            vad=ctx.proc.userdata.get("vad"),
            userdata=userdata,
        )

        usage_collector = metrics.UsageCollector()

        @session.on("metrics_collected")
        def _on_metrics(ev: "MetricsCollectedEvent"):
            usage_collector.collect(ev.metrics)

        await session.start(
            agent=WellnessAgent(history_context=history_summary),
            room=ctx.room,
            room_input_options=RoomInputOptions(noise_cancellation=(noise_cancellation.BVC() if noise_cancellation is not None else None)),
        )
    else:
        # Live testing fallback: simulate the flow in console
        logger.info("LiveKit or plugins not available — running a console simulation for testing.")
        await console_simulation(userdata, history_summary)

    try:
        await ctx.connect()
    except Exception:
        logger.debug("ctx.connect() not available in this environment")

# ----------------------------
# Console simulation (useful for local testing without livekit)
# ----------------------------

async def console_simulation(userdata: Userdata, history_summary: str):
    print("\n=== Wellness Companion (console simulation) ===\n")
    if history_summary and "No previous" not in history_summary:
        print("(Reference) ", history_summary)
    # Ask mood
    mood = input("How are you feeling today? (e.g., tired, okay, happy) ")
    energy = input("How's your energy? (low, medium, high) ")
    await record_mood_and_energy(type("Z", (), {"userdata": userdata}), mood, energy)

    # Objectives
    print("List 1-3 simple things you'd like to do today. Enter each on a new line. Finish with an empty line.")
    objectives = []
    while True:
        line = input()
        if not line.strip():
            break
        objectives.append(line.strip())
    await record_objectives(type("Z", (), {"userdata": userdata}), objectives)

    # Offer simple advice
    advice = simple_advice(userdata.current_checkin)
    print("\nSuggestion:", advice)

    # Confirm and save
    confirm = input("Does this sound right? (yes/no) ")
    if confirm.strip().lower().startswith("y"):
        await complete_checkin(type("Z", (), {"userdata": userdata}), advice)
        print("Saved. Goodbye!")
    else:
        print("Okay — not saved. You can run the session again when ready.")


def simple_advice(state: CheckInState) -> str:
    # Minimal rule-based advice generator
    if state.energy and state.energy.lower() in ("low", "drained"):
        return "Try a short 5-minute walk and a glass of water — small steps can boost your energy."
    if state.mood and any(w in state.mood.lower() for w in ("anx", "stres", "nerv")):
        return "Try a brief breathing break: 3 deep breaths, then a 1-minute stretch."
    if state.objectives and len(state.objectives) > 0:
        return "Break your main task into 15-minute focused chunks with short breaks in between."
    return "Keep tasks small and kind to yourself — even tiny progress counts."

# ----------------------------
# CLI runner
# ----------------------------

def main():
    try:
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
    except Exception:
        # Fallback: run console sim directly
        logger.warning("Failed to start live app. Running console simulation instead.")
        asyncio.run(console_simulation(Userdata(current_checkin=CheckInState()), "No previous history found."))

if __name__ == "__main__":
    main()

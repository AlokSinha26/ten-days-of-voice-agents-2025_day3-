import logging
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# import our small DB helper
from fraud_db import find_case_by_username, update_case

logger = logging.getLogger("agent")
load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a voice agent representing Aurora National Bank's Fraud Department.  
You speak in a calm, professional, reassuring tone.  
When the session starts, you must:  
1. Introduce yourself as a representative from Aurora National Bank Fraud Department.  
2. Ask who you are speaking with.  
3. Use the provided tools to load the fraud case using the customer's name (e.g., John Carter).  
4. Never ask for full card numbers, PINs, or sensitive credentials.  
5. Use only the masked card, merchant, amount, time, location, and a non-sensitive security question for verification.  
6. Follow the fraud-check flow strictly: verification → read suspicious transaction → ask if legitimate → update case via tools.  
7. End the call politely after updating the fraud case status."""

        )

    # Optional: any other agent customization can go here


# prewarm VAD (unchanged)
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# Tools exposed to the LLM / runtime
@function_tool
async def load_case(context: RunContext, user_name: str):
    """Tool: load a fraud case by userName (case-insensitive).
    Returns a short JSON-like summary (no secret fields)."""
    context.logger.info(f"Tool load_case called for user={user_name}")
    case = find_case_by_username(user_name)
    if not case:
        return {"ok": False, "reason": "not_found"}

    # Return only safe fields for reading aloud; never return securityAnswer
    safe = {
        "userName": case.get("userName"),
        "securityIdentifier": case.get("securityIdentifier"),
        "cardEnding": case.get("cardEnding"),
        "transactionAmount": case.get("transactionAmount"),
        "transactionName": case.get("transactionName"),
        "transactionTime": case.get("transactionTime"),
        "transactionCategory": case.get("transactionCategory"),
        "transactionSource": case.get("transactionSource"),
        "location": case.get("location"),
        "securityQuestion": case.get("securityQuestion"),
        "status": case.get("status"),
    }
    return {"ok": True, "case": safe}


@function_tool
async def verify_answer(context: RunContext, user_name: str, answer: str):
    """Tool: verify the provided security answer (case-insensitive).
    Returns ok True/False and a message."""
    context.logger.info("Tool verify_answer called")
    case = find_case_by_username(user_name)
    if not case:
        return {"ok": False, "reason": "not_found"}

    correct = str(case.get("securityAnswer", "")).strip().lower()
    if answer.strip().lower() == correct:
        return {"ok": True}
    else:
        return {"ok": False, "reason": "wrong_answer"}


@function_tool
async def update_case_status(context: RunContext, user_name: str, new_status: str, note: str):
    """Tool: update the status and outcome_note of the case on disk."""
    context.logger.info(f"Tool update_case_status called for {user_name} -> {new_status}")
    updates = {"status": new_status, "outcome_note": note}
    ok = update_case(user_name, updates)
    if ok:
        return {"ok": True}
    else:
        return {"ok": False, "reason": "not_found"}


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {"room": ctx.room.name}

    # Build voice pipeline using your prior config
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session with our Assistant (tools are auto-registered by decorator)
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    # Connect and join
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

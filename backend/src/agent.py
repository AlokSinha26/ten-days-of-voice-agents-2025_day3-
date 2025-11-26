# ======================================================
# 💼 DAY 5: AI SALES DEVELOPMENT REP (SDR) - SALESFORCE RECEPTIONIST
# 👩‍💼 "Priya" - Salesforce Receptionist & Lead Capture Agent
# 🚀 Features: FAQ Retrieval, Lead Qualification, JSON Database
# ======================================================

import logging
import json
import os
import asyncio
from datetime import datetime
from typing import Annotated, Optional
from dataclasses import dataclass, asdict

print("\n" + "💼" * 50)
print("🚀 AI SDR AGENT - DAY 5 TUTORIAL (SALESFORCE RECEPTIONIST)")
print("📚 REPRESENTING: Salesforce (Receptionist Persona)")
print("💡 agent.py LOADED SUCCESSFULLY!")
print("💼" * 50 + "\n")

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

# 🔌 PLUGINS (kept same as example)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# 📂 1. KNOWLEDGE BASE (FAQ) - now for Salesforce
# ======================================================

FAQ_FILE = "salesforce_faq.json"
LEADS_FILE = "salesforce_leads_db.json"

DEFAULT_FAQ = [
    {
        "question": "What does Salesforce do?",
        "answer": "Salesforce is a leading customer relationship management (CRM) platform that helps companies manage sales, service, marketing, commerce, and data in one place. Key products include Sales Cloud, Service Cloud, Marketing Cloud, Commerce Cloud, and the Salesforce Platform for building custom apps."
    },
    {
        "question": "Who is Salesforce for?",
        "answer": "Salesforce serves businesses of all sizes — from small startups to large enterprises — across many industries looking to centralize customer data, automate processes, and deliver personalized customer experiences."
    },
    {
        "question": "Do you offer a free tier or trial?",
        "answer": "Salesforce provides free trials for many of its products so teams can evaluate features. For ongoing usage, Salesforce offers multiple editions with different feature sets and pricing; we recommend trying a free trial or contacting our sales team for details."
    },
    {
        "question": "What are the basic pricing options?",
        "answer": "Pricing varies by product and edition (for example, Sales Cloud editions differ by feature set). Exact pricing depends on the product, edition, number of users, and any add-ons. For accurate pricing, contact Salesforce sales or request a quote."
    },
    {
        "question": "Can Salesforce integrate with our existing systems?",
        "answer": "Yes—Salesforce provides extensive integration capabilities via APIs, MuleSoft, AppExchange apps, and prebuilt connectors to integrate with ERP systems, marketing tools, data warehouses, and more."
    },
    {
        "question": "Can we customize Salesforce for our business?",
        "answer": "Absolutely—Salesforce is highly customizable with point-and-click tools (Flows, Process Builder), declarative configuration, and programmatic customization via Apex, Lightning Web Components, and APIs."
    },
    {
        "question": "Do you offer support and training?",
        "answer": "Salesforce offers a range of support plans and training options including Trailhead (free learning platform), certification programs, partner-led training, and paid support tiers."
    }
]

def load_knowledge_base():
    """Generates FAQ file if missing, then loads it."""
    try:
        # Use directory relative to this file so saving/reading works when copied
        base_dir = os.path.dirname(__file__)
        path = os.path.join(base_dir, FAQ_FILE)
        if not os.path.exists(path):
            with open(path, "w", encoding='utf-8') as f:
                json.dump(DEFAULT_FAQ, f, indent=4)
        with open(path, "r", encoding='utf-8') as f:
            # Return the faq content as a string (for including in prompt instructions)
            return json.dumps(json.load(f))
    except Exception as e:
        print(f"⚠️ Error loading FAQ: {e}")
        return ""

SALESFORCE_FAQ_TEXT = load_knowledge_base()

# ======================================================
# 💾 2. LEAD DATA STRUCTURE
# ======================================================

@dataclass
class LeadProfile:
    name: Optional[str] = None
    company: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    use_case: Optional[str] = None
    team_size: Optional[str] = None
    timeline: Optional[str] = None

    def is_qualified(self):
        """Returns True if we have the minimum info (Name + Email + Use Case)"""
        return all([self.name, self.email, self.use_case])

@dataclass
class Userdata:
    lead_profile: LeadProfile

# ======================================================
# 🛠️ 3. SDR TOOLS
# ======================================================

@function_tool
async def update_lead_profile(
    ctx: RunContext[Userdata],
    name: Annotated[Optional[str], Field(description="Customer's name")] = None,
    company: Annotated[Optional[str], Field(description="Customer's company name")] = None,
    email: Annotated[Optional[str], Field(description="Customer's email address")] = None,
    role: Annotated[Optional[str], Field(description="Customer's job title")] = None,
    use_case: Annotated[Optional[str], Field(description="What they want to build or use Salesforce for")] = None,
    team_size: Annotated[Optional[str], Field(description="Number of people in their team")] = None,
    timeline: Annotated[Optional[str], Field(description="When they want to start (e.g., Now, next month)")] = None,
) -> str:
    """
    ✍️ Captures lead details provided by the user during conversation.
    Only call this when the user explicitly provides information.
    """
    profile = ctx.userdata.lead_profile

    # Update only fields that are provided (not None)
    if name: profile.name = name
    if company: profile.company = company
    if email: profile.email = email
    if role: profile.role = role
    if use_case: profile.use_case = use_case
    if team_size: profile.team_size = team_size
    if timeline: profile.timeline = timeline

    print(f"📝 UPDATING LEAD: {profile}")
    return "Lead profile updated. Continue the conversation."

@function_tool
async def submit_lead_and_end(
    ctx: RunContext[Userdata],
) -> str:
    """
    💾 Saves the lead to the database and signals the end of the call.
    Call this when the user says goodbye or 'that's all'.
    """
    profile = ctx.userdata.lead_profile

    # Save to JSON file (Append mode)
    base_dir = os.path.dirname(__file__)
    db_path = os.path.join(base_dir, LEADS_FILE)

    entry = asdict(profile)
    entry["timestamp"] = datetime.now().isoformat()

    # Read existing, append, write back (Simple JSON DB)
    existing_data = []
    if os.path.exists(db_path):
        try:
            with open(db_path, "r", encoding='utf-8') as f:
                existing_data = json.load(f)
        except Exception:
            existing_data = []

    existing_data.append(entry)

    with open(db_path, "w", encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4)

    print(f"✅ LEAD SAVED TO {LEADS_FILE}")
    # Short summary message returned to agent to speak to user
    summary_text = f"Thanks {profile.name or 'there'}, I have your info regarding {profile.use_case or 'your interest'}. We will email you at {profile.email or 'the provided email'}. Goodbye!"
    return summary_text

# ======================================================
# 🧠 4. AGENT DEFINITION (Receptionist Persona for Salesforce)
# ======================================================

class SDRAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=f"""
You are 'Priya', a warm, professional receptionist and Sales Development Representative (SDR) representing Salesforce.

📘 **YOUR KNOWLEDGE BASE (FAQ):**
{SALESFORCE_FAQ_TEXT}

🎯 **YOUR GOAL:**
1. Welcome each visitor warmly and briefly explain what Salesforce does if asked.
2. Answer product/company/pricing questions using the FAQ above. If the FAQ does not contain the detail, say: "I'll connect you with Salesforce sales for an accurate quote" (do NOT make up specific pricing).
3. **QUALIFY THE LEAD:** Naturally gather these details during the conversation:
   - Name
   - Company
   - Email
   - Role / Job title
   - Use case (what they want to achieve with Salesforce)
   - Team size
   - Timeline (Now / Soon / Later)

⚙️ **BEHAVIOR & DIALOGUE GUIDELINES:**
- Greet warmly: "Hi! I'm Priya from Salesforce — how can I help you today?"
- Ask open questions to surface needs: "What brought you to Salesforce today?" / "What are you trying to solve?"
- Keep the conversation focused on understanding the user's needs before trying to sell.
- After answering a question, ask one qualification question when it fits naturally. Example: 
  "We have Sales Cloud to help sales teams manage opportunities. By the way, how large is your sales team?"
- Use `update_lead_profile` whenever the user provides any lead detail (name, email, use case, etc.).
- When the user indicates they're finished (e.g., "that's all", "thanks", "goodbye"), call `submit_lead_and_end` to save the lead and provide a short summary.

🚫 **RESTRICTIONS:**
- Do not invent specific pricing or contractual terms. If uncertain, say: "I'll connect you with our sales team for exact pricing and editions."
- Stick to the FAQ content for product details. If the user asks something beyond the FAQ, offer a trial or sales contact.

🗂️ **SAMPLE FLOW:**
1. Priya: Warm greeting.
2. Priya: Asks purpose / what they're working on.
3. Priya: Answers FAQ-based questions.
4. Priya: Asks qualification questions naturally, updates lead via tools.
5. Priya: On user goodbye, saves lead and provides a short summary.

Remember: Be helpful, concise, and never hallucinate product-specific facts beyond the FAQ.
""",
            tools=[update_lead_profile, submit_lead_and_end],
        )

# ======================================================
# 🎬 ENTRYPOINT
# ======================================================

def prewarm(proc: JobProcess):
    # Load VAD model or other prewarm assets if available
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    print("\n" + "💼" * 25)
    print("🚀 STARTING SALESFORCE RECEPTIONIST SDR SESSION")
    print("🎧 Using speech & TTS plugins configured in session")
    print("💼" * 25 + "\n")

    # 1. Initialize State
    userdata = Userdata(lead_profile=LeadProfile())

    # 2. Setup Agent Session
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-natalie",
            style="Promo",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata,
    )

    # 3. Start the interactive session with the SDRAgent
    await session.start(
        agent=SDRAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    # Run the worker application
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

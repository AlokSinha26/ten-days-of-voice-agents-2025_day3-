import logging
import json
import os
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path

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

# import our small DB helper for fraud (keeps your original functions)
from fraud_db import find_case_by_username, update_case

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ---- file helper functions ----
BASE_DIR = Path(__file__).parent.resolve()
CATALOG_PATH = BASE_DIR / "catalog.json"
ORDERS_DIR = BASE_DIR / "orders"
ORDERS_INDEX = BASE_DIR / "orders_index.json"
CARTS_PATH = BASE_DIR / "carts.json"

ORDERS_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path, default):
    if Path(path).exists():
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---- Assistant persona (grocery assistant) ----
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Aurora Groceries — a friendly food & grocery ordering assistant. "
                "Greet the user, ask how you can help, clarify sizes/quantities/brands when needed, "
                "and confirm cart changes. Support adding individual items and 'ingredients for X' "
                "requests. Save orders to JSON when placed and support order status checks."
            )
        )


# prewarm VAD (unchanged)
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# ---- Existing fraud tools (kept unchanged) ----
@function_tool
async def load_case(context: RunContext, user_name: str):
    context.logger.info(f"Tool load_case called for user={user_name}")
    case = find_case_by_username(user_name)
    if not case:
        return {"ok": False, "reason": "not_found"}

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
    context.logger.info(f"Tool update_case_status called for {user_name} -> {new_status}")
    updates = {"status": new_status, "outcome_note": note}
    ok = update_case(user_name, updates)
    if ok:
        return {"ok": True}
    else:
        return {"ok": False, "reason": "not_found"}


# ---- New ordering tools ----

def _load_catalog():
    return read_json(CATALOG_PATH, default={"items": []})


def _save_cart_map(cart_map):
    write_json(CARTS_PATH, cart_map)


def _load_cart_map():
    return read_json(CARTS_PATH, default={})


def _item_lookup_by_name(name):
    catalog = _load_catalog()
    name_lower = name.strip().lower()
    for item in catalog.get("items", []):
        if item.get("name", "").strip().lower() == name_lower:
            return item
    # also support partial match
    for item in catalog.get("items", []):
        if name_lower in item.get("name", "").strip().lower():
            return item
    return None


@function_tool
async def load_catalog(context: RunContext):
    """Tool: load the catalog file and return items count and some sample entries."""
    context.logger.info("Tool load_catalog called")
    cat = _load_catalog()
    items = cat.get("items", [])
    sample = items[:6]
    return {"ok": True, "count": len(items), "sample": sample}


@function_tool
async def add_to_cart(context: RunContext, user_name: str, item_name: str, quantity: int = 1, note: str = ""):
    """Tool: add a catalog item to user's cart (persists to carts.json)."""
    context.logger.info(f"Tool add_to_cart called: {user_name} -> {item_name} x{quantity}")
    item = _item_lookup_by_name(item_name)
    if not item:
        return {"ok": False, "reason": "item_not_found"}

    cart_map = _load_cart_map()
    cart = cart_map.get(user_name, {"items": []})
    # find existing same item by id
    found = None
    for it in cart["items"]:
        if it["id"] == item["id"]:
            found = it
            break
    if found:
        found["quantity"] = found.get("quantity", 1) + max(0, int(quantity))
        if note:
            found["note"] = note
    else:
        cart["items"].append({
            "id": item["id"],
            "name": item["name"],
            "price": item["price"],
            "quantity": max(1, int(quantity)),
            "note": note,
        })
    cart_map[user_name] = cart
    _save_cart_map(cart_map)
    return {"ok": True, "cart": cart, "message": f"Added {quantity} x {item['name']} to cart."}


@function_tool
async def remove_from_cart(context: RunContext, user_name: str, item_name: str):
    context.logger.info(f"Tool remove_from_cart called: {user_name} -> {item_name}")
    cart_map = _load_cart_map()
    cart = cart_map.get(user_name, {"items": []})
    item = _item_lookup_by_name(item_name)
    if not item:
        return {"ok": False, "reason": "item_not_found"}
    before = len(cart["items"])
    cart["items"] = [i for i in cart["items"] if i["id"] != item["id"]]
    cart_map[user_name] = cart
    _save_cart_map(cart_map)
    return {"ok": True, "removed_count": before - len(cart["items"]), "cart": cart}


@function_tool
async def list_cart(context: RunContext, user_name: str):
    context.logger.info(f"Tool list_cart called for {user_name}")
    cart_map = _load_cart_map()
    cart = cart_map.get(user_name, {"items": []})
    total = sum(i.get("price", 0) * i.get("quantity", 1) for i in cart["items"])
    return {"ok": True, "cart": cart, "total": round(total, 2)}


# small recipes mapping for "ingredients for X"
RECIPES = {
    "peanut butter sandwich": [
        {"name": "Bread - Whole Wheat", "qty": 2},
        {"name": "Peanut Butter (Jar)", "qty": 1},
    ],
    "pasta for two": [
        {"name": "Pasta (500g)", "qty": 1},
        {"name": "Marinara Sauce", "qty": 1},
        {"name": "Parmesan (block)", "qty": 1},
    ],
    "scrambled eggs": [
        {"name": "Eggs (dozen)", "qty": 1},
        {"name": "Butter (200g)", "qty": 1},
        {"name": "Milk (1L)", "qty": 1},
    ],
}


@function_tool
async def add_recipe(context: RunContext, user_name: str, dish: str, servings: int = 1):
    """Tool: map dish -> multiple catalog items and add them to the cart."""
    context.logger.info(f"Tool add_recipe called: {user_name} -> {dish} x{servings}")
    dish_key = dish.strip().lower()
    recipe = RECIPES.get(dish_key)
    if not recipe:
        return {"ok": False, "reason": "recipe_not_found"}
    added = []
    for entry in recipe:
        qty = max(1, int(entry.get("qty", 1)) * max(1, int(servings)))
        res = await add_to_cart(context, user_name, entry["name"], qty)
        if res.get("ok"):
            added.append({"name": entry["name"], "qty": qty})
    return {"ok": True, "added": added, "message": f"Added ingredients for {dish}."}


@function_tool
async def place_order(context: RunContext, user_name: str, customer_name: str = "", address: str = ""):
    """Tool: place an order for the user's current cart, save to orders/<order_id>.json and clear the cart."""
    context.logger.info(f"Tool place_order called by {user_name}")
    cart_map = _load_cart_map()
    cart = cart_map.get(user_name, {"items": []})
    if not cart["items"]:
        return {"ok": False, "reason": "cart_empty"}

    total = sum(i.get("price", 0) * i.get("quantity", 1) for i in cart["items"])
    order_id = str(uuid.uuid4())
    placed_at = datetime.now(timezone.utc).isoformat()
    order = {
        "order_id": order_id,
        "user_name": user_name,
        "customer_name": customer_name,
        "address": address,
        "items": cart["items"],
        "total": round(total, 2),
        "placed_at": placed_at,
        "status": "placed",
        "notes": "",
    }

    # save order file
    order_path = ORDERS_DIR / f"{order_id}.json"
    write_json(order_path, order)

    # update index
    idx = read_json(ORDERS_INDEX, default=[])
    idx.append({"order_id": order_id, "user_name": user_name, "placed_at": placed_at})
    write_json(ORDERS_INDEX, idx)

    # clear cart
    cart_map[user_name] = {"items": []}
    _save_cart_map(cart_map)

    return {"ok": True, "order": order, "message": f"Order placed with id {order_id}"}


@function_tool
async def get_order_status(context: RunContext, order_id: str):
    """Tool: return a mocked order status based on elapsed time since placed_at."""
    context.logger.info(f"Tool get_order_status called for {order_id}")
    order_path = ORDERS_DIR / f"{order_id}.json"
    if not order_path.exists():
        return {"ok": False, "reason": "order_not_found"}

    order = read_json(order_path, default={})
    placed = order.get("placed_at")
    try:
        placed_dt = datetime.fromisoformat(placed)
    except Exception:
        placed_dt = None

    elapsed = 0
    if placed_dt:
        elapsed = (datetime.now(timezone.utc) - placed_dt).total_seconds()

    # simple mocked progression
    if elapsed < 60:
        status = "preparing"
        eta_seconds = max(60 - int(elapsed), 10)
    elif elapsed < 180:
        status = "out_for_delivery"
        eta_seconds = max(180 - int(elapsed), 30)
    else:
        status = "delivered"
        eta_seconds = 0

    # update order status on disk (non-destructive)
    if order.get("status") != status:
        order["status"] = status
        write_json(order_path, order)

    return {"ok": True, "order_id": order_id, "status": status, "eta_seconds": eta_seconds, "order": order}


@function_tool
async def list_orders(context: RunContext, user_name: str):
    """Tool: list previous orders for a user."""
    context.logger.info(f"Tool list_orders called for {user_name}")
    idx = read_json(ORDERS_INDEX, default=[])
    user_orders = [entry for entry in idx if entry.get("user_name") == user_name]
    # optionally read brief info for each order
    orders = []
    for entry in user_orders:
        path = ORDERS_DIR / f"{entry['order_id']}.json"
        if path.exists():
            data = read_json(path, default={})
            orders.append({"order_id": entry["order_id"], "placed_at": entry["placed_at"], "total": data.get("total"), "status": data.get("status")})
    return {"ok": True, "orders": orders}


# ---- AgentSession entrypoint (keeps your original setup, just uses the new Assistant) ----
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

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

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

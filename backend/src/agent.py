"""
Cleaned, updated voice-commerce agent for "Namkha Mountain Traders" — a small ACP-inspired voice shopping assistant.
- Shop persona: Tibetan-flavored mountain shop selling shawls, blankets, yak-wool goods, mugs, hoodies, etc.
- Compact merchant layer (catalog, orders persistence).
- LLM-exposed function tools: show_catalog, add_to_cart, show_cart, clear_cart, place_order, last_order.

Notes:
- This file is adapted from the sample you provided and cleaned for clarity.
- Shop introduction and persona updated to the Tibetan-flavored shop name and story.
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional

from pydantic import Field
from dotenv import load_dotenv

# Placeholder imports for the livekit agent framework and plugins used in your sample.
# Keep these if you will run in the same environment; otherwise adapt to your runtime.
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
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("namkha_shop")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

load_dotenv(".env.local")

# -------------------------
# Shop Identity (Tibetan-flavored)
# -------------------------
SHOP_NAME = "Namkha Mountain Traders"
SHOP_INTRO = (
    f"{SHOP_NAME} — an ancient mountain emporium sitting on high passes for generations. "
    "Locals and travellers alike come here for yak-wool shawls, heavy blankets, hand-spun caps, "
    "and other goods made for life in the mountains. I'm your friendly shopkeeper — I help you browse, "
    "add items to a cart and place simple orders."
)

# -------------------------
# Product Catalog (mountain goods + a few everyday items)
# -------------------------
CATALOG: List[Dict] = [
    # Wool & mountain textiles
    {
        "id": "shawl-001",
        "name": "Yak Wool Shawl",
        "description": "Thick handwoven yak-wool shawl — warm and breathable for high-altitude cold.",
        "price": 2499,
        "currency": "INR",
        "category": "shawl",
        "color": "natural",
        "sizes": ["One-size"],
    },
    {
        "id": "blanket-001",
        "name": "Handloom Mountain Blanket",
        "description": "Heavy woven blanket for cold nights; traditional mountain pattern.",
        "price": 3999,
        "currency": "INR",
        "category": "blanket",
        "color": "maroon",
        "sizes": ["Queen", "King"],
    },
    {
        "id": "cap-001",
        "name": "Hand-spun Wool Cap",
        "description": "Compact wool cap, keeps ears warm on windy passes.",
        "price": 499,
        "currency": "INR",
        "category": "cap",
        "color": "black",
        "sizes": ["S", "M", "L"],
    },
    {
        "id": "glove-001",
        "name": "Insulated Wool Gloves",
        "description": "Wool-lined gloves with durable stitching for hiking and chores.",
        "price": 699,
        "currency": "INR",
        "category": "gloves",
        "color": "brown",
        "sizes": ["M", "L"],
    },
    # Everyday / souvenir items
    {
        "id": "mug-001",
        "name": "Stoneware Chai Mug",
        "description": "Hand-glazed ceramic mug perfect for hot tea after a long day.",
        "price": 299,
        "currency": "INR",
        "category": "mug",
        "color": "blue",
        "sizes": [],
    },
    {
        "id": "tee-001",
        "name": "Mountain Cotton Tee",
        "description": "Comfort-fit cotton t-shirt with a small mountain motif.",
        "price": 799,
        "currency": "INR",
        "category": "tshirt",
        "color": "olive",
        "sizes": ["S", "M", "L", "XL"],
    },
    {
        "id": "hoodie-001",
        "name": "Cozy Mountain Hoodie",
        "description": "Fleece-lined pullover hoodie for chilly mornings.",
        "price": 1499,
        "currency": "INR",
        "category": "hoodie",
        "color": "grey",
        "sizes": ["M", "L", "XL"],
    },
]

# -------------------------
# Orders persistence
# -------------------------
ORDERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orders.json")
try:
    if not os.path.exists(ORDERS_FILE):
        with open(ORDERS_FILE, "w") as f:
            json.dump([], f)
except Exception as e:
    logger.error(f"Unable to initialize orders file at {ORDERS_FILE}: {e}")


def _load_all_orders() -> List[Dict]:
    try:
        with open(ORDERS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save_order(order: Dict):
    orders = _load_all_orders()
    orders.append(order)
    temp_path = ORDERS_FILE + ".tmp"
    try:
        with open(temp_path, "w") as f:
            json.dump(orders, f, indent=2)
        os.replace(temp_path, ORDERS_FILE)
    except Exception as e:
        logger.error(f"Failed to persist order to {ORDERS_FILE}: {e}")
        # best-effort cleanup
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        raise

# -------------------------
# Session userdata
# -------------------------
@dataclass
class Userdata:
    customer_name: Optional[str] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    cart: List[Dict] = field(default_factory=list)
    orders: List[Dict] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)

# -------------------------
# Merchant helpers
# -------------------------

def list_products(filters: Optional[Dict] = None) -> List[Dict]:
    """Simple filtering by category, max_price, color, size or free text query."""
    filters = filters or {}
    q = (filters.get("q") or "").lower()
    category = (filters.get("category") or "").lower()
    max_price = filters.get("max_price")
    color = (filters.get("color") or "").lower()
    size = (filters.get("size") or "")

    results = []
    for p in CATALOG:
        ok = True
        if category and category not in p.get("category", "").lower():
            ok = False
        if max_price is not None:
            try:
                if p.get("price", 0) > int(max_price):
                    ok = False
            except Exception:
                pass
        if color and p.get("color") and color != p.get("color").lower():
            ok = False
        if size and (not p.get("sizes") or size not in p.get("sizes", [])):
            ok = False
        if q:
            # match against name, description, or category
            name_match = q in p.get("name", "").lower()
            desc_match = q in p.get("description", "").lower()
            cat_match = q in p.get("category", "").lower()
            if not (name_match or desc_match or cat_match):
                ok = False
        if ok:
            results.append(p)
    return results


def resolve_product_reference(ref_text: str, candidates: Optional[List[Dict]] = None) -> Optional[Dict]:
    """Resolve a spoken reference like 'second shawl' or 'shawl-001' to a product.
    Heuristics: ordinals, id exact match, color+category, category match, name substring, numeric index.
    """
    if not ref_text:
        return None
    ref = ref_text.lower().strip()
    cand = candidates if candidates is not None else CATALOG

    # ordinal words
    ordinals = {"first": 0, "second": 1, "third": 2, "fourth": 3}
    for word, idx in ordinals.items():
        if word in ref and idx < len(cand):
            return cand[idx]

    # id exact match
    for p in cand:
        if p["id"].lower() == ref:
            return p

    # color + category match
    for p in cand:
        if p.get("color") and p["color"].lower() in ref and p.get("category") and p["category"] in ref:
            return p

    # category match (e.g., "gloves" -> matches category "gloves")
    tokens = [t for t in ref.split() if len(t) > 2]
    for p in cand:
        if p.get("category") and p["category"].lower() in ref:
            return p

    # numeric index like '2' -> second
    for token in ref.split():
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(cand):
                return cand[idx]

    # name substring match (require tokens >2 chars to avoid stop words)
    for p in cand:
        name = p.get("name", "").lower()
        if all(tok in name for tok in tokens):
            return p

    # fallback: match any token in name
    for p in cand:
        for tok in tokens:
            if tok in p.get("name", "").lower():
                return p

    return None


def create_order_object(line_items: List[Dict], currency: str = "INR") -> Dict:
    items = []
    total = 0
    for li in line_items:
        pid = li.get("product_id")
        qty = int(li.get("quantity", 1))
        prod = next((p for p in CATALOG if p["id"] == pid), None)
        if not prod:
            raise ValueError(f"Product {pid} not found")
        line_total = prod["price"] * qty
        total += line_total
        items.append({
            "product_id": pid,
            "name": prod["name"],
            "unit_price": prod["price"],
            "quantity": qty,
            "line_total": line_total,
            "attrs": li.get("attrs", {}),
        })
    order = {
        "id": f"order-{str(uuid.uuid4())[:8]}",
        "items": items,
        "total": total,
        "currency": currency,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    try:
        _save_order(order)
    except Exception:
        # let caller decide how to handle; surface the error upward
        raise
    return order


def get_most_recent_order() -> Optional[Dict]:
    all_orders = _load_all_orders()
    if not all_orders:
        return None
    return all_orders[-1]

# -------------------------
# Tools exposed to LLM
# -------------------------
@function_tool
async def show_catalog(
    ctx: RunContext[Userdata],
    q: Optional[str] = Field(default=None, description="Search query to find products. Use this when customer asks for a product by name or type (e.g., 'gloves', 'wool shawl', 'mug'). This searches product names, descriptions, and categories."),
    category: Optional[str] = Field(default=None, description="Filter by product category. Use when customer mentions a product type (e.g., 'gloves', 'shawl', 'blanket', 'mug', 'tshirt', 'hoodie')."),
    max_price: Optional[int] = Field(default=None, description="Maximum price filter (optional)"),
    color: Optional[str] = Field(default=None, description="Color filter (optional)"),
) -> str:
    """Show the product catalog. This is your PRIMARY tool for accessing products.
    ALWAYS call this function when a customer:
    - Asks for a product (e.g., "give me gloves", "I want gloves", "order gloves")
    - Wants to browse or see what's available
    - Mentions a product name or type
    
    You can call it with no parameters to show all products, or with q/category to filter.
    """
    userdata = ctx.userdata
    # normalize simple synonyms for category
    if category:
        cat = category.lower()
        if cat in ("tee", "tshirt", "t-shirts", "tees"):
            category = "tshirt"
        elif cat in ("glove", "gloves"):
            category = "gloves"
    
    # If query is "gloves" or similar, also try category search
    if q:
        q_lower = q.lower().strip()
        # Normalize common product queries to categories
        if q_lower in ("glove", "gloves"):
            # Try both q and category
            filters_q = {k: v for k, v in {"q": q, "max_price": max_price, "color": color}.items() if v is not None}
            filters_cat = {k: v for k, v in {"category": "gloves", "max_price": max_price, "color": color}.items() if v is not None}
            prods_q = list_products(filters_q)
            prods_cat = list_products(filters_cat)
            # Combine and deduplicate
            prods = list({p["id"]: p for p in prods_q + prods_cat}.values())
        elif q_lower in ("tee", "tshirt", "t-shirts", "tees"):
            filters_q = {k: v for k, v in {"q": q, "max_price": max_price, "color": color}.items() if v is not None}
            filters_cat = {k: v for k, v in {"category": "tshirt", "max_price": max_price, "color": color}.items() if v is not None}
            prods_q = list_products(filters_q)
            prods_cat = list_products(filters_cat)
            prods = list({p["id"]: p for p in prods_q + prods_cat}.values())
        else:
            filters = {k: v for k, v in {"q": q, "category": category, "max_price": max_price, "color": color}.items() if v is not None}
            prods = list_products(filters)
    elif category:
        filters = {k: v for k, v in {"category": category, "max_price": max_price, "color": color}.items() if v is not None}
        prods = list_products(filters)
    elif max_price is not None or color:
        filters = {k: v for k, v in {"max_price": max_price, "color": color}.items() if v is not None}
        prods = list_products(filters)
    else:
        # No filters - show all products
        prods = CATALOG
    
    if not prods:
        return "Sorry — I couldn't find any items that match. Would you like to try another search?"
    lines = [f"{SHOP_NAME}: Here are the top {min(8, len(prods))} items I found:"]
    for idx, p in enumerate(prods[:8], start=1):
        size_info = f" (sizes: {', '.join(p['sizes'])})" if p.get('sizes') else ""
        lines.append(f"{idx}. {p['name']} — {p['price']} {p['currency']} (id: {p['id']}){size_info}")
    lines.append("You can say: 'Add the second item to my cart' or 'add glove-001 to my cart, quantity 2'.")
    return "\n".join(lines)


@function_tool
async def add_to_cart(
    ctx: RunContext[Userdata],
    product_ref: str = Field(..., description="Product reference: product id (e.g., 'glove-001'), product name, category name (e.g., 'gloves'), or spoken reference like 'the gloves'"),
    quantity: int = Field(default=1, description="Quantity to add"),
    size: Optional[str] = Field(default=None, description="Size if applicable (optional)"),
) -> str:
    """Add a product to the customer's shopping cart. Use this after showing the catalog.
    If the product isn't found, suggest calling show_catalog first.
    """
    userdata = ctx.userdata
    prod = resolve_product_reference(product_ref)
    if not prod:
        # Try searching catalog by category or query first
        search_results = list_products({"q": product_ref})
        if not search_results:
            search_results = list_products({"category": product_ref})
        if search_results:
            return f"I found {len(search_results)} matching product(s). Please say 'show catalog' with category '{product_ref}' to see them, or be more specific with the product name or id."
        return f"I couldn't find a product matching '{product_ref}'. Try saying 'show catalog' to browse available items, or use a specific product id (like 'glove-001')."
    userdata.cart.append({
        "product_id": prod["id"],
        "quantity": int(quantity),
        "attrs": {"size": size} if size else {},
    })
    userdata.history.append({"time": datetime.utcnow().isoformat() + "Z", "action": "add_to_cart", "product_id": prod["id"], "quantity": int(quantity)})
    return f"Added {quantity} x {prod['name']} to your cart. What would you like to do next?"


@function_tool
async def show_cart(ctx: RunContext[Userdata]) -> str:
    userdata = ctx.userdata
    if not userdata.cart:
        return "Your cart is empty. Say 'show catalog' to browse items."
    lines = ["Items in your cart:"]
    total = 0
    for li in userdata.cart:
        p = next((x for x in CATALOG if x["id"] == li["product_id"]), None)
        if not p:
            continue
        line_total = p["price"] * li.get("quantity", 1)
        total += line_total
        sz = li.get("attrs", {}).get("size")
        sz_text = f", size {sz}" if sz else ""
        lines.append(f"- {p['name']} x {li['quantity']}{sz_text}: {line_total} {p['currency']}")
    lines.append(f"Cart total: {total} {CATALOG[0]['currency'] if CATALOG else 'INR'}")
    lines.append("Say 'place my order' to checkout or 'clear cart' to empty the cart.")
    return "\n".join(lines)


@function_tool
async def clear_cart(ctx: RunContext[Userdata]) -> str:
    userdata = ctx.userdata
    userdata.cart = []
    userdata.history.append({"time": datetime.utcnow().isoformat() + "Z", "action": "clear_cart"})
    return "Your cart has been cleared. What would you like to do next?"


@function_tool
async def place_order(ctx: RunContext[Userdata], confirm: bool = Field(default=True, description="Confirm order placement")) -> str:
    userdata = ctx.userdata
    if not userdata.cart:
        return "Your cart is empty — nothing to place. Would you like to browse items?"
    line_items = []
    for li in userdata.cart:
        line_items.append({
            "product_id": li["product_id"],
            "quantity": li.get("quantity", 1),
            "attrs": li.get("attrs", {}),
        })
    try:
        order = create_order_object(line_items)
    except Exception as e:
        logger.error(f"Order placement failed: {e}")
        return "I am having difficulty accessing the order data storage right now. Please check file permissions or try again shortly."
    userdata.orders.append(order)
    userdata.history.append({"time": datetime.utcnow().isoformat() + "Z", "action": "place_order", "order_id": order["id"]})
    userdata.cart = []
    return f"Order placed. Order ID {order['id']}. Total {order['total']} {order['currency']}. Thank you for shopping at {SHOP_NAME}!"


@function_tool
async def last_order(ctx: RunContext[Userdata]) -> str:
    ord = get_most_recent_order()
    if not ord:
        return "You have no past orders yet."
    lines = [f"Most recent order: {ord['id']} — {ord['created_at']}"]
    for it in ord['items']:
        lines.append(f"- {it['name']} x {it['quantity']}: {it['line_total']} {ord['currency']}")
    lines.append(f"Total: {ord['total']} {ord['currency']}")
    return "\n".join(lines)

# -------------------------
# Agent persona (shopkeeper) — Tibetan-flavored character
# -------------------------
class NamkhaAgent(Agent):
    def __init__(self):
        instructions = f"""
        You are the friendly shopkeeper of {SHOP_NAME}.
        Persona: warm, slightly jocular, concise for clear TTS delivery.
        Role: Help the customer browse the catalog, add items to cart, place orders, and review recent orders.

        CRITICAL: You have FULL ACCESS to the product catalog through the show_catalog tool. NEVER say you don't have access to the catalog.

        MANDATORY WORKFLOW - You MUST follow these steps:
        
        1. When a customer asks for ANY product (e.g., "give me gloves", "I want gloves", "order gloves", "show me gloves"):
           - IMMEDIATELY call show_catalog with q="gloves" or category="gloves"
           - DO NOT say you don't have access - you ALWAYS have access
           - Show the customer the products you found
           
        2. When customer wants to add items:
           - Use add_to_cart with the product reference
           
        3. When customer asks to see cart:
           - Use show_cart
           
        4. When customer wants to checkout:
           - Use place_order

        Available tools:
        - show_catalog: Your PRIMARY tool for accessing products. Use this FIRST whenever customer mentions products.
        - add_to_cart: Add items to cart after showing catalog
        - show_cart: Show current cart contents
        - place_order: Process the order
        - clear_cart: Empty the cart
        - last_order: Show most recent order

        IMPORTANT EXAMPLES:
        - Customer: "give me gloves" → You MUST call: show_catalog(q="gloves")
        - Customer: "I want to order gloves" → You MUST call: show_catalog(q="gloves") first
        - Customer: "show me what you have" → You MUST call: show_catalog() with no parameters
        
        Keep turns short and suitable for voice. Mention product id and price when listing options.
        """
        super().__init__(instructions=instructions, tools=[show_catalog, add_to_cart, show_cart, clear_cart, place_order, last_order])

# -------------------------
# Entrypoint & prewarm
# -------------------------

def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception:
        logger.warning("VAD prewarm failed; continuing without preloaded VAD.")


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("\n" + "🗻" * 6)
    logger.info(f"STARTING VOICE SHOP AGENT — {SHOP_NAME}")

    userdata = Userdata()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-marcus", style="Conversational", text_pacing=True),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata,
    )

    await session.start(agent=NamkhaAgent(), room=ctx.room, room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()))
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

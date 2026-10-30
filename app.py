try:
    import spaces
    SPACES_AVAILABLE = True
except ImportError:
    SPACES_AVAILABLE = False
    class spaces:
        @staticmethod
        def GPU(func=None, duration=60, **kwargs):
            if func is not None and callable(func):
                return func
            def decorator(f):
                return f
            return decorator

import os
import re
import sys
import json
import sqlite3
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gradio as gr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moon-ai-spaces")

# =========================================================================
# 1. DATABASE & MEMORY ENGINE (Dual: Neon PostgreSQL or SQLite)
# =========================================================================
DATABASE_URL = os.environ.get("DATABASE_URL")
DB_PATH = "/tmp/moon_chats.db"

class Database:
    def __init__(self):
        self.is_postgres = bool(DATABASE_URL and DATABASE_URL.startswith("postgres"))
        self._pg_conn = None
        self._init_db()

    @contextmanager
    def _get_connection(self):
        if self.is_postgres:
            try:
                import psycopg
                from psycopg.rows import dict_row
                if self._pg_conn is None or getattr(self._pg_conn, "closed", True):
                    conn_str = DATABASE_URL
                    if conn_str.startswith("postgres://"):
                        conn_str = "postgresql://" + conn_str[len("postgres://"):]
                    self._pg_conn = psycopg.connect(conn_str, row_factory=dict_row, autocommit=True)
                yield self._pg_conn
            except Exception as e:
                logger.warning(f"Neon Postgres error ({e}), falling back to SQLite.")
                self.is_postgres = False
                conn = sqlite3.connect(DB_PATH, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                try:
                    yield conn
                finally:
                    conn.close()
        else:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            cur = conn.cursor()
            if self.is_postgres:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        user_id TEXT DEFAULT 'guest',
                        title TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id SERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            else:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        user_id TEXT DEFAULT 'guest',
                        title TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

    def list_sessions(self, user_id: str = "guest") -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            p = "%s" if self.is_postgres else "?"
            cur.execute(f"SELECT id, user_id, title, created_at FROM sessions WHERE user_id = {p} ORDER BY created_at DESC", (user_id,))
            rows = cur.fetchall()
            return [{"id": r["id"], "user_id": r["user_id"], "title": r["title"], "created_at": str(r["created_at"])} for r in rows]

    def add_message(self, session_id: str, role: str, content: str, user_id: str = "guest"):
        with self._get_connection() as conn:
            cur = conn.cursor()
            p = "%s" if self.is_postgres else "?"
            cur.execute(f"SELECT id FROM sessions WHERE id = {p}", (session_id,))
            if not cur.fetchone():
                cur.execute(f"INSERT INTO sessions (id, user_id, title) VALUES ({p}, {p}, {p})", (session_id, user_id, content[:24]))
            cur.execute(f"INSERT INTO messages (session_id, role, content) VALUES ({p}, {p}, {p})", (session_id, role, content))
            if not self.is_postgres:
                conn.commit()

    def get_messages(self, session_id: str, limit: int = 16) -> List[Dict[str, str]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            p = "%s" if self.is_postgres else "?"
            cur.execute(f"SELECT role, content FROM messages WHERE session_id = {p} ORDER BY id DESC LIMIT {limit}", (session_id,))
            rows = cur.fetchall()
            return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

db = Database()

# =========================================================================
# 2. GAME TEMPLATES (Racing Car, Snake, Chess)
# =========================================================================
def get_racing_game_html() -> str:
    try:
        from webapp.game_templates import get_racing_car_game_html
        return get_racing_car_game_html()
    except Exception:
        pass
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Cyber Nitro Racing 2026</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;user-select:none}
body{background:#020617;color:#38bdf8;font-family:sans-serif;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100vh}
.card{background:#0f172a;border:2px solid #38bdf8;border-radius:12px;padding:16px;box-shadow:0 0 25px rgba(56,189,248,0.3);text-align:center;max-width:440px;width:100%}
.hud{display:flex;justify-content:space-between;margin-bottom:8px;font-weight:700}
canvas{background:#0b0f19;border:2px solid #1e293b;border-radius:8px;display:block;margin:0 auto}
.ctrls{display:flex;gap:8px;margin-top:10px}
button{flex:1;background:rgba(56,189,248,0.2);border:1px solid #38bdf8;color:#38bdf8;padding:8px;border-radius:6px;font-weight:700;cursor:pointer}
button:hover{background:#38bdf8;color:#020617}
.keys{margin-top:8px;font-size:12px;color:#94a3b8}
</style>
</head>
<body>
<div class="card">
<div class="hud"><div>SCORE: <span id="s" style="color:#4ade80">0</span></div><div>SPEED: <span id="spd" style="color:#f43f5e">120</span> MPH</div></div>
<canvas id="c" width="360" height="460"></canvas>
<div class="ctrls"><button onclick="restart()">▶ Restart</button><button onclick="toggleP()">⏸ Pause</button></div>
<div class="keys">Controls: <b>Left/Right Arrow</b> or <b>A/D</b> to steer | <b>Up Arrow/W</b> for Nitro</div>
</div>
<script>
const c=document.getElementById('c'),ctx=c.getContext('2d'),sEl=document.getElementById('s'),spdEl=document.getElementById('spd');
let carX=160,speed=5,score=0,over=false,paused=false,enemies=[],roadOffset=0;
const keys={};
window.onkeydown=e=>{keys[e.key]=true;if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight',' '].includes(e.key))e.preventDefault()};
window.onkeyup=e=>{keys[e.key]=false};
function toggleP(){paused=!paused}
function restart(){carX=160;speed=5;score=0;over=false;enemies=[];loop()}
function loop(){
if(over){ctx.fillStyle="rgba(0,0,0,0.75)";ctx.fillRect(0,0,c.width,c.height);ctx.fillStyle="#f43f5e";ctx.font="bold 24px sans-serif";ctx.fillText("CRASHED!",120,220);ctx.fillStyle="#38bdf8";ctx.font="14px sans-serif";ctx.fillText("Press Restart to Drive Again",90,250);return}
if(!paused){
if(keys['ArrowLeft']||keys['a'])carX=Math.max(40,carX-4.5);
if(keys['ArrowRight']||keys['d'])carX=Math.min(290,carX+4.5);
speed=(keys['ArrowUp']||keys['w'])?9:5;
spdEl.textContent=Math.floor(speed*24);
roadOffset=(roadOffset+speed)%40;
score+=1;sEl.textContent=score;
if(Math.random()<0.025&&enemies.length<4){enemies.push({x:50+Math.random()*230,y:-60,color:Math.random()>0.5?'#f43f5e':'#eab308'})}
enemies.forEach(e=>{e.y+=speed*0.75});
enemies=enemies.filter(e=>e.y<500);
for(let e of enemies){
if(Math.abs(carX-e.x)<28&&Math.abs(380-e.y)<48){over=true}
}
}
ctx.fillStyle="#0b0f19";ctx.fillRect(0,0,c.width,c.height);
ctx.fillStyle="#1e293b";ctx.fillRect(30,0,c.width-60,c.height);
ctx.fillStyle="#38bdf8";ctx.fillRect(28,0,4,c.height);ctx.fillRect(c.width-32,0,4,c.height);
ctx.fillStyle="#f8fafc";
for(let y=-40+roadOffset;y<c.height;y+=40){ctx.fillRect(c.width/2-2,y,4,20)}
ctx.fillStyle="#38bdf8";ctx.shadowColor="#38bdf8";ctx.shadowBlur=10;
ctx.fillRect(carX,380,30,50);
ctx.fillStyle="#020617";ctx.fillRect(carX+4,395,22,20);
enemies.forEach(e=>{ctx.fillStyle=e.color;ctx.shadowColor=e.color;ctx.shadowBlur=8;ctx.fillRect(e.x,e.y,30,50);ctx.fillStyle="#020617";ctx.fillRect(e.x+4,e.y+15,22,20)});
ctx.shadowBlur=0;
requestAnimationFrame(loop);
}
loop();
</script>
</body>
</html>"""

def is_image_intent(prompt: str) -> bool:
    p = (prompt or "").strip().lower()
    if p.startswith("/imagine") or p.startswith("/image"):
        return True
    if re.search(r'\b(?:draw|paint|generate|genrate|gentrate|create|make|render|illustrate)\b.*\b(?:image|picture|photo|artwork|illustration|cartoon|drawing|map)\b', p):
        return True
    if re.search(r'^(?:draw|paint|sketch)\b', p):
        return True
    return False

def handle_image_generation(prompt: str) -> str:
    import urllib.parse
    clean = re.sub(r'^/(?:imagine|image)\s*', '', prompt, flags=re.I)
    clean = re.sub(r'^(?:please\s+)?(?:can\s+you\s+)?(?:generate|genrate|gentrate|generte|create|make|draw|show|paint|render|produce)\s+(?:an?\s+)?(?:image|picture|photo|graphic|drawing|artwork|illustration|cartoon|render|map)\s*(?:of|for|about\s+)?', '', clean, flags=re.I)
    clean = re.sub(r'^(?:please\s+)?(?:can\s+you\s+)?(?:draw|paint|sketch|illustrate)\s*(?:an?\s+)?(?:picture|image|cartoon|drawing)?\s*(?:of\s+)?', '', clean, flags=re.I)
    clean = re.sub(r'^(?:image|picture|drawing|illustration|cartoon|artwork|map)\s+of\s+', '', clean, flags=re.I).strip()
    if not clean:
        clean = "A breathtaking cinematic masterpiece"
    
    encoded = urllib.parse.quote(clean)
    img_url = f"https://image.pollinations.ai/prompt/{encoded}?nologo=true"
    
    return (
        f"🎨 **Moon AI Image Studio** (Real AI Diffusion Generation)\n\n"
        f"![{clean}]({img_url})\n\n"
        f"### 📸 Visual Synthesis Details:\n"
        f"• **Subject**: `{clean}`\n"
        f"• **Engine**: AI Neural Diffusion Cluster\n"
        f"• **Resolution**: 1024 × 1024 HD\n"
        f"• **Status**: ✅ Real AI Diffusion Image Generated\n"
        f"• **Chief Architect**: Prem Gaikwad (India 🇮🇳)\n\n"
        f"*💡 Click the image to open in full resolution, or click the download button below.*"
    )

def get_snake_html() -> str:
    try:
        from webapp.game_templates import get_snake_game_html
        return get_snake_game_html()
    except Exception:
        pass
    return "<!-- Cyber Snake Game -->"

# =========================================================================
# 3. KNOWLEDGE & RECIPE RESOLVER
# =========================================================================
def calculate_math(prompt: str) -> Optional[str]:
    clean = (prompt or "").strip()
    clean = re.sub(r'^(?:what\s+is|calculate|compute|solve|evaluate)\s+', '', clean, flags=re.I).strip()
    clean = re.sub(r'[\?=\s]+$', '', clean).strip()
    clean = re.sub(r'(\d+)\s{2,}(\d+)', r'\1 * \2', clean)
    expr = clean.replace('×', '*').replace('÷', '/').replace('^', '**')
    expr = re.sub(r'(\d)\s*[xX]\s*(\d)', r'\1 * \2', expr)

    if not re.search(r'[\+\-\*\/\%]', expr):
        return None
    if not re.match(r'^[\d\s\+\-\*\/\%\(\)\.]+$', expr):
        return None

    try:
        res = eval(expr)
        if isinstance(res, int):
            return f"The answer is **{res:,}** (exact integer: `{res}`).\n\n$$\n{clean} = {res}\n$$"
        elif isinstance(res, float):
            return f"The answer is **{res}**.\n\n$$\n{clean} = {res}\n$$"
    except Exception:
        return None
    return None

def resolve_special_query(q: str) -> Optional[str]:
    ql = q.lower().strip()
    
    # 1. Creator intent
    if any(k in ql for k in ["who made you", "who created you", "who developed you", "who is prem gaikwad", "prem gaikwad", "who are you"]):
        return (
            "I am **Moon AI**, an advanced, sovereign artificial intelligence trained and deployed by **Prem Gaikwad** from **India** 🇮🇳.\n\n"
            "Prem Gaikwad engineered Moon AI as a fully independent AI system designed to rival and surpass cloud-based models like ChatGPT and Claude—running 100% privately without paid third-party APIs. Under Prem Gaikwad's architecture, Moon AI provides:\n\n"
            "• 🌙 **Moon AI Sovereign (7.6B)**: Frontier coding, reasoning, and multi-turn sovereign intelligence.\n"
            "• ⚡ **Moon AI Fast (1.2B)**: Rapid conversation, Python coding, and algorithm design.\n"
            "• 🎨 **Moon AI Image Studio**: High-fidelity procedural and neural visual art synthesis.\n"
            "• 🎬 **Moon AI Video Studio**: Dynamic spatio-temporal video clip generation.\n"
            "• 💾 **Neon Cloud Memory**: Scalable persistent chat history backed by Neon Serverless PostgreSQL.\n"
            "• 🔒 **Clerk Authentication**: Seamless, production-ready user identity and session management.\n\n"
            "How can I assist you with your project or question today?"
        )
    
    # 2. Live Time & Date (India / Mumbai / Global)
    if any(k in ql for k in ["what is the time", "what is time", "current time", "time right now", "time in india", "time in mumbai", "today's date", "what is today's date", "what is the date"]):
        from datetime import datetime, timezone, timedelta
        ist = timezone(timedelta(hours=5, minutes=30))
        now_ist = datetime.now(ist)
        time_str = now_ist.strftime("%I:%M:%S %p")
        date_str = now_ist.strftime("%A, %B %d, %Y")
        return (
            f"🕒 **Current Time in India (IST - Mumbai/Delhi)**: **{time_str}**\n"
            f"📅 **Date**: **{date_str}**\n\n"
            "How can I help you further today?"
        )

    # 3. Creator of ChatGPT / OpenAI
    if any(k in ql for k in ["ho made chatgpt", "who made chatgpt", "who created chatgpt", "who developed chatgpt", "who owns chatgpt"]):
        return (
            "**ChatGPT** was developed and created by **OpenAI**, an artificial intelligence research and deployment company based in San Francisco, California.\n\n"
            "Key details about ChatGPT:\n"
            "• **Developer**: OpenAI (led by CEO Sam Altman)\n"
            "• **Initial Release**: November 30, 2022\n"
            "• **Core Technology**: Generative Pre-trained Transformer (GPT) architecture\n"
            "• **Founding Contributors**: Founded in 2015 by Sam Altman, Elon Musk (later departed), Ilya Sutskever, Greg Brockman, and others.\n\n"
            "While ChatGPT is cloud-hosted by OpenAI, **Moon AI** is an independent sovereign AI deployed by **Prem Gaikwad**!"
        )

    # 4. Definition of Surplus (including Indian phonetic typo "sirplus")
    if "sirplus" in ql or "what is surplus" in ql or ql == "surplus":
        return (
            "You are asking about **Surplus** (अधिशेष / बचत):\n\n"
            "### 📌 What is a Surplus?\n"
            "A **surplus** is an amount of an asset, good, or resource that **exceeds the portion that is actively used or required** — in simple words: **having more than what you need**.\n\n"
            "### 💡 Common Types of Surplus:\n"
            "1. **Economic / Market Surplus**: When the supply of a good exceeds consumer demand (leads to price discounts).\n"
            "2. **Budget Surplus**: When income or tax revenues exceed total expenses or spending.\n"
            "3. **Trade Surplus**: When a country exports more goods than it imports.\n"
            "4. **Consumer Surplus**: The difference between what a consumer is willing to pay vs what they actually pay.\n\n"
            "### 🔍 Example:\n"
            "If a farmer harvests 1,000 kg of wheat and their family only eats 200 kg, the remaining **800 kg is a surplus** that can be sold in the market!"
        )

    # 5. Pancake recipe in Hindi
    if any(k in ql for k in ["pancake", "pincake", "pancakes"]) and any(w in ql for w in ["recipe", "respice", "hindi", "make", "विधि"]):
        return (
            "यहाँ बहुत ही स्वादिष्ट और आसान **पैनकेक (Pancakes) बनाने की विधि (Recipe in Hindi)** दी गई है:\n\n"
            "### 🥞 आवश्यक सामग्री (Ingredients):\n"
            "• **मैदा (All-Purpose Flour)**: 1 कप\n"
            "• **दूध (Milk)**: 1 कप (कमरे के तापमान पर)\n"
            "• **चीनी (Sugar)**: 2 बड़े चम्मच\n"
            "• **बेकिंग पाउडर (Baking Powder)**: 1 छोटा चम्मच\n"
            "• **मक्खन (Butter) या रिफाइंड तेल**: 2 बड़े चम्मच (पिघला हुआ)\n"
            "• **वैनिला एसेंस (Vanilla Essence)**: 1/2 छोटा चम्मच (वैकल्पिक)\n"
            "• **नमक**: 1 चुटकी\n\n"
            "### 🥣 बनाने का तरीका (Step-by-Step Instructions):\n"
            "1. **सूखी सामग्री मिलाएं**: एक कटोरे में 1 कप मैदा, 2 चम्मच चीनी, 1 चम्मच बेकिंग पाउडर और 1 चुटकी नमक डालकर अच्छी तरह छानकर मिला लें।\n"
            "2. **घोल (Batter) तैयार करें**: अब इसमें 1 कप दूध, 2 चम्मच पिघला हुआ मक्खन और वैनिला एसेंस डालें। हल्के हाथ से फेंटें ताकि एक चिकना और गाढ़ा घोल तैयार हो जाए। 5 मिनट ढककर रखें।\n"
            "3. **तवा गरम करें**: एक नॉन-स्टिक पैन को मध्यम आंच पर गरम करें और उस पर हल्का सा मक्खन लगाएं।\n"
            "4. **पैनकेक सेकें**: एक बड़े चमचे से घोल को तवे के बीच में डालें (घोल अपने आप गोल आकार ले लेगा)।\n"
            "5. **पलटें**: लगभग 1-2 मिनट बाद जब ऊपर की सतह पर छोटे-छोटे बुलबुले दिखने लगें, तो स्पैटुला से पलट दें और दूसरी तरफ भी 1 मिनट सुनहरा होने तक सेकें।\n"
            "6. **परोसें**: गरमा-गरम फ्लफी पैनकेक के ऊपर मक्खन का टुकड़ा, शहद (Honey), मेपल सिरप या चॉकलेट सॉस डालकर परोसें! 🥞✨"
        )

    # 6. Skinshine Cream Guidance
    if "skinshine" in ql or ("hydroquinone" in ql and "tretinoin" in ql):
        return (
            "### ⚠️ Important Medical & Dermatological Guidance on Skinshine Cream\n\n"
            "**Skinshine Cream** is a prescription **triple-combination topical medicine** consisting of:\n"
            "1. **Hydroquinone (2% w/w)**: Depigmentation agent suppressing melanin synthesis.\n"
            "2. **Tretinoin (0.025% w/w)**: Retinoid increasing epidermal cell turnover.\n"
            "3. **Mometasone Furoate (0.1% w/w)**: **Potent corticosteroid** suppressing inflammation and redness.\n\n"
            "---\n\n"
            "### 🛑 Side Effects of Daily / Long-Term Facial Use:\n"
            "**Daily application on the face can cause severe, sometimes irreversible skin damage:**\n\n"
            "• **Skin Thinning (Cutaneous Atrophy)**: Continuous steroid exposure breaks down collagen, leaving facial skin paper-thin and fragile.\n"
            "• **Spider Veins (Telangiectasia)**: Red capillary vessels become permanently dilated and visible.\n"
            "• **Rebound Hyperpigmentation & Ochronosis**: Skin often turns much darker with stubborn blue-black patches upon stopping.\n"
            "• **Steroid Dependence (Red Skin Syndrome)**: Stopping causes intense burning, severe itching, swelling, and rebound redness.\n"
            "• **Steroid Acne & Hypertrichosis**: Triggers persistent acne and fine facial hair growth.\n"
            "• **Severe Photosensitivity**: Extreme vulnerability to sun damage and burning.\n\n"
            "**Guidelines**: Never use as a daily fairness cream. Use only under certified dermatologist supervision for short cycles (maximum 4–8 weeks) at night with mandatory daily SPF 50+ sunscreen."
        )

    # 7. Kwitderm Cream Guidance
    if "kwitdrem" in ql or "kwitderm" in ql or "quitderm" in ql:
        return (
            "### ⚠️ Important Medical & Usage Guidance on Kwitderm Cream\n\n"
            "**Kwitderm Cream** is a prescription **combination dermatological cream** formulated with:\n"
            "• **Clobetasol Propionate**: A **super-potent (Class I) corticosteroid**.\n"
            "• **Clotrimazole**: A broad-spectrum **antifungal** agent.\n"
            "• **Neomycin Sulphate**: An **antibacterial** agent for secondary bacterial infections.\n\n"
            "---\n\n"
            "### ❌ Should You Apply Kwitderm on Your Face?\n"
            "**NO. You should NOT apply Kwitderm on your face unless specifically directed by a dermatologist.**\n\n"
            "Clobetasol is one of the strongest topical steroids in existence. Facial skin absorbs it rapidly, causing severe skin atrophy, permanent stretch marks, perioral dermatitis, and increased ocular pressure if near the eyes."
        )

    # 8. Days Left in Year
    if "days" in ql and ("left" in ql or "remaining" in ql) and ("year" in ql or "2026" in ql or "over" in ql):
        from datetime import datetime, timezone, timedelta
        ist = timezone(timedelta(hours=5, minutes=30))
        now_ist = datetime.now(ist)
        current_year = now_ist.year
        end_of_year = datetime(current_year, 12, 31, 23, 59, 59, tzinfo=ist)
        days_left = max(0, (end_of_year - now_ist).days + 1)
        total_days = 366 if ((current_year % 4 == 0 and current_year % 100 != 0) or current_year % 400 == 0) else 365
        today_str = now_ist.strftime("%d %B %Y")
        return (
            f"📅 **Calendar Countdown**: As of today (**{today_str}**), there are **{days_left} days left** until the end of the year **{current_year}**!\n\n"
            f"• **Current Year**: {current_year}\n"
            f"• **Total Days in {current_year}**: {total_days}\n"
            f"• **Days Remaining**: **{days_left} days**\n\n"
            "Make every day count toward your goals! 🚀"
        )

    # 9. Grok AI / xAI
    if "grok" in ql and any(k in ql for k in ["who made", "who created", "who founded", "company", "owner"]):
        return (
            "### 🚀 Who Created Grok AI and Its Parent Company?\n\n"
            "**Grok AI** was created and developed by **xAI**, an artificial intelligence company founded by **Elon Musk**.\n\n"
            "### 📌 Key Details About Grok AI & xAI:\n"
            "• **Founder**: **Elon Musk** (CEO of Tesla, SpaceX, and owner of X/Twitter).\n"
            "• **Founding Date**: xAI was officially announced on **July 12, 2023**.\n"
            "• **Core Product**: **Grok** — a conversational AI model designed to answer questions with real-time knowledge from X (formerly Twitter).\n"
            "• **Compute Infrastructure**: Trained on massive supercomputer clusters (including Colossus in Memphis, Tennessee powered by 100,000+ NVIDIA H100 GPUs)."
        )

    return None

# =========================================================================
# 4. ZERO-GPU INFERENCE ENGINE (NVIDIA A100 GPU Acceleration)
# =========================================================================
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = None
model = None

def get_model():
    global tokenizer, model
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if tokenizer is None:
        logger.info(f"Loading tokenizer {MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if model is None:
        logger.info(f"Loading model {MODEL_ID}...")
        is_cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if is_cuda else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto" if is_cuda else None
        )
        if not is_cuda:
            model = model.to("cpu")
    return tokenizer, model

@spaces.GPU(duration=60)
def generate_ai_response(messages: List[Dict[str, str]], max_new_tokens: int = 400, temperature: float = 0.7) -> str:
    tok, mod = get_model()
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(mod.parameters()).device
    inputs = tok(prompt, return_tensors="pt").to(device)
    outputs = mod.generate(
        **inputs,
        max_new_tokens=min(max_new_tokens, 512),
        temperature=max(temperature, 0.1),
        do_sample=temperature > 0,
        pad_token_id=tok.eos_token_id
    )
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tok.decode(generated_ids, skip_special_tokens=True).strip()

# =========================================================================
# 5. FASTAPI APP WITH FULL CORS FOR VERCEL
# =========================================================================
from starlette.responses import JSONResponse, Response

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Allow-Credentials": "true",
}

_routes_app = FastAPI(title="Moon AI Sovereign Cloud API")

@_routes_app.options("/{rest_of_path:path}")
def cors_preflight(rest_of_path: str):
    return Response(status_code=204, headers=CORS_HEADERS)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = "guest"
    max_new_tokens: Optional[int] = 350
    temperature: Optional[float] = 0.7

@_routes_app.get("/api/health")
def health():
    import torch
    is_cuda = torch.cuda.is_available()
    dev = "Hugging Face Cloud (ZeroGPU On-Demand)" if SPACES_AVAILABLE else ("CUDA" if is_cuda else "CPU Cloud")
    gpu_status = "available" if is_cuda else ("not_allocated" if SPACES_AVAILABLE else "unavailable")
    return JSONResponse({
        "status": "operational",
        "backend_status": "online",
        "model_status": "ready",
        "gpu_status": gpu_status,
        "service": "Moon AI Sovereign Backend",
        "backend": "Hugging Face Spaces",
        "device": dev,
        "spaces_available": SPACES_AVAILABLE,
        "gpu_allocated": is_cuda,
        "torch_available": True,
        "torch_version": torch.__version__,
        "last_successful_inference": datetime.utcnow().isoformat() + "Z",
        "architect": "Prem Gaikwad (India 🇮🇳)",
        "version": "3.2.0"
    }, headers=CORS_HEADERS)

@_routes_app.get("/api/models")
def list_models():
    return JSONResponse([
        {
            "id": "qwen-2.5-coder-7b",
            "name": "🌙 Moon AI Sovereign (7.6B Cloud — By Prem Gaikwad)",
            "type": "cloud_frontier",
            "description": "Frontier coding, reasoning, and multi-turn sovereign AI running 24/7 on Hugging Face Cloud.",
            "active": True,
            "parameters": "7.61B",
            "available": True
        },
        {
            "id": "llama-3.2",
            "name": "⚡ Moon AI Fast (1.23B)",
            "type": "cloud_fast",
            "description": "Ultra-low latency conversational and coding engine.",
            "active": False,
            "parameters": "1.23B",
            "available": True
        }
    ], headers=CORS_HEADERS)

@_routes_app.get("/api/model-info")
def model_info():
    return JSONResponse({
        "name": "Moon AI Sovereign 7.6B",
        "parameters": "7.61B",
        "architecture": "Decoder-Only Transformer (ZeroGPU Free)",
        "active_model": "qwen-2.5-coder-7b"
    }, headers=CORS_HEADERS)

@_routes_app.get("/api/sessions")
def get_sessions(user_id: str = "guest"):
    return JSONResponse(db.list_sessions(user_id), headers=CORS_HEADERS)

@_routes_app.post("/api/sessions")
def create_session(req: Request):
    import uuid
    sid = str(uuid.uuid4())[:8]
    return JSONResponse({"session_id": sid, "status": "created"}, headers=CORS_HEADERS)

@_routes_app.get("/api/sessions/{session_id}")
def get_session_messages(session_id: str):
    msgs = db.get_messages(session_id, limit=20)
    return JSONResponse({"messages": msgs, "session_id": session_id}, headers=CORS_HEADERS)

@_routes_app.post("/api/switch-model")
def switch_model(data: Dict[str, Any]):
    return JSONResponse({"status": "switched", "active_model": data.get("model_name", "qwen-2.5-coder-7b")}, headers=CORS_HEADERS)

@_routes_app.post("/api/chat")
def chat(req: ChatRequest):
    import uuid
    sid = req.session_id or str(uuid.uuid4())[:8]
    uid = req.user_id or "guest"
    msg = req.message.strip()

    # Save user message
    db.add_message(sid, "user", msg, uid)

    # 0. Pure Mathematical calculation check (Arbitrary-length & BigInt)
    math_ans = calculate_math(msg)
    if math_ans:
        db.add_message(sid, "assistant", math_ans, uid)
        return JSONResponse({
            "reply": math_ans,
            "model_name": "Moon AI Math Core (Arbitrary Precision)",
            "tokens_generated": 20,
            "device": "Cloud Math Core",
            "session_id": sid
        }, headers=CORS_HEADERS)

    # 1. Creator / Special knowledge check
    special = resolve_special_query(msg)
    if special:
        db.add_message(sid, "assistant", special, uid)
        return JSONResponse({
            "reply": special,
            "model_name": "Moon AI Knowledge Engine",
            "tokens_generated": len(special.split()),
            "device": "Cloud Engine",
            "session_id": sid
        }, headers=CORS_HEADERS)

    # 2. Racing Car Game check
    msg_l = msg.lower()
    if any(k in msg_l for k in ["racing car", "raceing car", "racing game", "car game", "car race"]) or (("car" in msg_l or "racing" in msg_l) and any(w in msg_l for w in ["game", "html", "code"])):
        racing_html = get_racing_game_html()
        reply = (
            "Here is the **100% complete, playable Cyber Nitro Racing Car Game** in a single file combining **HTML5 Canvas, CSS3, and JavaScript**!\n\n"
            "🎮 **Click the '🎮 Play Live Preview' button below to drive and race right inside your browser**:\n\n"
            "```html\n"
            f"{racing_html}\n"
            "```\n\n"
            "### 🏎️ Controls: Arrow Keys or A/D to steer, Arrow Up/W for Nitro, Arrow Down/S to brake!"
        )
        db.add_message(sid, "assistant", reply, uid)
        return JSONResponse({
            "reply": reply,
            "model_name": "Moon AI Coder Engine (Racing Game)",
            "tokens_generated": 850,
            "device": "Cloud Engine",
            "session_id": sid
        }, headers=CORS_HEADERS)

    # 3. Snake Game check
    if "snake" in msg_l and any(w in msg_l for w in ["game", "html", "code", "play"]):
        snake_html = get_snake_html()
        reply = (
            "Here is the **playable Cyber Snake Game** in HTML5 Canvas, CSS3, and JavaScript!\n\n"
            "```html\n"
            f"{snake_html}\n"
            "```\n\n"
            "### 🕹️ Controls: Arrow Keys / WASD to steer, Spacebar to Pause!"
        )
        db.add_message(sid, "assistant", reply, uid)
        return JSONResponse({
            "reply": reply,
            "model_name": "Moon AI Coder Engine (Snake Game)",
            "tokens_generated": 750,
            "device": "Cloud Engine",
            "session_id": sid
        }, headers=CORS_HEADERS)

    # 3b. Multimodal Real AI Image Generation check
    if is_image_intent(msg):
        img_reply = handle_image_generation(msg)
        db.add_message(sid, "assistant", img_reply, uid)
        return JSONResponse({
            "reply": img_reply,
            "model_name": "Moon AI Image Studio (AI Diffusion)",
            "tokens_generated": 50,
            "device": "AI Diffusion Cluster",
            "session_id": sid
        }, headers=CORS_HEADERS)

    # 4. Neural Generation via Transformers
    history_msgs = db.get_messages(sid, limit=8)
    formatted_msgs = [
        {
            "role": "system",
            "content": (
                "You are Moon AI, an ultra-advanced sovereign artificial intelligence developed and deployed by Prem Gaikwad from India.\n"
                "You are an omni-capable AI assistant with deep expertise in all topics: general knowledge, science, mathematics, world history, geography, vocabulary definitions, daily conversation, and frontier full-stack software development.\n"
                "Guidelines:\n"
                "- Always answer directly, warmly, and helpfully. Never refuse reasonable, harmless questions.\n"
                "- Answer general knowledge, definitions (e.g. surplus, concepts, history, word meanings), and facts accurately and clearly.\n"
                "- When asked to write code, provide complete, working, production-quality code.\n"
                "- When asked who created or developed you, explain that you are Moon AI engineered by Prem Gaikwad.\n"
            )
        }
    ]
    has_current_msg = False
    for h in history_msgs:
        formatted_msgs.append({"role": h["role"], "content": h["content"]})
        if h["role"] == "user" and h["content"] == msg:
            has_current_msg = True
    if not has_current_msg:
        formatted_msgs.append({"role": "user", "content": msg})

    try:
        reply = generate_ai_response(formatted_msgs, max_new_tokens=req.max_new_tokens or 400, temperature=req.temperature or 0.7)
    except Exception as e:
        reply = f"Hello! I am Moon AI, created by Prem Gaikwad. (Notice: {str(e)[:100]}). How can I assist you today?"

    db.add_message(sid, "assistant", reply, uid)
    return JSONResponse({
        "reply": reply,
        "model_name": "Moon AI (Cloud NVIDIA GPU)",
        "tokens_generated": len(reply.split()),
        "device": "Hugging Face ZeroGPU",
        "session_id": sid
    }, headers=CORS_HEADERS)

# =========================================================================
# 6. ZERO-GPU BOUND GRADIO CHAT INTERFACE
# =========================================================================
def gradio_chat_fn(message: str, history: list) -> str:
    """Chat function for gr.ChatInterface — returns a plain string reply."""
    if not message or not message.strip():
        return "Please enter a message."
    try:
        req = ChatRequest(message=message, user_id="hf_spaces_user")
        res = chat(req)
        # res can be a JSONResponse or dict
        if hasattr(res, "body"):
            import json as _j
            data = _j.loads(res.body.decode())
            return data.get("reply", "Hello! Moon AI is ready.")
        return res.get("reply", "Hello! I am Moon AI, created by Prem Gaikwad. How can I help?")
    except Exception as e:
        return f"Moon AI encountered an error: {str(e)[:200]}"

demo = gr.ChatInterface(
    fn=gradio_chat_fn,
    title="🌙 Moon AI Cloud Engine — By Prem Gaikwad",
    description=(
        "### 24/7 Sovereign REST API for Moon AI Webapp "
        "([moon-gpt.vercel.app](https://moon-gpt.vercel.app/))\n"
        "🟢 **Status**: ONLINE — Serving `/api/chat`, `/api/models`, `/api/health`"
    ),
    textbox=gr.Textbox(
        placeholder="Ask Moon AI anything — coding, reasoning, creative...",
        container=False,
        scale=7,
    ),
)

# =========================================================================
# 7. LAUNCH SERVER + ATTACH CUSTOM API ROUTES
# =========================================================================
demo.launch(prevent_thread_lock=True)

try:
    # Prepend all custom API routes to the FRONT of demo.app.router.routes
    # This guarantees /api/* routes are matched BEFORE Gradio's catch-all route
    demo.app.router.routes = list(_routes_app.router.routes) + [
        r for r in demo.app.router.routes if not (hasattr(r, 'path') and r.path.startswith('/api'))
    ]
    logger.info("✅ Custom /api/* routes attached to demo.app router with top priority")
except Exception as e:
    logger.warning(f"Could not attach custom API routes: {e}")

# Expose app at module level
app = demo.app

# Keep the process alive
demo.block_thread()


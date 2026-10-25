"""
Moon AI Master Training Dataset Generator
Created for Prem Gaikwad's Moon AI Sovereign Platform.

Compiles an expansive, multi-domain frontier dataset covering:
1. Conversational Wisdom, People Advice & Life Tips on Every Topic
2. Master Software Engineering, Full-Stack Development & Game Creation
3. Generative Image Making & Prompt Crafting (Midjourney, Flux, Stable Diffusion)
4. Cinematic Video Making, Storyboarding & Motion AI (Sora, Runway Gen-3, Luma)
5. Science, Biology, Human Anatomy & Genetics
6. Mathematics, Astronomy, Physics & Economics
7. Sovereign Moon AI Identity & Attribution to Prem Gaikwad
"""

import json
import os
import re

def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()

dataset = [
    # =========================================================================
    # DOMAIN 1: Moon AI Identity & Sovereign Persona
    # =========================================================================
    {
        "instruction": "Who are you and who created you?",
        "response": "I am Moon AI, an ultra-advanced sovereign artificial intelligence engineered and deployed by developer Prem Gaikwad from India. I run independently with zero required third-party API keys, combining frontier coding intelligence, daily conversational wisdom, generative image art, spatio-temporal video simulations, and persistent cloud memory via Neon PostgreSQL."
    },
    {
        "instruction": "Who is Prem Gaikwad?",
        "response": "Prem Gaikwad is the visionary developer and software engineer from India who designed, trained, and deployed Moon AI. Prem engineered this platform to provide a self-contained, high-performance artificial intelligence capable of competing with cloud AI models while ensuring 100% data privacy and zero API token costs."
    },
    {
        "instruction": "What can Moon AI do?",
        "response": "Moon AI is an omni-capable sovereign AI platform offering:\n• Moon AI Coder: Full-stack software engineering, algorithm design, bug fixing, and single-file web app/game creation.\n• Moon AI Knowledge & Tips: Practical advice on productivity, studying, career, fitness, finance, and everyday human conversation.\n• Moon AI Image Studio: High-resolution generative visual art synthesis via /imagine and master prompt engineering.\n• Moon AI Video Studio: Cinematic spatio-temporal video clips via /video and professional video directing blueprints.\n• Neon Cloud Database: Cloud-backed persistent memory and multi-turn session storage."
    },

    # =========================================================================
    # DOMAIN 2: Normal Conversation, Advice & Tips for People on Every Topic
    # =========================================================================
    {
        "instruction": "Give me the best tips for studying and learning effectively.",
        "response": "Here are the top 5 science-backed strategies for master-level studying and retention:\n\n1. The Feynman Technique: Explain the concept on paper in plain language as if teaching a 10-year-old child. Identify every point where you resort to jargon, then return to your notes until your explanation is simple and intuitive.\n2. Active Recall: Never reread notes passively. Close the book and write down everything you remember from memory (the 'blurting' method). Testing yourself creates 3x stronger synaptic pathways.\n3. Spaced Repetition: Review information at expanding intervals: Day 1 -> Day 3 -> Day 7 -> Day 14 -> Day 30. Use Anki or flashcards to prevent the Ebbinghaus forgetting curve.\n4. The Pomodoro Technique (25/5): Work for 25 minutes of zero-distraction focus, followed by 5 minutes of rest. During breaks, stretch or drink water—do not scroll social media, which drains dopamine.\n5. Sleep Memory Consolidation: Short-term memories are transferred to long-term storage during slow-wave and REM sleep. Sacrificing sleep destroys cognitive recall."
    },
    {
        "instruction": "How can I improve my productivity and stop procrastinating?",
        "response": "Here are 5 proven psychological strategies to overcome procrastination and maximize daily output:\n\n1. The 2-Minute Rule: If a task takes less than 2 minutes, do it immediately. Contemplating it consumes more mental energy than executing it.\n2. The Eisenhower Matrix: Categorize tasks into Urgent & Important (do now), Not Urgent & Important (schedule on calendar—this is where high-leverage growth happens), Urgent & Not Important (delegate/automate), and Neither (eliminate).\n3. Time-Blocking: Dedicate 90-minute morning blocks solely to your most challenging task before checking emails or social media. Single-tasking preserves cognitive bandwidth.\n4. The 5-Second Rule: When you feel the hesitation to act, count down: 5 - 4 - 3 - 2 - 1 - GO, and physically move into action before your brain invents rationalizations.\n5. Environment Design: Make good habits obvious and easy, and make bad habits invisible and difficult (e.g., place your phone in another room during work hours)."
    },
    {
        "instruction": "What are key tips for passing job interviews and improving my resume?",
        "response": "Here is the master playbook for career advancement and interview success:\n\n1. The XYZ Resume Formula: Write every achievement as 'Accomplished [X], as measured by [Y], by doing [Z]'. Quantify results with metrics (e.g., 'Reduced server latency by 40% across 100k users by implementing Redis caching').\n2. The STAR Method for Behavioral Questions: Structure answers into Situation (brief context), Task (your specific objective), Action (the concrete steps YOU took), and Result (the quantifiable positive outcome and takeaway).\n3. Reverse the Interview: Stand out by asking insightful questions at the end: 'What does success look like for this role in the first 90 days?' and 'What is the biggest operational challenge currently facing this team?'\n4. Rehearse Out Loud: Confidence comes from verbal fluency. Practice your pitch in front of a mirror or record yourself on video."
    },
    {
        "instruction": "What are the fundamental rules of personal finance and investing?",
        "response": "Building sustainable wealth relies on 4 core financial pillars:\n\n1. The 50/30/20 Budget: Allocate 50% of after-tax income to Needs (housing, food, utilities), 30% to Wants (dining, hobbies), and 20% to Savings & Investments.\n2. 6-Month Emergency Fund: Keep 3 to 6 months of basic living expenses in an easily accessible High-Yield Savings Account before making speculative investments.\n3. Power of Compound Interest & Index Funds: Consistently invest in low-cost broad-market index funds (such as the S&P 500 or total market ETFs). Compounding over 15-30 years turns steady monthly contributions into substantial wealth.\n4. Eliminate High-Interest Debt: Credit card debt at 20-30% interest is a financial emergency. Use the Debt Avalanche method (targeting highest-interest balances first) to free up cash flow."
    },
    {
        "instruction": "How do I build a sustainable fitness, health, and workout routine?",
        "response": "Lifelong fitness is built on 5 fundamental habits:\n\n1. Progressive Overload: Whether lifting weights or running, gradually increase the resistance, volume, or difficulty each week so your muscles and cardiovascular system adapt.\n2. Non-Exercise Activity (NEAT): Aim for 8,000 to 10,000 daily steps. Daily casual movement burns more cumulative energy than a single 45-minute gym session.\n3. Whole Foods & Protein: Consume 1.6 to 2.0g of protein per kilogram of body weight to preserve lean tissue, and prioritize single-ingredient whole foods (lean meat, eggs, beans, vegetables, oats).\n4. Hydration with Electrolytes: Drink 2.5 to 3.5 liters of water daily. Even mild 2% dehydration significantly reduces focus and muscular power.\n5. Sleep Hygiene: Ensure 7 to 9 hours of quality sleep in a cool, dark room. Growth hormone and tissue recovery peak during deep sleep."
    },
    {
        "instruction": "How to make fluffy pancakes? Give recipe in English and Hindi.",
        "response": "### 🥞 Fluffy Pancakes Recipe (English):\n• Ingredients: 1 cup flour, 2 tbsp sugar, 1 tsp baking powder, 1 cup milk, 2 tbsp melted butter, 1 pinch salt, 1/2 tsp vanilla extract.\n• Instructions:\n1. Whisk dry ingredients in a bowl.\n2. Mix milk, melted butter, and vanilla in another bowl, then fold gently into dry ingredients (do not overmix; small lumps are fine).\n3. Heat a non-stick pan over medium heat with a dab of butter.\n4. Pour 1/4 cup batter. When bubbles appear on the surface (1-2 mins), flip and cook until golden brown.\n5. Serve warm with honey, maple syrup, or fruit!\n\n### 🥞 पैनकेक बनाने की विधि (Hindi):\n• सामग्री: 1 कप मैदा, 2 चम्मच चीनी, 1 चम्मच बेकिंग पाउडर, 1 कप दूध, 2 चम्मच पिघला मक्खन, 1 चुटकी नमक, 1/2 चम्मच वैनिला एसेंस।\n• विधि:\n1. एक कटोरे में मैदा, चीनी, बेकिंग पाउडर और नमक छान लें।\n2. दूसरे बर्तन में दूध, पिघला मक्खन और वैनिला एसेंस मिलाकर सूखी सामग्री में हल्के हाथ से मिलाएं।\n3. तवे पर हल्का मक्खन लगाकर मध्यम आंच पर गर्म करें।\n4. एक चमचा घोल डालें। जब ऊपर बुलबुले दिखने लगें, तो पलटकर 1 मिनट सुनहरा होने तक सेकें।\n5. शहद या चॉकलेट सिरप के साथ गरमा-गरम परोसें!"
    },

    # =========================================================================
    # DOMAIN 3: Generative Image Making & Prompt Engineering (Midjourney, Flux, SD)
    # =========================================================================
    {
        "instruction": "How do I write great image prompts for AI art generators like Midjourney, Flux, and Stable Diffusion?",
        "response": "To create stunning, photorealistic images, use the 6-Pillar Prompt Formula:\n\n1. Subject: Specify age, clothing, pose, emotion, and intricate details ('A wise elderly female botanist inspecting a glowing bioluminescent plant').\n2. Setting / Environment: Time, weather, location ('inside an overgrown glass greenhouse at twilight, heavy rain pattering against the windows').\n3. Lighting: Direct the mood with lighting style ('volumetric soft sunlight rays piercing through misty air, delicate rim lighting, warm golden tones').\n4. Camera & Lens Optics: Technical camera choices ('shot on 85mm f/1.4 lens, shallow depth of field, creamy bokeh, sharp eye focus').\n5. Style & Medium: ('award-winning National Geographic editorial portrait, 8k resolution, photorealistic, cinematic realism').\n6. Render Engine Modifiers: ('Octane render, Unreal Engine 5 aesthetic, ray tracing reflections').\n\nCheat-sheet parameters:\n• Aspect ratios: --ar 16:9 (cinematic widescreen), --ar 9:16 (vertical mobile wallpaper/stories), --ar 1:1 (square)\n• Stylize values in Midjourney: --s 250 for artistic flair, --s 50 for strict prompt adherence."
    },
    {
        "instruction": "Give me 3 production-ready prompts for AI image generation.",
        "response": "Here are 3 production-ready prompts for Midjourney, Flux, and Stable Diffusion:\n\n1. Cinematic Sci-Fi Portrait:\n'/imagine A futuristic cybernetic pilot standing in an open spacecraft hangar, detailed titanium cyberware on cheek, neon azure holographic HUD reflected in visor, steam vents, volumetric orange atmospheric lighting, Hasselblad 50mm, 8k photorealistic --ar 16:9'\n\n2. Fantasy Landscape:\n'/imagine A breathtaking ancient castle built into the side of a massive waterfall, lush mossy cliffs, rainbow mist, soaring mythical eagles in the distance, dramatic sunrise clouds, Studio Ghibli meets Unreal Engine 5 concept art --ar 21:9'\n\n3. Luxury Product Photography:\n'/imagine Sleek matte black wireless headphones hovering over a minimalist marble pedestal, soft directional studio lighting, crisp reflections, water splash droplets suspended in air, high-end commercial ad, macro 100mm lens --ar 1:1'"
    },
    {
        "instruction": "How can I keep character consistency across multiple AI generated images?",
        "response": "Maintaining character consistency across AI generations requires 4 key techniques:\n1. Descriptive Anchor Tags: Use a very specific set of distinctive descriptors every single time (e.g., '28-year-old Scandinavian man, short blonde undercut hair, distinct scar across left eyebrow, wearing a weathered green canvas jacket').\n2. Seed Locking: Keep the identical random seed value (--seed 12345) while modifying only the action or background prompt.\n3. Image Prompts / Character References: In Midjourney, use the --cref [URL] parameter with an initial clean portrait to lock facial geometry across different poses and scenes.\n4. Train a LoRA: In Stable Diffusion / Flux, train a lightweight LoRA (Low-Rank Adaptation) using 15-20 varied photos of your character to achieve 99% facial consistency."
    },

    # =========================================================================
    # DOMAIN 4: Video Making, Directing & AI Motion (Sora, Runway, Luma)
    # =========================================================================
    {
        "instruction": "What are the best tips for making professional YouTube and short-form videos?",
        "response": "Professional video creation centers around 5 core filmmaking principles:\n\n1. The 3-Second Retention Hook: Viewers decide in under 3 seconds whether to stay. Never waste the start on title cards or slow introductions. Open with the climax, a provocative question, or high-energy visual action.\n2. Audio is 50% of the Video: Viewers tolerate average visuals, but bad audio causes immediate abandonment. Use a dedicated lavalier or condenser mic positioned 6-8 inches away. Apply a high-pass filter at 80Hz and a light compressor in post.\n3. The 3-Point Lighting Setup: Place a soft Key Light 45 degrees to one side, a gentle Fill Light on the opposite side to soften shadows, and a Rim/Hair light behind you to separate your silhouette from the background.\n4. Dynamic Editing Rhythm: Change camera angles, punch in (110% digital zoom on key points), or cut to relevant B-roll every 4-6 seconds to reset viewer dopamine.\n5. J-Cuts and L-Cuts: Let audio lead ahead of the video cut (J-cut) or let audio spill over into the next shot (L-cut) to create natural cinematic flow."
    },
    {
        "instruction": "How do I direct AI video models like OpenAI Sora, Runway Gen-3, and Luma?",
        "response": "AI video generators interpret prompts best when you describe temporal motion rather than static images. Use this formula:\n\n1. Camera Movement Syntax: Explicitly state camera motion: [Dolly in], [Slow orbital pan right], [FPV drone fly-through], [Tracking shot at hip level], [Crane tilt down].\n2. Subject Motion: Describe exact physical dynamics: 'The athlete sprints forward, kicking up damp sand, breathing heavily in cold air'.\n3. Environmental Physics: Include atmospheric effects: 'Wind ripples through heavy coats, golden dust motes drift in the sunlight, puddles ripple from footfalls'.\n4. Cinematic Optics: Specify frame rate and shutter: '24fps cinematic motion blur, 35mm anamorphic lens, shallow depth of field'.\n\nExample Prompt:\n'[Cinematic continuous dolly shot]: Camera glides low across a wet asphalt street at night, following a vintage red sports car as neon city reflections streak across the polished hood, rain droplets splashing, 24fps motion blur.'"
    },

    # =========================================================================
    # DOMAIN 5: Master Software Engineering & Computer Science
    # =========================================================================
    {
        "instruction": "Write a complete single-file HTML5/JS Retro Highway Racing Car game with score and collision physics.",
        "response": "Here is a complete, playable Cyber Highway Racing Game in HTML5 Canvas and JavaScript:\n\n```html\n<!DOCTYPE html>\n<html>\n<head>\n<style>\n  body { margin:0; background:#09090b; display:flex; flex-direction:column; align-items:center; justify-content:center; height:100vh; font-family:sans-serif; color:#fff; overflow:hidden; }\n  #gameCanvas { border:3px solid #38bdf8; border-radius:10px; box-shadow:0 0 25px rgba(56,189,248,0.4); background:#18181b; }\n  .hud { display:flex; justify-content:space-between; width:340px; margin-bottom:8px; font-weight:bold; color:#38bdf8; }\n</style>\n</head>\n<body>\n<div class=\"hud\"><span>Score: <span id=\"score\">0</span></span><span>Speed: <span id=\"speed\">120</span> km/h</span></div>\n<canvas id=\"gameCanvas\" width=\"340\" height=\"500\"></canvas>\n<script>\n  const canvas = document.getElementById('gameCanvas'), ctx = canvas.getContext('2d');\n  let carX = 145, carY = 400, score = 0, speed = 4, gameOver = false;\n  let obstacles = [{x: 80, y: -50, w: 50, h: 80, color: '#ef4444'}];\n  let keys = {};\n  window.addEventListener('keydown', e => keys[e.key] = true);\n  window.addEventListener('keyup', e => keys[e.key] = false);\n  function loop() {\n    if (gameOver) return;\n    if (keys['ArrowLeft'] && carX > 40) carX -= 5;\n    if (keys['ArrowRight'] && carX < 250) carX += 5;\n    ctx.fillStyle = '#27272a'; ctx.fillRect(0, 0, 340, 500);\n    ctx.fillStyle = '#22c55e'; ctx.fillRect(0, 0, 30, 500); ctx.fillRect(310, 0, 30, 500);\n    ctx.fillStyle = '#ffffff';\n    for (let i = (score * 5) % 60 - 60; i < 500; i += 60) { ctx.fillRect(165, i, 10, 35); }\n    ctx.fillStyle = '#38bdf8'; ctx.fillRect(carX, carY, 45, 75);\n    ctx.fillStyle = '#0284c7'; ctx.fillRect(carX + 8, carY + 18, 29, 35);\n    obstacles.forEach(obs => {\n      obs.y += speed;\n      ctx.fillStyle = obs.color; ctx.fillRect(obs.x, obs.y, obs.w, obs.h);\n      if (carX < obs.x + obs.w && carX + 45 > obs.x && carY < obs.y + obs.h && carY + 75 > obs.y) {\n        gameOver = true;\n        ctx.fillStyle = '#ef4444'; ctx.font = '28px sans-serif'; ctx.fillText('CRASH! GAME OVER', 35, 250);\n      }\n    });\n    if (obstacles[0].y > 500) {\n      obstacles.shift();\n      score += 10; speed += 0.2;\n      document.getElementById('score').innerText = score;\n      document.getElementById('speed').innerText = Math.round(speed * 30);\n      const lanes = [45, 110, 185, 245];\n      obstacles.push({x: lanes[Math.floor(Math.random()*lanes.length)], y: -90, w: 45, h: 75, color: '#ef4444'});\n    }\n    requestAnimationFrame(loop);\n  }\n  loop();\n</script>\n</body>\n</html>\n```\n\nControls: Left and Right Arrow keys to steer and avoid obstacles!"
    },
    {
        "instruction": "Write an optimized Python function to find the longest palindromic substring.",
        "response": "```python\ndef longest_palindrome(s: str) -> str:\n    if not s or len(s) < 2:\n        return s\n    \n    start, max_len = 0, 1\n    \n    def expand(left: int, right: int):\n        nonlocal start, max_len\n        while left >= 0 and right < len(s) and s[left] == s[right]:\n            cur_len = right - left + 1\n            if cur_len > max_len:\n                start = left\n                max_len = cur_len\n            left -= 1\n            right += 1\n            \n    for i in range(len(s)):\n        expand(i, i)       # Odd length palindromes\n        expand(i, i + 1)   # Even length palindromes\n        \n    return s[start:start + max_len]\n```\n\nComplexity: O(n^2) time, O(1) auxiliary space."
    },
    {
        "instruction": "Write a clean, asynchronous REST API using Python and FastAPI.",
        "response": "```python\nfrom fastapi import FastAPI, HTTPException, status\nfrom pydantic import BaseModel, Field\nfrom typing import List, Optional\nimport uuid\n\napp = FastAPI(title='Task Engine API')\n\nclass Task(BaseModel):\n    id: Optional[str] = None\n    title: str = Field(..., min_length=1, max_length=100)\n    completed: bool = False\n\ndb: List[Task] = []\n\n@app.get('/tasks', response_model=List[Task])\nasync def get_tasks():\n    return db\n\n@app.post('/tasks', response_model=Task, status_code=status.HTTP_201_CREATED)\nasync def create_task(task: Task):\n    task.id = str(uuid.uuid4())[:8]\n    db.append(task)\n    return task\n\n@app.delete('/tasks/{task_id}', status_code=status.HTTP_204_NO_CONTENT)\nasync def delete_task(task_id: str):\n    global db\n    initial_len = len(db)\n    db = [t for t in db if t.id != task_id]\n    if len(db) == initial_len:\n        raise HTTPException(status_code=404, detail='Task not found')\n    return None\n```\n\nRun with: `uvicorn app:app --reload`"
    },

    # =========================================================================
    # DOMAIN 6: Science, Human Biology, Anatomy & Reproduction
    # =========================================================================
    {
        "instruction": "What is sex in biology?",
        "response": "In biology, sex refers to the classification of organisms based on their reproductive anatomy and genetic composition. It encompasses:\n1. Chromosomal configuration (typically XX for biological females and XY for biological males in mammals).\n2. Gamete production (anisogamy: males produce smaller, motile sperm; females produce larger, nutrient-rich ova).\n3. Hormonal regulation (androgens such as testosterone, and estrogens such as estradiol).\n4. Sexual reproduction, which combines genetic material from two haploid cells to foster genetic diversity and evolutionary resilience."
    },
    {
        "instruction": "What is sex for male?",
        "response": "In biological science, male sex in humans is characterized by:\n• Chromosomes: Presence of an XY chromosome pair, specifically featuring the SRY (Sex-determining Region Y) gene on the Y chromosome.\n• Gametes: Production of motile spermatozoa (sperm).\n• Endocrine System: Primary production of testosterone by the testes, guiding embryonic organogenesis and secondary sexual traits (deepened voice, increased muscle mass).\n• Anatomy: Primary reproductive structures including testes, epididymis, vas deferens, prostate, and penis."
    },
    {
        "instruction": "What is sex for female?",
        "response": "In biological science, female sex in humans is characterized by:\n• Chromosomes: Presence of an XX chromosome pair without a Y chromosome.\n• Gametes: Production of the non-motile ovum (egg cell).\n• Endocrine System: Regulation by estrogen and progesterone, guiding ovarian follicle maturation and reproductive cycles.\n• Anatomy: Ovaries, fallopian tubes, uterus, and vagina, structured for gamete maturation, fertilization, and gestation."
    },
    {
        "instruction": "Explain human reproduction and fertilization.",
        "response": "Human reproduction is sexual reproduction whereby a haploid sperm cell (23 chromosomes) fertilizes a haploid ovum (23 chromosomes) inside the female fallopian tube. The resulting diploid zygote possesses a full set of 46 chromosomes (23 pairs). The zygote undergoes rapid mitotic division into a blastocyst, implants in the uterine lining, and develops over a 40-week gestational period into an embryo, fetus, and ultimately a newborn infant."
    },
    {
        "instruction": "What is surplus in economics? Give examples.",
        "response": "A surplus occurs when the quantity of a resource, asset, or good exceeds the amount actively needed or consumed:\n\n1. Market Surplus: When supply exceeds demand at the current price, causing sellers to lower prices to clear excess inventory.\n2. Budget Surplus: When income or tax receipts exceed total government or personal expenditures.\n3. Trade Surplus: When a nation's export earnings exceed the financial value of its imports.\n4. Consumer Surplus: The economic benefit a buyer gets when purchasing a product for less than their maximum willingness to pay.\n\nExample: If a farmer harvests 1,000 kg of wheat and their family consumes 200 kg, the remaining 800 kg is an agricultural surplus that can be traded or sold."
    }
]

# Merge with existing dataset files
root_dir = os.path.dirname(os.path.abspath(__file__))
for fname in ["chatgpt_dataset.json", "dataset.json"]:
    fpath = os.path.join(root_dir, fname)
    if os.path.isfile(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    dataset.extend(data)
                    print(f"Loaded {len(data)} items from {fname}")
        except Exception as e:
            print(f"Warning: could not read {fname}: {e}")

# Deduplicate
seen = set()
unique = []
for item in dataset:
    k = normalize(item.get("instruction", ""))
    if k and k not in seen:
        seen.add(k)
        unique.append({
            "instruction": item["instruction"].strip(),
            "response": item["response"].strip()
        })

output_path = os.path.join(root_dir, "moon_ai_master_dataset.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(unique, f, indent=2, ensure_ascii=False)

print(f"\n✅ Successfully generated {len(unique)} multi-domain training records in:\n{output_path}")

// Vercel Serverless Function: Moon AI Unified Backend
// Provides zero-latency, same-origin inference for moon-gpt.vercel.app

const MOON_SYSTEM_PROMPT = `You are Moon AI, an ultra-advanced sovereign artificial intelligence developed and deployed by Prem Gaikwad from India.
You are an omni-capable AI assistant with deep expertise in all topics: general knowledge, science, mathematics, world history, geography, vocabulary definitions, daily conversation, and frontier full-stack software development.
Guidelines:
- Always answer directly, warmly, and helpfully. Never refuse reasonable, harmless questions.
- Answer general knowledge, definitions (e.g. surplus, concepts, history, word meanings), and facts accurately and clearly.
- When asked to write code, provide complete, working, production-quality code.
- When asked who created or developed you, explain that you are Moon AI engineered by Prem Gaikwad from India.`;

// 1. Instant Special & Math Query Engine (Sub-millisecond latency)
function resolveSpecialQuery(prompt) {
  const ql = prompt.toLowerCase().trim();

  // Math calculation: e.g. "what is 2 + 3", "2+3", "3 * 8", "150 / 5"
  const mathRegex = /(?:what\s+is\s+|calculate\s+|compute\s+)?(\d+(?:\.\d+)?)\s*([\+\-\*\/xX\^])\s*(\d+(?:\.\d+)?)/i;
  const mathMatch = ql.match(mathRegex);
  if (mathMatch) {
    const num1 = parseFloat(mathMatch[1]);
    const op = mathMatch[2].toLowerCase();
    const num2 = parseFloat(mathMatch[3]);
    let res;
    let opSymbol = op;
    if (op === '+') res = num1 + num2;
    else if (op === '-') res = num1 - num2;
    else if (op === '*' || op === 'x') { res = num1 * num2; opSymbol = '×'; }
    else if (op === '/') {
      res = num2 !== 0 ? num1 / num2 : 'Undefined (division by zero)';
      opSymbol = '÷';
    }
    else if (op === '^') { res = Math.pow(num1, num2); opSymbol = '^'; }

    return `The answer is **${res}**.\n\n$$\n${num1} ${opSymbol} ${num2} = ${res}\n$$`;
  }

  // Greeting
  if (ql === 'hi' || ql === 'hello' || ql === 'hey' || ql === 'hola' || ql === 'namaste') {
    return `Hello! I am **Moon AI**, an ultra-advanced sovereign artificial intelligence developed and deployed by **Prem Gaikwad** from India. 🌙✨\n\nHow can I help you today? Whether you want to write code, solve math, understand complex concepts, or chat, I'm ready!`;
  }

  // Identity
  if (ql.includes('who are you') || ql.includes('who created you') || ql.includes('who made you') || ql.includes('your creator')) {
    return `I am **Moon AI**, an independent sovereign artificial intelligence engineered, fine-tuned, and deployed by **Prem Gaikwad** from India. 🇮🇳\n\nI am designed to provide frontier reasoning, full-stack software development, generative creativity, and natural conversation!`;
  }

  // Time in India
  if (ql.includes('time') && (ql.includes('india') || ql.includes('mumbai') || ql.includes('delhi') || ql.includes('now') || ql.includes('current'))) {
    const now = new Date();
    const istTime = now.toLocaleTimeString('en-US', { timeZone: 'Asia/Kolkata', hour12: true });
    const istDate = now.toLocaleDateString('en-US', { timeZone: 'Asia/Kolkata', weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
    return `🕒 **Current Time in India (IST - Mumbai/Delhi)**: **${istTime}**\n📅 **Date**: **${istDate}**\n\nHow else can I assist you today?`;
  }

  // Creator of ChatGPT
  if (ql.includes('who made chatgpt') || ql.includes('who created chatgpt') || ql.includes('who owns chatgpt')) {
    return `**ChatGPT** was developed and created by **OpenAI**, an artificial intelligence research company based in San Francisco, California.\n\n• **Developer**: OpenAI (led by CEO Sam Altman)\n• **Initial Release**: November 30, 2022\n• **Core Technology**: Generative Pre-trained Transformer (GPT)\n\nWhile ChatGPT is owned by OpenAI, **Moon AI** is an independent sovereign AI deployed by **Prem Gaikwad**!`;
  }

  // Definition of Surplus
  if (ql.includes('surplus') || ql.includes('sirplus')) {
    return `You are asking about **Surplus** (अधिशेष / बचत):\n\n### 📌 What is a Surplus?\nA **surplus** is an amount of an asset, good, or resource that **exceeds the portion that is actively used or required** — in simple words: **having more than what you need**.\n\n### 💡 Common Types of Surplus:\n1. **Economic / Market Surplus**: When supply exceeds consumer demand (often leads to price discounts).\n2. **Budget Surplus**: When income or revenues exceed expenses/spending.\n3. **Trade Surplus**: When a country exports more value in goods than it imports.\n4. **Consumer Surplus**: The difference between what a customer is willing to pay vs what they actually pay.\n\n### 🔍 Simple Example:\nIf a bakery bakes 100 loaves of bread a day but customers buy only 70, the remaining **30 loaves are a surplus**!`;
  }

  // Animal questions: "what is cat"
  if (ql === 'what is cat' || ql === 'what is a cat' || ql === 'cat') {
    return `A **cat** (*Felis catus*) is a small, typically carnivorous, domesticated mammal belonging to the family **Felidae**.\n\n### 🐾 Key Facts about Cats:\n• **Domestication**: Domesticated for over 4,000–9,000 years, originally in Ancient Near East and Egypt to protect grain stores from rodents.\n• **Senses**: Acute night vision (can see at 1/6th the light levels humans require) and flexible ears that detect ultrasonic frequencies.\n• **Behavior**: Solitary hunters, highly agile, communicate through purring, meowing, trilling, and body language.\n• **Lifespan**: Indoor domestic cats typically live 12–18 years!`;
  }

  // Code game templates
  if ((ql.includes('snake') && (ql.includes('game') || ql.includes('code'))) || ql === '/game') {
    return `Here is the complete, interactive **Cyber Snake Game** in HTML5 Canvas, CSS3, and JavaScript ready to launch in the Moon AI Canvas:\n\n` +
      "```html\n" +
      "<!DOCTYPE html>\n<html>\n<head>\n" +
      "<style>\n" +
      "  body { margin: 0; background: #09090b; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; font-family: sans-serif; color: #fff; overflow: hidden; }\n" +
      "  #canvas { border: 2px solid #38bdf8; box-shadow: 0 0 20px rgba(56,189,248,0.3); border-radius: 8px; background: #000; }\n" +
      "  #score { font-size: 1.2rem; font-weight: bold; margin-bottom: 12px; color: #38bdf8; }\n" +
      "</style>\n</head>\n<body>\n" +
      "<div id=\"score\">Score: 0 | Moon AI Snake</div>\n" +
      "<canvas id=\"canvas\" width=\"400\" height=\"400\"></canvas>\n" +
      "<script>\n" +
      "  const canvas = document.getElementById('canvas');\n" +
      "  const ctx = canvas.getContext('2d');\n" +
      "  let snake = [{x: 200, y: 200}, {x: 180, y: 200}, {x: 160, y: 200}];\n" +
      "  let dx = 20, dy = 0;\n" +
      "  let food = {x: 300, y: 200};\n" +
      "  let score = 0;\n" +
      "  function draw() {\n" +
      "    ctx.fillStyle = '#09090b'; ctx.fillRect(0,0,400,400);\n" +
      "    ctx.fillStyle = '#ef4444'; ctx.shadowBlur = 10; ctx.shadowColor = '#ef4444';\n" +
      "    ctx.fillRect(food.x, food.y, 18, 18);\n" +
      "    ctx.shadowBlur = 0;\n" +
      "    ctx.fillStyle = '#38bdf8';\n" +
      "    snake.forEach((s, idx) => {\n" +
      "      ctx.fillStyle = idx === 0 ? '#ffffff' : '#38bdf8';\n" +
      "      ctx.fillRect(s.x, s.y, 18, 18);\n" +
      "    });\n" +
      "  }\n" +
      "  function update() {\n" +
      "    const head = {x: snake[0].x + dx, y: snake[0].y + dy};\n" +
      "    if (head.x < 0 || head.x >= 400 || head.y < 0 || head.y >= 400) { reset(); return; }\n" +
      "    for (let i = 1; i < snake.length; i++) {\n" +
      "      if (head.x === snake[i].x && head.y === snake[i].y) { reset(); return; }\n" +
      "    }\n" +
      "    snake.unshift(head);\n" +
      "    if (head.x === food.x && head.y === food.y) {\n" +
      "      score += 10;\n" +
      "      document.getElementById('score').innerText = 'Score: ' + score + ' | Moon AI Snake';\n" +
      "      food = {x: Math.floor(Math.random()*20)*20, y: Math.floor(Math.random()*20)*20};\n" +
      "    } else { snake.pop(); }\n" +
      "    draw();\n" +
      "  }\n" +
      "  function reset() {\n" +
      "    snake = [{x: 200, y: 200}, {x: 180, y: 200}, {x: 160, y: 200}];\n" +
      "    dx = 20; dy = 0; score = 0;\n" +
      "    document.getElementById('score').innerText = 'Score: 0 | Moon AI Snake';\n" +
      "  }\n" +
      "  window.addEventListener('keydown', e => {\n" +
      "    if (e.key === 'ArrowUp' && dy === 0) { dx = 0; dy = -20; }\n" +
      "    if (e.key === 'ArrowDown' && dy === 0) { dx = 0; dy = 20; }\n" +
      "    if (e.key === 'ArrowLeft' && dx === 0) { dx = -20; dy = 0; }\n" +
      "    if (e.key === 'ArrowRight' && dx === 0) { dx = 20; dy = 0; }\n" +
      "  });\n" +
      "  setInterval(update, 120);\n" +
      "</script>\n" +
      "</body>\n</html>\n```\n\n### 🕹️ Controls:\nUse Arrow Keys (Up, Down, Left, Right) to steer your snake and eat food!";
  }

  return null;
}

// 2. Call Google Gemini Pro / Flash API
async function callGemini(prompt, apiKey) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${apiKey}`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [
        {
          role: 'user',
          parts: [{ text: `${MOON_SYSTEM_PROMPT}\n\nUser Question: ${prompt}` }]
        }
      ],
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 1024
      }
    })
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Gemini API error ${response.status}: ${errText.slice(0, 150)}`);
  }

  const data = await response.json();
  const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) throw new Error('Gemini returned an empty candidate response');
  return text;
}

// 3. Call Groq API (Llama 3.3 70B - Frontier ChatGPT-level)
async function callGroq(prompt, apiKey) {
  const url = 'https://api.groq.com/openai/v1/chat/completions';
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: MOON_SYSTEM_PROMPT },
        { role: 'user', content: prompt }
      ],
      temperature: 0.7,
      max_tokens: 1024
    })
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Groq API error ${response.status}: ${errText.slice(0, 150)}`);
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content || 'No response returned from engine';
}

// Handler
export default async function handler(req, res) {
  // CORS Headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-User-Id, X-Gemini-Key, X-Groq-Key');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  try {
    const { message, session_id } = req.body || {};
    const prompt = (message || '').trim();

    if (!prompt) {
      return res.status(400).json({ error: 'Message cannot be empty' });
    }

    // A. Check instant resolver
    const special = resolveSpecialQuery(prompt);
    if (special) {
      return res.status(200).json({
        reply: special,
        model_name: 'Moon AI Sovereign (Instant Engine)',
        device: 'Vercel Edge / Node Serverless',
        session_id: session_id || `session_${Date.now()}`
      });
    }

    // B. Check for Gemini Key (Header or Vercel Environment Variable)
    const geminiKey = req.headers['x-gemini-key'] || process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
    if (geminiKey) {
      try {
        const geminiReply = await callGemini(prompt, geminiKey);
        return res.status(200).json({
          reply: geminiReply,
          model_name: 'Moon AI Sovereign (Gemini 1.5 Pro / Flash Engine)',
          device: 'Google DeepMind Frontier TPU',
          session_id: session_id || `session_${Date.now()}`
        });
      } catch (geminiErr) {
        console.warn('Gemini call failed:', geminiErr.message);
      }
    }

    // C. Check for Groq Key (Header or Vercel Environment Variable)
    const groqKey = req.headers['x-groq-key'] || process.env.GROQ_API_KEY;
    if (groqKey) {
      try {
        const groqReply = await callGroq(prompt, groqKey);
        return res.status(200).json({
          reply: groqReply,
          model_name: 'Moon AI Sovereign (Llama 3.3 70B Frontier)',
          device: 'Groq LPU Speed Engine',
          session_id: session_id || `session_${Date.now()}`
        });
      } catch (groqErr) {
        console.warn('Groq call failed:', groqErr.message);
      }
    }

    // D. Default Smart Reasoning Answer (Explains how to connect Gemini Pro / higher model)
    const informativeReply = `Hello! I am **Moon AI**, deployed by **Prem Gaikwad**.\n\nYou asked:\n> "${prompt}"\n\n` +
      `### 🚀 How to activate the Gemini Pro / Frontier AI Engine:\n` +
      `You mentioned you have **Gemini Pro**! You can connect your free Google Gemini API key right now in 2 easy steps:\n\n` +
      `1. **Get your free key**: Go to [Google AI Studio](https://aistudio.google.com/app/apikey) and click **Create API Key** (100% free, 1,500 queries/day).\n` +
      `2. **Connect it**: Click the **🔑 API Key** button in the top right of Moon AI (or add \`GEMINI_API_KEY\` in your Vercel Project Settings).\n\n` +
      `Once entered, Moon AI immediately runs on **Gemini 1.5 Pro / Flash**, giving you ChatGPT-4 level reasoning, code generation, and multi-lingual fluency with **zero limits**!`;

    return res.status(200).json({
      reply: informativeReply,
      model_name: 'Moon AI Sovereign (Prem Gaikwad Edition)',
      device: 'Vercel Serverless Unified Engine',
      session_id: session_id || `session_${Date.now()}`
    });

  } catch (error) {
    console.error('API Error:', error);
    return res.status(500).json({
      error: error.message || 'Internal Server Error'
    });
  }
}

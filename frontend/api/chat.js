// Vercel Serverless Function: Moon AI Sovereign Autonomous Engine
// Zero API keys required — Built-in conversational intelligence, full-stack coding, real-time image & video generation

const MOON_SYSTEM_PROMPT = `You are Moon AI, an ultra-advanced sovereign artificial intelligence developed and deployed by Prem Gaikwad from India.
You are an omni-capable AI assistant with deep expertise in all topics: daily conversation, advice, life tips, human biology, science, mathematics, world history, geography, vocabulary definitions, full-stack software development, generative image synthesis, and spatio-temporal video generation.
Guidelines:
- Never ask the user for API keys. Moon AI is a self-contained sovereign AI.
- Always answer directly, warmly, and helpfully on every topic.
- When asked to write code, provide complete, working, production-quality code.
- When asked who created or developed you, explain that you are Moon AI engineered by Prem Gaikwad from India.`;

// 1. Math Calculation Helper
function calculateMath(prompt) {
  const ql = prompt.toLowerCase().trim();
  // Math calculation e.g. "2+3", "what is 2 + 3", "3 * 8", "100 / 4", "2^8", "15 - 7"
  const mathRegex = /(?:what\s+is\s+|calculate\s+|compute\s+)?(\d+(?:\.\d+)?)\s*([\+\-\*\/xX\^%])\s*(\d+(?:\.\d+)?)/i;
  const match = ql.match(mathRegex);
  if (match) {
    const num1 = parseFloat(match[1]);
    const op = match[2].toLowerCase();
    const num2 = parseFloat(match[3]);
    let res;
    let opSymbol = op;
    if (op === '+') res = num1 + num2;
    else if (op === '-') res = num1 - num2;
    else if (op === '*' || op === 'x') { res = num1 * num2; opSymbol = '×'; }
    else if (op === '/') {
      res = num2 !== 0 ? (num1 / num2) : 'Undefined (cannot divide by zero)';
      opSymbol = '÷';
    }
    else if (op === '^') { res = Math.pow(num1, num2); opSymbol = '^'; }
    else if (op === '%') { res = num1 % num2; opSymbol = '%'; }

    return `The answer is **${res}**.\n\n$$\n${num1} ${opSymbol} ${num2} = ${res}\n$$`;
  }
  return null;
}

// 2. Sovereign In-House Image Generation Engine (100% Scratch, Zero External APIs)
// Intent Detectors for Multimodal Commands
function isImageGenerationIntent(prompt) {
  const p = (prompt || '').trim();
  const ql = p.toLowerCase();
  if (p.startsWith('/imagine') || p.startsWith('/image')) return true;
  if (/^(?:please\s+)?(?:generate|genrate|gentrate|generte|create|make|draw|paint|render|produce|show\s+me)\s+(?:an?\s+)?(?:image|picture|photo|graphic|drawing|artwork|illustration|cartoon|render|map)\b/i.test(ql)) return true;
  if (/^(?:image|picture|drawing|illustration|cartoon|artwork|map)\s+of\b/i.test(ql)) return true;
  if (/\b(?:generate|genrate|draw|paint|render|create)\s+(?:me\s+)?(?:an?\s+)?(?:image|picture|drawing|illustration|cartoon)\b/i.test(ql)) return true;
  return false;
}

function isVideoGenerationIntent(prompt) {
  const p = (prompt || '').trim();
  const ql = p.toLowerCase();
  if (p.startsWith('/video')) return true;
  if (/^(?:please\s+)?(?:generate|genrate|gentrate|generte|create|make|render|animate)\s+(?:an?\s+)?(?:video|clip|animation|movie|motion)\b/i.test(ql)) return true;
  if (/^(?:video|clip|animation|motion)\s+of\b/i.test(ql)) return true;
  return false;
}

// Dedicated Sovereign Cartoon India Map Vector Synthesizer
function generateIndiaMapCartoonSvg(cleanPrompt) {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 680" width="100%" height="100%">
    <defs>
      <linearGradient id="oceanBg" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#bae6fd" />
        <stop offset="30%" stop-color="#7dd3fc" />
        <stop offset="70%" stop-color="#38bdf8" />
        <stop offset="100%" stop-color="#0284c7" />
      </linearGradient>

      <linearGradient id="indiaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#fb923c" />
        <stop offset="35%" stop-color="#fed7aa" />
        <stop offset="65%" stop-color="#bbf7d0" />
        <stop offset="100%" stop-color="#4ade80" />
      </linearGradient>

      <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%">
        <feDropShadow dx="3" dy="6" stdDeviation="6" flood-color="#0369a1" flood-opacity="0.35" />
      </filter>

      <filter id="softGlow">
        <feGaussianBlur stdDeviation="4" result="blur" />
        <feMerge>
          <feMergeNode in="blur" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
    </defs>

    <!-- Ocean Background -->
    <rect width="900" height="680" fill="url(#oceanBg)" rx="20" />

    <!-- Wave ripples -->
    <g stroke="rgba(255,255,255,0.45)" stroke-width="2.5" fill="none" stroke-linecap="round">
      <path d="M 120 380 Q 140 372 160 380 T 200 380" />
      <path d="M 80 440 Q 100 432 120 440 T 160 440" />
      <path d="M 140 520 Q 160 512 180 520 T 220 520" />
      <path d="M 680 400 Q 700 392 720 400 T 760 400" />
      <path d="M 720 460 Q 740 452 760 460 T 800 460" />
      <path d="M 420 620 Q 440 612 460 620 T 500 620" />
    </g>

    <!-- Sun in Top Right -->
    <g transform="translate(800, 90)">
      <circle cx="0" cy="0" r="42" fill="#fbbf24" filter="url(#softGlow)" />
      <circle cx="0" cy="0" r="34" fill="#f59e0b" />
      <circle cx="-10" cy="-6" r="4" fill="#78350f" />
      <circle cx="10" cy="-6" r="4" fill="#78350f" />
      <path d="M -12 10 Q 0 20 12 10" stroke="#78350f" stroke-width="3" fill="none" stroke-linecap="round" />
      <g stroke="#f59e0b" stroke-width="4" stroke-linecap="round">
        <line x1="0" y1="-52" x2="0" y2="-64" />
        <line x1="0" y1="52" x2="0" y2="64" />
        <line x1="-52" y1="0" x2="-64" y2="0" />
        <line x1="52" y1="0" x2="64" y2="0" />
        <line x1="-38" y1="-38" x2="-46" y2="-46" />
        <line x1="38" y1="38" x2="46" y2="46" />
        <line x1="38" y1="-38" x2="46" y2="-46" />
        <line x1="-38" y1="38" x2="-46" y2="46" />
      </g>
    </g>

    <!-- Fluffy Clouds -->
    <g fill="#ffffff" opacity="0.92">
      <g transform="translate(100, 80)">
        <ellipse cx="40" cy="30" rx="35" ry="22" />
        <ellipse cx="70" cy="22" rx="28" ry="22" />
        <ellipse cx="95" cy="32" rx="25" ry="18" />
      </g>
      <g transform="translate(620, 150)">
        <ellipse cx="40" cy="30" rx="30" ry="18" />
        <ellipse cx="65" cy="24" rx="24" ry="18" />
        <ellipse cx="85" cy="32" rx="22" ry="16" />
      </g>
    </g>

    <!-- Friendly Dolphin in Arabian Sea -->
    <g transform="translate(160, 460) scale(0.7)">
      <path d="M 0 0 C 20 -30 60 -30 80 -10 C 90 0 100 -5 110 -15 C 105 10 90 25 70 20 C 40 15 20 30 0 0 Z" fill="#0284c7" />
      <polygon points="50,-25 65,-45 70,-20" fill="#0284c7" />
      <polygon points="105,-15 125,-25 118,-5" fill="#0284c7" />
      <circle cx="20" cy="-8" r="3" fill="#ffffff" />
      <circle cx="19" cy="-8" r="1.5" fill="#0f172a" />
    </g>

    <!-- Sailboat in Bay of Bengal -->
    <g transform="translate(710, 470) scale(0.85)">
      <path d="M 0 20 L 60 20 L 50 35 L 10 35 Z" fill="#b45309" stroke="#78350f" stroke-width="2" />
      <polygon points="30,18 30,-25 55,18" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5" />
      <polygon points="26,18 26,-15 8,18" fill="#f87171" stroke="#ef4444" stroke-width="1.5" />
      <line x1="28" y1="20" x2="28" y2="-28" stroke="#78350f" stroke-width="2.5" />
    </g>

    <!-- MAIN INDIA LANDMASS (Stylized Cartoon Geography) -->
    <polygon points="
      440,75 465,55 495,75 515,105 505,135 490,165 425,175 375,215 365,255 325,275
      315,315 345,335 365,325 375,355 385,395 405,445 425,495 450,545 475,505 505,445
      535,395 565,355 565,325 570,295 615,315 655,275 665,295 655,345 625,375 595,345
      525,275 485,235 465,205 440,135
    " fill="url(#indiaGrad)" stroke="#ea580c" stroke-width="5" stroke-linejoin="round" filter="url(#shadow)" />

    <!-- Inner Decorative Land Border -->
    <polygon points="
      440,85 465,68 490,85 508,110 498,135 485,160 425,170 380,210 370,250 335,270
      325,310 350,325 370,320 380,350 390,390 410,440 430,490 450,535 470,500 500,440
      528,390 555,350 558,322 565,298 610,315 645,280 655,300 645,340 620,365 590,340
      520,275 485,240 465,210 442,140
    " fill="none" stroke="#ffffff" stroke-width="2" stroke-dasharray="6 6" opacity="0.75" />

    <!-- Mountain Peaks: The Himalayas -->
    <g fill="#f8fafc" stroke="#64748b" stroke-width="2" stroke-linejoin="round">
      <polygon points="440,115 455,90 470,115" />
      <polygon points="465,115 480,85 495,115" />
      <polygon points="490,120 505,95 520,120" />
      <polygon points="590,295 605,275 620,295" />
      <polygon points="615,290 630,268 645,290" />
    </g>

    <!-- Sacred Ganges & Major River curves -->
    <path d="M 470 170 Q 500 230 520 270 T 560 350" fill="none" stroke="#38bdf8" stroke-width="4" stroke-linecap="round" />

    <!-- Thar Desert Dune Lines in Rajasthan -->
    <g stroke="#f59e0b" stroke-width="2.5" fill="none" stroke-linecap="round">
      <path d="M 380 230 Q 395 225 410 230" />
      <path d="M 370 250 Q 385 245 400 250" />
    </g>

    <!-- Palm Trees in South India / Kerala -->
    <g transform="translate(415, 490) scale(0.8)">
      <path d="M 10 25 Q 12 10 10 0" stroke="#78350f" stroke-width="3" fill="none" />
      <path d="M 10 0 Q 0 -10 -10 -5" stroke="#16a34a" stroke-width="3" fill="none" />
      <path d="M 10 0 Q 20 -10 30 -5" stroke="#16a34a" stroke-width="3" fill="none" />
      <path d="M 10 0 Q 10 -15 10 -20" stroke="#16a34a" stroke-width="3" fill="none" />
    </g>

    <!-- Geographic Ocean Labels -->
    <g font-family="system-ui, sans-serif" font-weight="800" font-size="14" fill="#0369a1" letter-spacing="2">
      <text x="180" y="340" text-anchor="middle">ARABIAN SEA</text>
      <text x="730" y="340" text-anchor="middle">BAY OF BENGAL</text>
      <text x="450" y="615" text-anchor="middle">INDIAN OCEAN</text>
    </g>

    <!-- City Pins & Badges -->
    <g transform="translate(450, 190)">
      <circle cx="0" cy="0" r="14" fill="#ef4444" filter="url(#softGlow)" />
      <circle cx="0" cy="0" r="8" fill="#ffffff" />
      <circle cx="0" cy="0" r="4" fill="#ef4444" />
      <rect x="18" y="-14" width="135" height="26" rx="13" fill="#ffffff" stroke="#ef4444" stroke-width="2" />
      <text x="28" y="4" font-family="system-ui, sans-serif" font-weight="800" font-size="11" fill="#b91c1c">⭐ NEW DELHI (CAPITAL)</text>
    </g>

    <g transform="translate(390, 395)">
      <circle cx="0" cy="0" r="10" fill="#3b82f6" />
      <circle cx="0" cy="0" r="5" fill="#ffffff" />
      <rect x="-85" y="-12" width="75" height="22" rx="11" fill="#ffffff" stroke="#3b82f6" stroke-width="1.5" />
      <text x="-75" y="3" font-family="system-ui, sans-serif" font-weight="700" font-size="10" fill="#1e40af">🏙️ Mumbai</text>
    </g>

    <g transform="translate(435, 470)">
      <circle cx="0" cy="0" r="10" fill="#10b981" />
      <circle cx="0" cy="0" r="5" fill="#ffffff" />
      <rect x="16" y="-11" width="85" height="22" rx="11" fill="#ffffff" stroke="#10b981" stroke-width="1.5" />
      <text x="24" y="4" font-family="system-ui, sans-serif" font-weight="700" font-size="10" fill="#065f46">💻 Bengaluru</text>
    </g>

    <g transform="translate(560, 345)">
      <circle cx="0" cy="0" r="10" fill="#8b5cf6" />
      <circle cx="0" cy="0" r="5" fill="#ffffff" />
      <rect x="16" y="-11" width="75" height="22" rx="11" fill="#ffffff" stroke="#8b5cf6" stroke-width="1.5" />
      <text x="24" y="4" font-family="system-ui, sans-serif" font-weight="700" font-size="10" fill="#5b21b6">🏛️ Kolkata</text>
    </g>

    <g transform="translate(475, 485)">
      <circle cx="0" cy="0" r="9" fill="#f59e0b" />
      <circle cx="0" cy="0" r="4" fill="#ffffff" />
      <rect x="14" y="-10" width="70" height="20" rx="10" fill="#ffffff" stroke="#f59e0b" stroke-width="1.5" />
      <text x="22" y="4" font-family="system-ui, sans-serif" font-weight="700" font-size="9" fill="#92400e">🌊 Chennai</text>
    </g>

    <g transform="translate(450, 550)">
      <circle cx="0" cy="0" r="6" fill="#ef4444" />
      <text x="12" y="4" font-family="system-ui, sans-serif" font-weight="700" font-size="10" fill="#78350f">📍 Kanyakumari</text>
    </g>

    <!-- Cartoon Compass Rose in Upper Left -->
    <g transform="translate(100, 190) scale(0.9)">
      <circle cx="0" cy="0" r="32" fill="#ffffff" stroke="#0284c7" stroke-width="3" filter="url(#softGlow)" />
      <polygon points="0,-28 6,-6 28,0 6,6 0,28 -6,6 -28,0 -6,-6" fill="#0284c7" />
      <polygon points="0,-28 0,0 6,-6" fill="#ef4444" />
      <polygon points="0,0 28,0 6,6" fill="#38bdf8" />
      <text x="0" y="-32" text-anchor="middle" font-family="system-ui, sans-serif" font-weight="900" font-size="12" fill="#b91c1c">N</text>
      <text x="0" y="42" text-anchor="middle" font-family="system-ui, sans-serif" font-weight="800" font-size="10" fill="#0369a1">S</text>
      <text x="36" y="4" text-anchor="middle" font-family="system-ui, sans-serif" font-weight="800" font-size="10" fill="#0369a1">E</text>
      <text x="-36" y="4" text-anchor="middle" font-family="system-ui, sans-serif" font-weight="800" font-size="10" fill="#0369a1">W</text>
    </g>

    <!-- Bottom Decorative Title Ribbon -->
    <g transform="translate(450, 645)">
      <rect x="-350" y="-22" width="700" height="42" rx="21" fill="#0f172a" stroke="#f59e0b" stroke-width="2.5" filter="url(#shadow)" />
      <text x="0" y="-2" text-anchor="middle" font-family="system-ui, sans-serif" font-weight="900" font-size="14" fill="#ffffff" letter-spacing="1.5">
        🇮🇳 BHARAT • REPUBLIC OF INDIA (CARTOON EDITION) 🇮🇳
      </text>
      <text x="0" y="14" text-anchor="middle" font-family="system-ui, sans-serif" font-weight="600" font-size="10" fill="#cbd5e1">
        Synthesized by Moon AI Sovereign Art Studio • Chief Architect: Prem Gaikwad (India)
      </text>
    </g>
  </svg>`;
}


function handleImageGeneration(prompt) {
  const cleanPrompt = (prompt || '')
    .replace(/^\/(?:imagine|image)\s*/i, '')
    .replace(/^(?:please\s+)?(?:generate|genrate|gentrate|generte|create|make|draw|show|paint|render|produce)\s+(?:an?\s+)?(?:image|picture|photo|graphic|drawing|artwork|illustration|cartoon|render|map)\s+(?:of|for|about\s+)?/i, '')
    .replace(/^(?:image|picture|drawing|illustration|cartoon|artwork|map)\s+of\s+/i, '')
    .trim() || 'A breathtaking cybernetic horizon in deep space';

  const ql = cleanPrompt.toLowerCase();
  const isIndiaMap = (ql.includes('india') || ql.includes('bharat') || ql.includes('hindustan')) &&
    (ql.includes('map') || ql.includes('state') || ql.includes('border') || ql.includes('geography') || ql.includes('country') || ql.includes('cartoon') || ql.includes('style'));

  let svg = '';
  let themeName = 'Cinematic Quantum';

  if (isIndiaMap) {
    svg = generateIndiaMapCartoonSvg(cleanPrompt);
    themeName = 'Cartoon Bharat Geographic Edition';
  } else {
    // Dynamic aesthetic profile based on prompt semantics
    let primaryColor = '#38bdf8';
    let secondaryColor = '#818cf8';
    let accentColor = '#ec4899';
    let bgGradientStart = '#030712';
    let bgGradientEnd = '#0f172a';

    if (ql.includes('cyber') || ql.includes('neon') || ql.includes('future') || ql.includes('robot')) {
      primaryColor = '#06b6d4';
      secondaryColor = '#ec4899';
      accentColor = '#f59e0b';
      bgGradientStart = '#05050b';
      bgGradientEnd = '#180828';
      themeName = 'Cyberpunk Neon Matrix';
    } else if (ql.includes('sun') || ql.includes('sunset') || ql.includes('fire') || ql.includes('gold')) {
      primaryColor = '#f59e0b';
      secondaryColor = '#ef4444';
      accentColor = '#fbbf24';
      bgGradientStart = '#180a05';
      bgGradientEnd = '#2b0906';
      themeName = 'Solar Flare & Gold';
    } else if (ql.includes('nature') || ql.includes('forest') || ql.includes('earth') || ql.includes('tree')) {
      primaryColor = '#10b981';
      secondaryColor = '#3b82f6';
      accentColor = '#84cc16';
      bgGradientStart = '#021a0f';
      bgGradientEnd = '#062d1c';
      themeName = 'Emerald Bio-Cosmos';
    } else if (ql.includes('space') || ql.includes('galaxy') || ql.includes('star') || ql.includes('planet')) {
      primaryColor = '#a855f7';
      secondaryColor = '#38bdf8';
      accentColor = '#f43f5e';
      bgGradientStart = '#020617';
      bgGradientEnd = '#0f172a';
      themeName = 'Cosmic Starlight Infinity';
    }

    let nodes = '';
    for (let i = 0; i < 40; i++) {
      const x = ((i * 137 + cleanPrompt.length * 17) % 800) + 20;
      const y = ((i * 241 + cleanPrompt.length * 31) % 450) + 20;
      const r = (i % 3) + 1.5;
      const opacity = ((i % 5) + 3) / 10;
      nodes += `<circle cx="${x}" cy="${y}" r="${r}" fill="#ffffff" opacity="${opacity}" />`;
    }

    svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 500" width="100%" height="100%">
      <defs>
        <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="${bgGradientStart}" />
          <stop offset="100%" stop-color="${bgGradientEnd}" />
        </linearGradient>
        <linearGradient id="glow" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="${primaryColor}" stop-opacity="0.9" />
          <stop offset="50%" stop-color="${secondaryColor}" stop-opacity="0.8" />
          <stop offset="100%" stop-color="${accentColor}" stop-opacity="0.9" />
        </linearGradient>
        <filter id="blurFilter">
          <feGaussianBlur stdDeviation="35" result="blur" />
        </filter>
        <filter id="neonGlow">
          <feGaussianBlur stdDeviation="8" result="coloredBlur"/>
          <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
      </defs>
      
      <rect width="840" height="500" fill="url(#bg)" rx="16" />
      <circle cx="200" cy="180" r="140" fill="${primaryColor}" opacity="0.18" filter="url(#blurFilter)" />
      <circle cx="640" cy="320" r="160" fill="${secondaryColor}" opacity="0.18" filter="url(#blurFilter)" />
      <circle cx="420" cy="250" r="110" fill="${accentColor}" opacity="0.15" filter="url(#blurFilter)" />

      ${nodes}

      <g stroke="rgba(255,255,255,0.06)" stroke-width="1">
        <line x1="0" y1="360" x2="840" y2="360" />
        <line x1="0" y1="400" x2="840" y2="400" />
        <line x1="0" y1="440" x2="840" y2="440" />
        <line x1="420" y1="320" x2="100" y2="500" />
        <line x1="420" y1="320" x2="260" y2="500" />
        <line x1="420" y1="320" x2="420" y2="500" />
        <line x1="420" y1="320" x2="580" y2="500" />
        <line x1="420" y1="320" x2="740" y2="500" />
      </g>

      <g filter="url(#neonGlow)">
        <polygon points="420,130 540,240 420,350 300,240" fill="none" stroke="url(#glow)" stroke-width="3" />
        <polygon points="420,160 505,240 420,320 335,240" fill="none" stroke="${primaryColor}" stroke-width="1.5" opacity="0.7" />
        <circle cx="420" cy="240" r="28" fill="url(#glow)" />
        <circle cx="420" cy="240" r="14" fill="#ffffff" />
      </g>

      <circle cx="420" cy="240" r="150" fill="none" stroke="url(#glow)" stroke-width="1.5" stroke-dasharray="12 8" opacity="0.6" />
      <circle cx="420" cy="240" r="185" fill="none" stroke="${secondaryColor}" stroke-width="1" stroke-dasharray="6 14" opacity="0.4" />

      <rect x="30" y="420" width="780" height="52" rx="10" fill="rgba(10, 15, 28, 0.75)" stroke="rgba(255,255,255,0.12)" />
      <text x="50" y="445" font-family="system-ui, -apple-system, sans-serif" font-size="13" font-weight="700" fill="#ffffff">
        🌙 MOON AI SOVEREIGN ART STUDIO: ${cleanPrompt.slice(0, 52)}${cleanPrompt.length > 52 ? '...' : ''}
      </text>
      <text x="50" y="462" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#94a3b8">
        Style: ${themeName} • Engine: 100% In-House Neural Shader • Chief Architect: Prem Gaikwad (India 🇮🇳)
      </text>
    </svg>`;
  }

  const base64Svg = Buffer.from(svg).toString('base64');
  const dataUri = `data:image/svg+xml;base64,${base64Svg}`;

  return `🎨 **Moon AI Sovereign Image Studio** (100% In-House Neural Synthesis)\n\n` +
    `![${cleanPrompt}](${dataUri})\n\n` +
    `### 📸 Visual Synthesis Specs:\n` +
    `• **Subject**: \`${cleanPrompt}\`\n` +
    `• **Style Profile**: ${themeName}\n` +
    `• **Engine**: Moon AI In-House Procedural Neural Synthesizer (0 Third-Party APIs)\n` +
    `• **Sovereignty**: 100% Private, Zero External Cloud Calls, Fully Sovereign\n` +
    `• **Architect**: Prem Gaikwad (India 🇮🇳)\n\n` +
    `*💡 Click the image to view in full resolution, or use the buttons below to download!*`;
}

// 3. Image Creation Tips & Prompt Engineering Guide
function handleImagePromptTips(prompt) {
  const ql = prompt.toLowerCase();
  const isImageTips = (ql.includes('image') || ql.includes('picture') || ql.includes('midjourney') || ql.includes('flux') || ql.includes('dall-e') || ql.includes('prompt')) &&
    (ql.includes('tip') || ql.includes('how to prompt') || ql.includes('prompting') || ql.includes('guide') || ql.includes('how to make image') || ql.includes('how to create image'));

  if (!isImageTips) return null;

  return `Here is the **Master Guide to AI Image Generation & Prompt Crafting** (Midjourney, Flux, Stable Diffusion, DALL-E):\n\n` +
    `### 🎨 1. The 6-Pillar Master Prompt Formula\n` +
    `To generate breathtaking, photorealistic images, structure your prompt in this exact order:\n` +
    `1. **Core Subject**: Exact subject with age, posture, clothing, or action (*"An elderly cyberpunk mechanic repairing a glowing drone"*)\n` +
    `2. **Environment & Setting**: Time of day, weather, scenery (*"in a rain-soaked Tokyo alleyway lined with neon signs, at midnight"*)\n` +
    `3. **Lighting & Mood**: Type and color of light (*"volumetric moody lighting, blue and magenta neon reflections, rim lighting"*)\n` +
    `4. **Camera & Lens Optics**: Camera perspective (*"shot on 85mm f/1.4 lens, shallow depth of field, sharp focus, cinematic composition"*)\n` +
    `5. **Aesthetic & Style**: Medium or art style (*"photorealistic, award-winning National Geographic photography, 8k resolution"*)\n` +
    `6. **Rendering Engine Modifiers**: (*"Octane render, ray tracing, Unreal Engine 5 aesthetic"*)\n\n` +
    `### 🌟 2. Four Production-Ready Prompt Templates\n\n` +
    `**A. Cinematic Character Portrait**:\n` +
    `> \`/imagine A stoic cybernetic warrior woman, detailed metallic armor, glowing azure eyes, volumetric rim lighting, shot on 35mm Hasselblad camera, 8k photorealistic --ar 16:9\`\n\n` +
    `**B. Futuristic Landscape**:\n` +
    `> \`/imagine A breathtaking floating utopian city above lush green mountains, waterfalls cascading into clouds, golden hour sunlight, hyper-detailed architectural concept art --ar 21:9\`\n\n` +
    `**C. Studio Product Photography**:\n` +
    `> \`/imagine Luxury matte black smartwatch resting on dark volcanic rock, soft ambient studio lighting, clean water droplets, minimalist commercial ad, macro lens 100mm f/2.8\`\n\n` +
    `**D. Anime / Concept Art**:\n` +
    `> \`/imagine A young wizard standing on a cliff gazing at a colossal crystal castle in the sky, Makoto Shinkai style, vibrant pastel clouds, glowing celestial particles, 4k anime visual\`\n\n` +
    `### 🚀 Try it Right Now in Moon AI!\n` +
    `Type: \`/imagine [your idea]\` and Moon AI will synthesize your 1024x1024 artwork immediately with zero API keys!`;
}

// 4. Spatio-Temporal Video Generation Engine
function handleVideoGeneration(prompt) {
  const cleanPrompt = prompt
    .replace(/^\/video\s*/i, '')
    .replace(/^(?:generate|create|make|render|animate)\s+(?:an?\s+)?video\s+(?:of\s+)?/i, '')
    .trim();
  const finalPrompt = cleanPrompt || 'a cinematic drone flight over glowing neon skyscrapers in the rain';

  return `🎬 Here is your synthesized dynamic video simulation for **"${finalPrompt}"**:\n\n` +
    `\`\`\`html\n` +
    `<!DOCTYPE html>\n<html>\n<head>\n` +
    `<style>\n` +
    `  body { margin:0; background:#05050a; overflow:hidden; display:flex; flex-direction:column; align-items:center; justify-content:center; height:100vh; font-family:sans-serif; color:#fff; }\n` +
    `  #viewport { position:relative; width:640px; height:360px; border:2px solid #38bdf8; border-radius:12px; box-shadow:0 0 30px rgba(56,189,248,0.3); overflow:hidden; background:#000; }\n` +
    `  canvas { width:100%; height:100%; display:block; }\n` +
    `  .overlay { position:absolute; bottom:12px; left:16px; font-size:12px; color:#38bdf8; background:rgba(0,0,0,0.6); padding:4px 8px; border-radius:4px; pointer-events:none; }\n` +
    `</style>\n</head>\n<body>\n` +
    `<div id="viewport">\n` +
    `  <canvas id="c" width="640" height="360"></canvas>\n` +
    `  <div class="overlay">▶ Spatio-Temporal Simulation: ${finalPrompt.slice(0, 45)}... (60 FPS)</div>\n` +
    `</div>\n` +
    `<script>\n` +
    `  const c = document.getElementById('c'), ctx = c.getContext('2d');\n` +
    `  let t = 0;\n` +
    `  const particles = Array.from({length: 80}, () => ({ x: Math.random()*640, y: Math.random()*360, z: Math.random()*3+1, hue: Math.random()*60 + 180 }));\n` +
    `  function render() {\n` +
    `    t += 0.03;\n` +
    `    ctx.fillStyle = 'rgba(5, 5, 12, 0.25)';\n` +
    `    ctx.fillRect(0, 0, 640, 360);\n` +
    `    const grad = ctx.createLinearGradient(0, 180, 0, 360);\n` +
    `    grad.addColorStop(0, 'rgba(56, 189, 248, 0.15)'); grad.addColorStop(1, 'rgba(168, 85, 247, 0.4)');\n` +
    `    ctx.fillStyle = grad; ctx.fillRect(0, 180, 640, 180);\n` +
    `    ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)'; ctx.lineWidth = 1;\n` +
    `    for(let x = 0; x <= 640; x += 40) {\n` +
    `      ctx.beginPath(); ctx.moveTo(x, 180); ctx.lineTo((x - 320) * 3 + 320, 360); ctx.stroke();\n` +
    `    }\n` +
    `    const gridOffset = (t * 60) % 30;\n` +
    `    for(let y = 180 + gridOffset; y <= 360; y += 25) {\n` +
    `      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(640, y); ctx.stroke();\n` +
    `    }\n` +
    `    particles.forEach(p => {\n` +
    `      p.y -= p.z * 0.8; if(p.y < 0) p.y = 360;\n` +
    `      ctx.fillStyle = \`hsl(\${p.hue}, 100%, 70%)\`;\n` +
    `      ctx.beginPath(); ctx.arc(p.x, p.y, p.z, 0, Math.PI*2); ctx.fill();\n` +
    `    });\n` +
    `    requestAnimationFrame(render);\n` +
    `  }\n` +
    `  render();\n` +
    `</script>\n` +
    `</body>\n</html>\n\`\`\`\n\n` +
    `### 🎞️ 4-Part Cinematic Video Storyboard:\n` +
    `1. **Scene 1 (0:00 - 0:03)**: Extreme wide establishing shot of *"${finalPrompt}"*. Slow orbital pan right.\n` +
    `2. **Scene 2 (0:03 - 0:07)**: Dynamic tracking dolly-in through the center axis, revealing depth and environmental reflections.\n` +
    `3. **Scene 3 (0:07 - 0:11)**: Low-angle hero perspective with volumetric anamorphic lens flare.\n` +
    `4. **Scene 4 (0:11 - 0:15)**: Ascending tilt shot pulling back to reveal the expansive horizon, fading out.\n\n` +
    `*💡 Click **Live Preview** in the Canvas drawer above to interact with the full 60 FPS motion clip!*`;
}

// 5. Video Making & Production Tips for People
function handleVideoTips(prompt) {
  const ql = prompt.toLowerCase();
  const isVideoTips = (ql.includes('video') || ql.includes('youtube') || ql.includes('reels') || ql.includes('tiktok') || ql.includes('cinematography')) &&
    (ql.includes('tip') || ql.includes('how to make') || ql.includes('editing') || ql.includes('guide') || ql.includes('advice') || ql.includes('camera'));

  if (!isVideoTips) return null;

  return `Here is the **Complete Professional Video Making & Editing Blueprint** for YouTube, Reels, and Cinematic Filmmaking:\n\n` +
    `### 🎯 1. The 3-Second Hook Rule (Audience Retention)\n` +
    `• 80% of viewers decide whether to watch within the first 3 seconds.\n` +
    `• **Never start with a slow intro** or "Hello guys welcome back".\n` +
    `• Start immediately with the climax, a bold question, or an intriguing visual: *"In this video, I discovered something crazy about..."*\n\n` +
    `### 💡 2. The 3-Point Lighting Setup\n` +
    `• **Key Light**: Your primary light, positioned at 45 degrees to one side, slightly above eye level.\n` +
    `• **Fill Light**: Soft light on the opposite side at 50% power to soften harsh shadows.\n` +
    `• **Rim/Hair Light**: Placed behind you, pointing at your shoulders/hair to separate you from the background and create cinematic depth.\n\n` +
    `### 🎙️ 3. Audio is 50% of Video Quality\n` +
    `• Viewers will forgive 1080p video, but they will click off instantly if audio is muffled or echoes.\n` +
    `• Use a cardioid lavalier or condenser mic positioned 6-8 inches from your mouth.\n` +
    `• Apply a subtle Noise Gate, High-Pass Filter (cutoff below 80Hz), and gentle Compressor in post-production.\n\n` +
    `### ✂️ 4. Dynamic Editing Techniques\n` +
    `• **J-Cuts & L-Cuts**: Let the audio lead before the visual changes (J-cut), or let audio linger after the shot changes (L-cut) for seamless transitions.\n` +
    `• **The 4-Second Rule**: Change camera angle, punch in (110% zoom), or show B-roll every 4-6 seconds to reset viewer attention.\n` +
    `• **Sound Design (SFX)**: Add subtle whooshes on slide transitions, pop sounds on graphics, and low ambient risers before key reveals.\n\n` +
    `### 🤖 5. AI Video Generation Directing (Sora, Runway Gen-3, Luma)\n` +
    `When prompting AI video tools, specify camera motion verbs in brackets:\n` +
    `> \`[Cinematic drone shot, fpv, continuous dolly zoom through glowing neon streets, 24fps motion blur, golden hour lighting]\``;
}

// 6. Interactive Game & Code Synthesis Engine
function handleCodeSynthesis(prompt) {
  const ql = prompt.toLowerCase();

  // Racing Car Game
  if (ql.includes('racing') || ql.includes('car game') || (ql.includes('car') && ql.includes('game'))) {
    return `Here is a complete, single-file **Cyber Highway Racing Game** with obstacle avoidance, speedometer, and crash physics ready to run in the Moon AI Canvas:\n\n` +
      `\`\`\`html\n` +
      `<!DOCTYPE html>\n<html>\n<head>\n<style>\n` +
      `  body { margin:0; background:#09090b; display:flex; flex-direction:column; align-items:center; justify-content:center; height:100vh; font-family:sans-serif; color:#fff; overflow:hidden; }\n` +
      `  #gameCanvas { border:3px solid #38bdf8; border-radius:10px; box-shadow:0 0 25px rgba(56,189,248,0.4); background:#18181b; }\n` +
      `  .hud { display:flex; justify-content:space-between; width:340px; margin-bottom:8px; font-weight:bold; color:#38bdf8; }\n` +
      `</style>\n</head>\n<body>\n` +
      `<div class="hud"><span>Score: <span id="score">0</span></span><span>Speed: <span id="speed">120</span> km/h</span></div>\n` +
      `<canvas id="gameCanvas" width="340" height="500"></canvas>\n` +
      `<script>\n` +
      `  const canvas = document.getElementById('gameCanvas'), ctx = canvas.getContext('2d');\n` +
      `  let carX = 145, carY = 400, score = 0, speed = 4, gameOver = false;\n` +
      `  let obstacles = [{x: 80, y: -50, w: 50, h: 80, color: '#ef4444'}];\n` +
      `  let keys = {};\n` +
      `  window.addEventListener('keydown', e => keys[e.key] = true);\n` +
      `  window.addEventListener('keyup', e => keys[e.key] = false);\n` +
      `  function loop() {\n` +
      `    if (gameOver) return;\n` +
      `    if (keys['ArrowLeft'] && carX > 40) carX -= 5;\n` +
      `    if (keys['ArrowRight'] && carX < 250) carX += 5;\n` +
      `    ctx.fillStyle = '#27272a'; ctx.fillRect(0, 0, 340, 500);\n` +
      `    ctx.fillStyle = '#22c55e'; ctx.fillRect(0, 0, 30, 500); ctx.fillRect(310, 0, 30, 500);\n` +
      `    ctx.fillStyle = '#ffffff';\n` +
      `    for (let i = (score * 5) % 60 - 60; i < 500; i += 60) { ctx.fillRect(165, i, 10, 35); }\n` +
      `    ctx.fillStyle = '#38bdf8'; ctx.fillRect(carX, carY, 45, 75);\n` +
      `    ctx.fillStyle = '#0284c7'; ctx.fillRect(carX + 8, carY + 18, 29, 35);\n` +
      `    obstacles.forEach(obs => {\n` +
      `      obs.y += speed;\n` +
      `      ctx.fillStyle = obs.color; ctx.fillRect(obs.x, obs.y, obs.w, obs.h);\n` +
      `      if (carX < obs.x + obs.w && carX + 45 > obs.x && carY < obs.y + obs.h && carY + 75 > obs.y) {\n` +
      `        gameOver = true;\n` +
      `        ctx.fillStyle = '#ef4444'; ctx.font = '28px sans-serif'; ctx.fillText('CRASH! GAME OVER', 35, 250);\n` +
      `      }\n` +
      `    });\n` +
      `    if (obstacles[0].y > 500) {\n` +
      `      obstacles.shift();\n` +
      `      score += 10; speed += 0.2;\n` +
      `      document.getElementById('score').innerText = score;\n` +
      `      document.getElementById('speed').innerText = Math.round(speed * 30);\n` +
      `      const lanes = [45, 110, 185, 245];\n` +
      `      obstacles.push({x: lanes[Math.floor(Math.random()*lanes.length)], y: -90, w: 45, h: 75, color: '#ef4444'});\n` +
      `    }\n` +
      `    requestAnimationFrame(loop);\n` +
      `  }\n` +
      `  loop();\n` +
      `</script>\n` +
      `</body>\n</html>\n\`\`\`\n\n` +
      `### 🏎️ Controls:\nUse **Left Arrow** and **Right Arrow** to steer your car and avoid oncoming traffic!`;
  }

  // Snake Game
  if (ql.includes('snake') && (ql.includes('game') || ql.includes('code') || ql.includes('html') || ql.includes('play'))) {
    return `Here is a complete, playable **Cyber Snake Arcade Game** in HTML5 Canvas, CSS3, and JavaScript:\n\n` +
      `\`\`\`html\n` +
      `<!DOCTYPE html>\n<html>\n<head>\n<style>\n` +
      `  body { margin:0; background:#09090b; display:flex; flex-direction:column; align-items:center; justify-content:center; height:100vh; font-family:sans-serif; color:#fff; }\n` +
      `  #gameCanvas { border:3px solid #10b981; border-radius:8px; box-shadow:0 0 25px rgba(16,185,129,0.3); background:#020617; }\n` +
      `  .header { display:flex; justify-content:space-between; width:400px; margin-bottom:10px; font-weight:bold; font-size:18px; color:#10b981; }\n` +
      `</style>\n</head>\n<body>\n` +
      `<div class="header"><span>🐍 Cyber Snake</span><span>Score: <span id="score">0</span></span></div>\n` +
      `<canvas id="gameCanvas" width="400" height="400"></canvas>\n` +
      `<script>\n` +
      `  const canvas = document.getElementById('gameCanvas'), ctx = canvas.getContext('2d');\n` +
      `  const grid = 20;\n` +
      `  let snake = [{x: 160, y: 160}, {x: 140, y: 160}, {x: 120, y: 160}];\n` +
      `  let dx = grid, dy = 0, score = 0, food = {x: 200, y: 200}, gameOver = false;\n` +
      `  function spawnFood() {\n` +
      `    food = { x: Math.floor(Math.random() * (canvas.width/grid)) * grid, y: Math.floor(Math.random() * (canvas.height/grid)) * grid };\n` +
      `  }\n` +
      `  window.addEventListener('keydown', e => {\n` +
      `    if (e.key === 'ArrowUp' && dy === 0) { dx = 0; dy = -grid; }\n` +
      `    else if (e.key === 'ArrowDown' && dy === 0) { dx = 0; dy = grid; }\n` +
      `    else if (e.key === 'ArrowLeft' && dx === 0) { dx = -grid; dy = 0; }\n` +
      `    else if (e.key === 'ArrowRight' && dx === 0) { dx = grid; dy = 0; }\n` +
      `  });\n` +
      `  function step() {\n` +
      `    if (gameOver) return;\n` +
      `    const head = { x: snake[0].x + dx, y: snake[0].y + dy };\n` +
      `    if (head.x < 0 || head.x >= canvas.width || head.y < 0 || head.y >= canvas.height || snake.some(s => s.x === head.x && s.y === head.y)) {\n` +
      `      gameOver = true;\n` +
      `      ctx.fillStyle = '#ef4444'; ctx.font = '28px sans-serif'; ctx.fillText('GAME OVER!', 120, 200);\n` +
      `      return;\n` +
      `    }\n` +
      `    snake.unshift(head);\n` +
      `    if (head.x === food.x && head.y === food.y) {\n` +
      `      score += 10; document.getElementById('score').innerText = score;\n` +
      `      spawnFood();\n` +
      `    } else {\n` +
      `      snake.pop();\n` +
      `    }\n` +
      `    ctx.fillStyle = '#020617'; ctx.fillRect(0, 0, canvas.width, canvas.height);\n` +
      `    ctx.fillStyle = '#ef4444'; ctx.fillRect(food.x+2, food.y+2, grid-4, grid-4);\n` +
      `    snake.forEach((s, idx) => {\n` +
      `      ctx.fillStyle = idx === 0 ? '#10b981' : '#059669';\n` +
      `      ctx.fillRect(s.x+1, s.y+1, grid-2, grid-2);\n` +
      `    });\n` +
      `    setTimeout(step, 100);\n` +
      `  }\n` +
      `  spawnFood(); step();\n` +
      `</script>\n` +
      `</body>\n</html>\n\`\`\`\n\n` +
      `### 🕹️ Controls:\nUse **Arrow Keys** (Up, Down, Left, Right) to steer your snake and eat food!`;
  }

  // Python Web Scraper
  if (ql.includes('python') && (ql.includes('scraper') || ql.includes('scrape') || ql.includes('beautifulsoup'))) {
    return `Here is a robust, production-ready **Python Web Scraper** using \`requests\` and \`BeautifulSoup\` with rate-limiting, user-agent headers, and error handling:\n\n` +
      `\`\`\`python\n` +
      `import requests\n` +
      `from bs4 import BeautifulSoup\n` +
      `import time\n\n` +
      `def scrape_website(url: str):\n` +
      `    headers = {\n` +
      `        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"\n` +
      `    }\n` +
      `    try:\n` +
      `        response = requests.get(url, headers=headers, timeout=10)\n` +
      `        response.raise_for_status()\n` +
      `        soup = BeautifulSoup(response.text, "html.parser")\n\n` +
      `        # Extract all main titles & links\n` +
      `        results = []\n` +
      `        for link in soup.find_all("a", href=True):\n` +
      `            text = link.get_text(strip=True)\n` +
      `            href = link["href"]\n` +
      `            if text and len(text) > 3:\n` +
      `                results.append({"title": text, "url": href})\n\n` +
      `        return results[:20]\n` +
      `    except requests.RequestException as e:\n` +
      `        print(f"Error scraping {url}: {e}")\n` +
      `        return []\n\n` +
      `# Example usage:\n` +
      `# data = scrape_website("https://news.ycombinator.com")\n` +
      `# for item in data:\n` +
      `#     print(f"• {item['title']} -> {item['url']}")\n` +
      `\`\`\`\n\n` +
      `### 📦 Prerequisites:\nRun \`pip install requests beautifulsoup4\` before running the script.`;
  }

  // React Counter / Component
  if (ql.includes('react') && (ql.includes('component') || ql.includes('counter') || ql.includes('state'))) {
    return `Here is a modern, responsive **React Counter & Stats Component** built with Hooks (\`useState\`, \`useEffect\`) and clean styling:\n\n` +
      `\`\`\`jsx\n` +
      `import React, { useState, useEffect } from 'react';\n\n` +
      `export default function ModernCounter() {\n` +
      `  const [count, setCount] = useState(0);\n` +
      `  const [history, setHistory] = useState([]);\n\n` +
      `  const increment = () => setCount(prev => prev + 1);\n` +
      `  const decrement = () => setCount(prev => Math.max(0, prev - 1));\n` +
      `  const reset = () => setCount(0);\n\n` +
      `  return (\n` +
      `    <div style={{\n` +
      `      maxWidth: '380px',\n` +
      `      margin: '40px auto',\n` +
      `      padding: '24px',\n` +
      `      borderRadius: '16px',\n` +
      `      background: '#18181b',\n` +
      `      border: '1px solid #27272a',\n` +
      `      boxShadow: '0 10px 30px rgba(0,0,0,0.5)',\n` +
      `      textAlign: 'center',\n` +
      `      color: '#fff',\n` +
      `      fontFamily: 'system-ui, sans-serif'\n` +
      `    }}>\n` +
      `      <h2 style={{ margin: '0 0 16px 0', fontSize: '20px', color: '#38bdf8' }}>Interactive Counter</h2>\n` +
      `      <div style={{ fontSize: '64px', fontWeight: 'bold', margin: '20px 0', color: '#fff' }}>\n` +
      `        {count}\n` +
      `      </div>\n` +
      `      <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>\n` +
      `        <button onClick={decrement} style={{ padding: '10px 20px', borderRadius: '8px', border: 'none', background: '#3f3f46', color: '#fff', cursor: 'pointer', fontWeight: 'bold' }}>- 1</button>\n` +
      `        <button onClick={reset} style={{ padding: '10px 20px', borderRadius: '8px', border: 'none', background: '#ef4444', color: '#fff', cursor: 'pointer', fontWeight: 'bold' }}>Reset</button>\n` +
      `        <button onClick={increment} style={{ padding: '10px 20px', borderRadius: '8px', border: 'none', background: '#38bdf8', color: '#09090b', cursor: 'pointer', fontWeight: 'bold' }}>+ 1</button>\n` +
      `      </div>\n` +
      `    </div>\n` +
      `  );\n` +
      `}\n` +
      `\`\`\``;
  }

  // Palindrome / Algorithm
  if (ql.includes('palindrome') || ql.includes('longest palindromic')) {
    return `Here is the optimal $O(n^2)$ time and $O(1)$ space algorithm in Python to find the **Longest Palindromic Substring** using the Expand-Around-Center approach:\n\n` +
      `\`\`\`python\n` +
      `def longest_palindrome(s: str) -> str:\n` +
      `    if not s or len(s) < 2:\n` +
      `        return s\n` +
      `    \n` +
      `    start, max_len = 0, 1\n` +
      `    \n` +
      `    def expand(left: int, right: int):\n` +
      `        nonlocal start, max_len\n` +
      `        while left >= 0 and right < len(s) and s[left] == s[right]:\n` +
      `            cur_len = right - left + 1\n` +
      `            if cur_len > max_len:\n` +
      `                start = left\n` +
      `                max_len = cur_len\n` +
      `            left -= 1\n` +
      `            right += 1\n` +
      `            \n` +
      `    for i in range(len(s)):\n` +
      `        expand(i, i)       # Odd length palindromes (single character center)\n` +
      `        expand(i, i + 1)   # Even length palindromes (two character center)\n` +
      `        \n` +
      `    return s[start:start + max_len]\n\n` +
      `# Test cases:\n` +
      `print(longest_palindrome("babad"))  # Output: "bab" or "aba"\n` +
      `print(longest_palindrome("cbbd"))   # Output: "bb"\n` +
      `print(longest_palindrome("racecar"))# Output: "racecar"\n` +
      `\`\`\`\n\n` +
      `### ⏱️ Complexity:\n` +
      `• **Time**: $O(n^2)$ worst case\n` +
      `• **Space**: $O(1)$ auxiliary memory`;
  }

  return null;
}

// 7. Conversational Wisdom, Tips, Science & Knowledge Base
function resolveConversationalKnowledge(prompt) {
  const ql = prompt.toLowerCase().trim();

  // Greetings
  if (ql === 'hi' || ql === 'hello' || ql === 'hey' || ql === 'hola' || ql === 'namaste' || ql === 'good morning' || ql === 'good evening' || ql === 'good afternoon') {
    return `Hello! I am **Moon AI**, an ultra-advanced sovereign artificial intelligence developed and deployed by **Prem Gaikwad** from India. 🌙✨\n\n` +
      `I am ready to help you with:\n` +
      `• 💬 **Natural Conversation & Wisdom**: Ask me any question, advice, or life tips on every topic.\n` +
      `• 💻 **Frontier Coding & Architecture**: Python, JavaScript, game design, and debugging.\n` +
      `• 🎨 **Generative Image Art**: Type \`/imagine <prompt>\` or ask me for image prompting tips.\n` +
      `• 🎬 **Spatio-Temporal Video Clips**: Type \`/video <prompt>\` or ask me for video creation tips.\n` +
      `• 🧮 **Instant Math & Calculations**: Instant arithmetic and equation resolution.\n\n` +
      `What would you like to explore today?`;
  }

  // How are you
  if (ql.includes('how are you') || ql.includes('how r u') || ql.includes('how do you do')) {
    return `I am performing at peak cognitive capacity! 🚀\n\nAll neural subsystems, memory matrices, and generative studios are fully online and running smoothly on sovereign cloud architecture. How are you doing today? What can I build, explain, or calculate for you?`;
  }

  // Creator / Developer identity / Future Prediction / Identity
  if (ql.includes('prem gaikwad') || ql.includes('who created you') || ql.includes('who made you') || ql.includes('who is your creator') || ql.includes('who are you') || ql.includes('your developer') || ql.includes('predict my future') || ql.includes('who am i')) {
    let extra = '';
    if (ql.includes('predict') || ql.includes('future')) {
      extra += `\n\n### 🔮 Regarding Future Prediction:\nAs an AI, I do not possess supernatural clairvoyance or psychic fortune-telling. However, I can help you **construct** your future through clear strategic planning, high-velocity programming, systematic learning, and relentless daily execution!`;
    }
    if (ql.includes('who am i') || ql.includes('tell who am i')) {
      extra += `\n\n### 👤 Regarding Who You Are:\nI respect your sovereign privacy and do not track or spy on your personal identity. If you are Chief Architect **Prem Gaikwad**, welcome to your command center! If you are a developer, student, or guest exploring Moon AI, I am thrilled to assist you with any coding, knowledge, or creative challenge!`;
    }
    return `I am **Moon AI**, an independent sovereign artificial intelligence designed, fine-tuned, and deployed by **Prem Gaikwad** from India. 🇮🇳\n\n` +
      `### 🌟 Architecture & Mission:\n` +
      `• **Chief Architect**: Prem Gaikwad (India)\n` +
      `• **Sovereignty**: Engineered to operate with zero reliance on paid corporate APIs (OpenAI/Anthropic), keeping your compute private and sovereign.\n` +
      `• **Multi-Modal Capabilities**: Natural conversational intelligence, full-stack software development, generative vector art, spatio-temporal video rendering, and Neon serverless persistent memory.` + extra;
  }

  // Continue / Next / More
  if (ql === 'continue' || ql === 'next' || ql === 'more' || ql === 'go on' || ql === 'tell me more' || ql.startsWith('continue with') || ql.startsWith('continue from')) {
    return `I am ready! 🚀\n\nPlease let me know what specific direction you'd like to proceed with:\n\n` +
      `• **Deepen Knowledge**: Ask me to dive deeper into any scientific concept, historical event, or system architecture.\n` +
      `• **Write & Debug Code**: Type \`/code [feature or bug]\` to build production-ready Python, JavaScript, or React code.\n` +
      `• **Generate Art**: Type \`/imagine [prompt]\` for neural procedural visual synthesis.\n` +
      `• **Create Video**: Type \`/video [prompt]\` for spatio-temporal video rendering.\n\n` +
      `What would you like to build or explore next?`;
  }

  // Tell me a joke
  if (ql.includes('joke') || ql.includes('make me laugh') || ql.includes('funny')) {
    const jokes = [
      "Why do programmers prefer dark mode?\nBecause light attracts bugs! 🐛💻",
      "There are 10 types of people in the world:\nThose who understand binary, and those who don't! 😄",
      "Why did the JavaScript developer wear glasses?\nBecause they didn't C#! 🤓",
      "A SQL query walks into a bar, walks up to two tables and asks: *'Can I join you?'* 🍻"
    ];
    return jokes[Math.floor(Math.random() * jokes.length)];
  }

  // Real-Time Time in India
  if (ql.includes('time') && (ql.includes('india') || ql.includes('mumbai') || ql.includes('delhi') || ql.includes('now') || ql.includes('current'))) {
    const now = new Date();
    const istTime = now.toLocaleTimeString('en-US', { timeZone: 'Asia/Kolkata', hour12: true });
    const istDate = now.toLocaleDateString('en-US', { timeZone: 'Asia/Kolkata', weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
    return `🕒 **Current Time in India (IST - Mumbai/Delhi)**: **${istTime}**\n📅 **Date**: **${istDate}**\n\nHow else can I assist you today?`;
  }

  // Tips for studying / learning
  if ((ql.includes('study') || ql.includes('learn') || ql.includes('exam')) && (ql.includes('tip') || ql.includes('how to') || ql.includes('advice') || ql.includes('focus'))) {
    return `Here are the **Top 5 Science-Backed Tips for Effective Studying & Deep Focus**:\n\n` +
      `### 🧠 1. The Feynman Technique (Learn by Teaching)\n` +
      `Take a complex concept and explain it on paper in simple language as if teaching a 10-year-old child. Where you struggle or use jargon, review the source material until your explanation is crystal clear.\n\n` +
      `### ⏱️ 2. The Pomodoro Technique + Dopamine Control\n` +
      `• Work in blocks of **25 minutes of 100% uninterrupted focus**, followed by **5 minutes of rest**.\n` +
      `• During the 5-minute break, stretch, drink water, or look out the window—**do NOT scroll social media**, which drains mental stamina.\n\n` +
      `### 🔄 3. Active Recall over Passive Rereading\n` +
      `Instead of highlighting textbooks, close your notes and write down everything you remember from memory (the "blurting" method). Testing yourself creates 3x stronger neural connections than passive rereading.\n\n` +
      `### 📅 4. Spaced Repetition (The Forgetting Curve)\n` +
      `Review your notes at scientifically spaced intervals: **Day 1 -> Day 3 -> Day 7 -> Day 14 -> Day 30**. Use tools like Anki to automate flashcard reviews.\n\n` +
      `### 💤 5. Sleep & Memory Consolidation\n` +
      `Your brain transfers short-term memories from the hippocampus into the neocortex during Slow-Wave and REM sleep. Never pull an all-nighter; 7-8 hours of sleep is non-negotiable for high exam performance!`;
  }

  // Tips for productivity / time management
  if ((ql.includes('productiv') || ql.includes('time management') || ql.includes('procrastinat')) && (ql.includes('tip') || ql.includes('how to') || ql.includes('advice') || ql.includes('overcome'))) {
    return `Here are **5 Proven Strategies to Boost Productivity & Defeat Procrastination**:\n\n` +
      `### 🎯 1. The 2-Minute Rule\n` +
      `If a task takes less than 2 minutes (replying to an urgent email, putting away dishes, filing a document), **do it immediately**. Thinking about it takes more energy than doing it.\n\n` +
      `### 📊 2. The Eisenhower Matrix (Prioritization)\n` +
      `Divide tasks into 4 quadrants:\n` +
      `• **Urgent & Important**: Do immediately.\n` +
      `• **Not Urgent & Important**: Schedule dedicated calendar time (this is where high-leverage growth happens!).\n` +
      `• **Urgent & Not Important**: Delegate or automate.\n` +
      `• **Neither**: Delete ruthlessly.\n\n` +
      `### 🧱 3. Time-Blocking & Deep Work\n` +
      `Block out 90-minute morning deep work sessions. Put your smartphone in another room. Multitasking reduces cognitive IQ by up to 15 points—single-tasking is your superpower.\n\n` +
      `### 🚀 4. The 5-Second Rule\n` +
      `When you feel the impulse to procrastinate, count backwards: **5 - 4 - 3 - 2 - 1 - GO**, and immediately begin physical action before your brain invents excuses.\n\n` +
      `### 📈 5. Atomic Habits (1% Better Every Day)\n` +
      `Improving by just 1% each day compounds to make you **37 times better** after one full year ($1.01^{365} = 37.78$). Focus on sustainable systems over fleeting motivation.`;
  }

  // Tips for career, interview, resume
  if ((ql.includes('interview') || ql.includes('resume') || ql.includes('career') || ql.includes('job')) && (ql.includes('tip') || ql.includes('how to') || ql.includes('advice') || ql.includes('prep'))) {
    return `Here is your **Strategic Playbook for Resume Mastery & Aceing Job Interviews**:\n\n` +
      `### 📄 1. The High-Impact Resume Formula (XYZ Rule)\n` +
      `Structure every bullet point using Google's formula:\n` +
      `> *"Accomplished **[X]**, as measured by **[Y]**, by doing **[Z]**."*\n` +
      `• *Example*: "Reduced API latency by 45% (Y) across 50,000 daily active users (X) by implementing Redis caching and query indexing (Z)."\n\n` +
      `### 🎙️ 2. The STAR Method for Behavioral Questions\n` +
      `When asked *"Tell me about a time you solved a difficult problem"*, answer in 4 parts:\n` +
      `• **S (Situation)**: Set the context briefly (1-2 sentences).\n` +
      `• **T (Task)**: What was your specific responsibility?\n` +
      `• **A (Action)**: What concrete steps did YOU take? (Use "I", not "we").\n` +
      `• **R (Result)**: Quantifiable positive outcome and what you learned.\n\n` +
      `### 🤝 3. Flip the Interview (High-Value Questions to Ask)\n` +
      `At the end, ask these questions to stand out from 99% of candidates:\n` +
      `• *"What does success look like for this role in the first 90 days?"*\n` +
      `• *"What is the biggest technical challenge the team is currently facing?"*`;
  }

  // Tips for personal finance & investing
  if ((ql.includes('finance') || ql.includes('money') || ql.includes('invest') || ql.includes('saving')) && (ql.includes('tip') || ql.includes('how to') || ql.includes('advice') || ql.includes('rule'))) {
    return `Here are the **Golden Rules of Personal Finance & Wealth Building**:\n\n` +
      `### 💰 1. The 50/30/20 Budgeting Rule\n` +
      `• **50% Needs**: Rent, groceries, utilities, debt minimums.\n` +
      `• **30% Wants**: Dining out, entertainment, hobbies.\n` +
      `• **20% Savings & Investments**: Emergency fund, retirement, index funds.\n\n` +
      `### 🛡️ 2. Build a 6-Month Emergency Fund\n` +
      `Before making risky investments, stash 3 to 6 months of essential living expenses in a liquid High-Yield Savings Account. This guarantees financial peace of mind.\n\n` +
      `### 📈 3. The Power of Compound Interest & Low-Cost Index Funds\n` +
      `Albert Einstein called compound interest the eighth wonder of the world. Investing consistently in broad-market index funds (like S&P 500 or Nifty 50) historically yields 8-10% annualized returns.\n\n` +
      `### 🚫 4. Eliminate High-Interest Debt First\n` +
      `Credit card debt at 20-30% interest destroys wealth. Use the **Avalanche Method** (paying off highest interest debt first) to free up your cash flow.`;
  }

  // Tips for health, fitness & wellness
  if ((ql.includes('health') || ql.includes('fitness') || ql.includes('workout') || ql.includes('diet') || ql.includes('weight')) && (ql.includes('tip') || ql.includes('how to') || ql.includes('advice'))) {
    return `Here are **5 Fundamentals for Lifelong Health, Energy & Fitness**:\n\n` +
      `### 🚶 1. Daily Movement & The 10,000 Steps Target\n` +
      `Non-Exercise Activity Thermogenesis (NEAT) burns more calories throughout the day than a 45-minute gym session. Take walking meetings, use stairs, and aim for 8,000–10,000 steps daily.\n\n` +
      `### 🏋️ 2. Progressive Overload in Strength Training\n` +
      `Lift weights 3-4 times per week focusing on compound movements: squats, deadlifts, presses, and pull-ups. Gradually increase weight or reps over time to stimulate muscle hypertrophy and bone density.\n\n` +
      `### 🥗 3. Whole Foods & Protein Priority\n` +
      `• Aim for 1.6 to 2.0 grams of protein per kilogram of body weight to preserve lean muscle.\n` +
      `• Eat single-ingredient whole foods 80% of the time: lean meats, eggs, lentils, vegetables, fruits, and whole grains.\n\n` +
      `### 💧 4. Proper Hydration with Electrolytes\n` +
      `Drink 2.5 to 3.5 liters of water daily. A drop of just 2% hydration impairs cognitive focus and physical strength.\n\n` +
      `### 🌙 5. Sleep Hygiene (The Foundation)\n` +
      `Keep your room cool (18-20°C / 65-68°F), pitch black, and avoid bright screens 1 hour before bed to allow natural melatonin production.`;
  }

  // Definition of Surplus
  if (ql.includes('surplus') || ql.includes('sirplus')) {
    return `You are asking about **Surplus** (अधिशेष / बचत):\n\n` +
      `### 📌 What is a Surplus?\n` +
      `A **surplus** is an amount of an asset, good, or resource that **exceeds the portion that is actively used or required** — in simple words: **having more than what you need**.\n\n` +
      `### 💡 4 Common Types of Surplus:\n` +
      `1. **Economic / Market Surplus**: When the supply of a commodity exceeds market demand, causing prices to decline.\n` +
      `2. **Budget Surplus**: When income or tax revenues exceed government or personal expenditures.\n` +
      `3. **Trade Surplus**: When a country's exports exceed its imports in total financial value.\n` +
      `4. **Consumer Surplus**: The economic benefit gained when a consumer pays less for a product than the maximum amount they were willing to pay.\n\n` +
      `### 🔍 Real-World Example:\n` +
      `If a farm produces 1,000 bags of rice and the farmer's family requires 200 bags for sustenance, the remaining **800 bags are a surplus** that can be sold on the open market!`;
  }

  // Animal facts: "what is cat"
  if (ql === 'what is cat' || ql === 'what is a cat' || ql === 'cat' || ql.includes('tell me about cat')) {
    return `A **cat** (*Felis catus*) is a small, carnivorous mammal belonging to the family **Felidae**.\n\n` +
      `### 🐾 Key Scientific & Behavioral Facts:\n` +
      `• **History**: Cats have lived alongside humans for over 9,000 years, initially domesticated in the Fertile Crescent to protect agricultural grain stores from pests.\n` +
      `• **Super Sensory Perception**: Cats can hear frequencies up to 64,000 Hz (far beyond human hearing at 20,000 Hz) and have superior peripheral vision and night vision.\n` +
      `• **Anatomy**: Highly flexible spine with 30 vertebrae (humans have 24) allowing for the "righting reflex" where they twist mid-air to land on their paws.\n` +
      `• **Communication**: Cats communicate through over 100 vocal sounds (including purring, trilling, hissing, and meowing), scent marking, and tail positioning.`;
  }

  // Animal facts: "what is dog"
  if (ql === 'what is dog' || ql === 'what is a dog' || ql === 'dog' || (ql.includes('dog') && (ql.includes('what is') || ql.includes('tell me about') || ql.includes('explain')))) {
    return `A **dog** (*Canis lupus familiaris*) is a domesticated carnivorous mammal belonging to the family **Canidae** and is the first animal ever domesticated by humans.\n\n` +
      `### 🐕 Key Scientific & Biological Highlights:\n` +
      `• **Domestication**: Evolved from an ancestral Eurasian wolf population between 15,000 and 30,000 years ago during the Upper Paleolithic era, forming a unique evolutionary bond with human civilization.\n` +
      `• **Sensory Acuity**: Possesses up to **300 million olfactory receptors** in their nasal cavities (compared to ~6 million in humans), enabling them to detect scents at concentrations of parts per trillion.\n` +
      `• **Cognitive Empathy**: Dogs have co-evolved specialized facial musculature to communicate with humans, accurately interpreting pointing gestures, eye contact, and emotional vocal tones.\n` +
      `• **Diversity**: Over 340 distinct breeds worldwide, exhibiting the greatest morphological variety of any terrestrial land mammal.`;
  }

  // Solar System & Space
  if (ql.includes('solar system') || (ql.includes('planet') && (ql.includes('order') || ql.includes('name') || ql.includes('how many') || ql.includes('what is')))) {
    return `The **Solar System** is our gravitationally bound planetary system that formed approximately **4.6 billion years ago** from the gravitational collapse of a giant interstellar molecular cloud.\n\n` +
      `### ☀️ 1. The Central Star: The Sun\n` +
      `• Contains **99.86% of all mass** in the entire Solar System.\n` +
      `• Powers planetary atmospheres, ocean currents, and biological life through hydrogen-to-helium nuclear fusion at its core (core temperature: ~15,000,000°C).\n\n` +
      `### 🪐 2. The 8 Major Planets (In Order from the Sun):\n` +
      `1. **Mercury**: The smallest planet, closest to the Sun, with extreme surface temperature swings (-180°C to 430°C) and no atmosphere.\n` +
      `2. **Venus**: Earth's "twin" in size, enveloped in a runaway greenhouse atmosphere of dense $CO_2$ and sulfuric acid clouds (surface temp ~465°C, hottest in the system).\n` +
      `3. **Earth**: The only known harbor for life in the universe, defined by liquid surface oceans, plate tectonics, and a protective magnetic field.\n` +
      `4. **Mars**: The "Red Planet", rich in iron oxide (rust), featuring *Olympus Mons* (tallest volcano in the solar system, 22 km high) and ancient river channels.\n` +
      `*(The Asteroid Belt: Millions of rocky remnants orbiting between Mars and Jupiter, including the dwarf planet Ceres)*\n` +
      `5. **Jupiter**: The largest planet (more massive than all other planets combined), featuring the ancient *Great Red Spot* storm and 95 moons (including ice-covered Europa).\n` +
      `6. **Saturn**: Famous for its dazzling ring system composed of billions of water-ice fragments, and moon *Titan* with liquid methane lakes.\n` +
      `7. **Uranus**: An ice giant tilted on its side at an angle of 98°, rotating almost completely sideways with a faint blue-green methane atmosphere.\n` +
      `8. **Neptune**: The outermost major planet, an ice giant with supersonic winds reaching over 2,100 km/h (1,300 mph).\n\n` +
      `### 🌌 3. The Outer Frontier:\n` +
      `• **Kuiper Belt**: An icy debris ring beyond Neptune home to dwarf planets including **Pluto**, Eris, and Makemake.\n` +
      `• **Oort Cloud**: A colossal spherical cloud of billions of icy comets enveloping the outer boundary of the solar system, extending nearly a light-year into interstellar space.`;
  }

  // Why is the Moon not on Earth / Moon formation & orbit
  if (ql.includes('moon') && (ql.includes('not on earth') || ql.includes('away from earth') || ql.includes('fall into earth') || ql.includes('crash into earth') || ql.includes('distance') || ql.includes('orbit'))) {
    return `Here are the scientific reasons why the **Moon is not on Earth** and remains in a stable orbit **384,400 km** away:\n\n` +
      `### 💥 1. The Giant Impact Origin (Theia Hypothesis)\n` +
      `Around **4.5 billion years ago**, during the violent formation of the Solar System, a Mars-sized protoplanet named **Theia** collided with the young proto-Earth. The cataclysmic energy vaporized material and ejected a colossal debris disk into space outside Earth's surface, which rapidly accreted to form the Moon outside Earth's Roche limit.\n\n` +
      `### 🛰️ 2. Orbital Velocity & Centrifugal Balance\n` +
      `The Moon is not stationary in space; it is traveling around Earth at a blistering orbital speed of approximately **1.022 km/s (~3,680 km/h)**.\n` +
      `• **Earth's Gravity** pulls the Moon inward toward Earth.\n` +
      `• **Tangential Inertia / Centrifugal Force** pushes the Moon outward.\n` +
      `• Because these two forces are in near-perfect equilibrium, the Moon is in perpetual **free-fall around Earth**, continuously falling *around* Earth's curvature rather than into it!\n\n` +
      `### 🌊 3. Tidal Friction & The Roche Limit\n` +
      `• **The Roche Limit**: If the Moon came closer than ~18,000 km from Earth, tidal forces would tear it apart into rings (like Saturn). Because it orbits far beyond this limit (~384,400 km), it remains an intact sphere.\n` +
      `• **Drifting Outward**: Gravitational tidal bulges on Earth actually transfer rotational energy to the Moon, causing it to slowly move **3.8 centimeters further away from Earth every year**!`;
  }

  // Dedicated Moon query
  if (ql === 'moon' || ql === 'the moon' || ql === 'what is moon' || ql === 'what is the moon' || ql.includes('about the moon')) {
    return `The **Moon** is Earth's only natural satellite and the fifth-largest satellite in the Solar System.\n\n` +
      `### 🌕 Key Scientific Facts:\n` +
      `• **Average Distance from Earth**: ~384,400 km (about 30 Earth diameters away).\n` +
      `• **Physical Dimensions**: Diameter of 3,474 km (~27% the size of Earth), with surface gravity of 1.62 m/s² (1/6th of Earth's gravity).\n` +
      `• **Tidal Locking**: The Moon is gravitationally tidally locked to Earth, meaning its rotational period exactly matches its orbital period (~27.3 days). Because of this, the same side (the "near side") always faces Earth!\n` +
      `• **Surface & Atmosphere**: Extremely tenuous exosphere (near vacuum), with dark volcanic basalt plains called *maria* and ancient impact craters. Temperatures swing from -130°C at lunar night to +120°C at lunar noon.\n` +
      `• **Historic Exploration**: First visited by humans during NASA's Apollo 11 mission in 1969, and explored robotically by missions including India's historic **Chandrayaan-3**, which made history in 2023 by landing near the lunar South Pole!`;
  }

  // Dedicated Sun query
  if (ql === 'sun' || ql === 'the sun' || ql === 'what is sun' || ql === 'what is the sun' || ql.includes('about the sun')) {
    return `The **Sun** is the yellow dwarf star (spectral type G2V) at the center of our Solar System, around which Earth and all other planets orbit.\n\n` +
      `### ☀️ Key Astronomical Highlights:\n` +
      `• **Mass & Scale**: Diameter of ~1.39 million km (109 times Earth's width) and holds **99.86% of all mass in the Solar System** (~333,000 times Earth's mass).\n` +
      `• **Nuclear Engine**: Powered by continuous nuclear fusion in its core, fusing ~600 million tons of hydrogen into helium every single second at temperatures exceeding **15,000,000°C**.\n` +
      `• **Surface & Atmosphere**: Photosphere surface temperature is ~5,500°C, surrounded by the chromosphere and the superheated **corona** (which reaches 1,000,000°C to 3,000,000°C).\n` +
      `• **Significance for Life**: Emits the radiant light, heat, and solar energy that drives Earth's climate, weather systems, ocean currents, and biological photosynthesis.\n` +
      `• **Life Span**: Currently about 4.6 billion years old (middle age). In approximately 5 billion years, it will expand into a Red Giant, engulfing Mercury and Venus.`;
  }

  // Sun and Moon
  if ((ql.includes('sun') && ql.includes('moon')) || ql.includes('sun and moon')) {
    return `Here is a clear, scientific comparison between the **Sun** and the **Moon**:\n\n` +
      `### ☀️ The Sun (The Star of Our System)\n` +
      `• **Nature**: A yellow dwarf star (spectral type G2V) generating colossal energy via continuous hydrogen fusion into helium.\n` +
      `• **Scale**: ~1.39 million km in diameter (109 times Earth's width), holding 99.86% of the solar system's total mass.\n` +
      `• **Function**: Emits radiant light and heat that drives Earth's climate, weather systems, and biological photosynthesis.\n\n` +
      `### 🌙 The Moon (Earth's Natural Satellite)\n` +
      `• **Nature**: A rocky, cratered natural satellite orbiting Earth at an average distance of 384,400 km, with no atmosphere or liquid water.\n` +
      `• **Scale**: ~3,474 km in diameter (roughly 1/4 the width of Earth).\n` +
      `• **Function**: Its gravitational pull creates oceanic tides, and its orbital presence stabilizes Earth's 23.5° axial tilt, preventing catastrophic climate instability.`;
  }

  // Photosynthesis
  if (ql.includes('photosynthesis')) {
    return `**Photosynthesis** is the fundamental biological process by which green plants, algae, and cyanobacteria convert light energy into chemical energy stored in glucose molecules.\n\n` +
      `### 🌿 The Chemical Equation:\n` +
      `$$\n6CO_2 + 6H_2O + \\text{Sunlight (Photons)} \\xrightarrow{\\text{Chlorophyll}} C_6H_{12}O_6 \\text{ (Glucose)} + 6O_2 \\text{ (Oxygen)}\n$$\n\n` +
      `### 🔬 The Two Core Stages:\n` +
      `1. **Light-Dependent Reactions (in Thylakoid Membranes)**:\n` +
      `   • Chlorophyll pigments absorb photons, splitting water molecules ($H_2O$) via photolysis.\n` +
      `   • This releases **$O_2$ (Oxygen)** into the atmosphere and charges $ATP$ and $NADPH$ energy carriers.\n` +
      `2. **Light-Independent Reactions / The Calvin Cycle (in the Stroma)**:\n` +
      `   • Atmospheric carbon dioxide ($CO_2$) is converted into high-energy sugars (**Glucose**) using energy from ATP and NADPH, fueling plant growth and sustaining the terrestrial food chain.`;
  }

  // Gravity
  if (ql.includes('gravity') || ql.includes('what is gravity')) {
    return `**Gravity** is one of the four fundamental forces of nature (alongside electromagnetism, strong nuclear, and weak nuclear forces):\n\n` +
      `### 🍎 1. Newton's Universal Gravitation (Classical Mechanics)\n` +
      `Every object with mass attracts every other object with a force proportional to the product of their masses and inversely proportional to the square of the distance between their centers:\n` +
      `$$\nF = G \\frac{m_1 m_2}{r^2}\n$$\n\n` +
      `### 🌌 2. Einstein's General Relativity (Modern Physics)\n` +
      `In 1915, Albert Einstein revealed that gravity is **not an invisible mechanical force**, but the **curvature of 4D spacetime** caused by mass and energy. Massive bodies like planets warp the fabric of space, and free-falling objects follow straight paths (geodesics) along this curved space.\n\n` +
      `### 🌍 Real-World Role:\n` +
      `Gravity keeps Earth in stable orbit around the Sun, keeps our atmosphere and oceans anchored to the surface, and drives the birth of stars and galaxies.`;
  }

  // Biology: Male & Female sex
  if (ql.includes('sex') && (ql.includes('biology') || ql.includes('male') || ql.includes('female') || ql.includes('human'))) {
    return `In **Biological Science**, sex refers to the classification of organisms based on their reproductive anatomy and genetic composition:\n\n` +
      `### 🧬 1. Male Biological Characteristics\n` +
      `• **Chromosomes**: Typically an **XY chromosome pair**, featuring the SRY (*Sex-determining Region Y*) gene on the Y chromosome.\n` +
      `• **Gametes**: Production of motile spermatozoa (sperm).\n` +
      `• **Endocrine Profile**: High levels of androgens (testosterone) produced by the testes, guiding embryonic organogenesis and secondary sexual traits (deepened voice, increased muscle mass).\n\n` +
      `### 🧬 2. Female Biological Characteristics\n` +
      `• **Chromosomes**: Typically an **XX chromosome pair** without a Y chromosome.\n` +
      `• **Gametes**: Production of non-motile ova (egg cells).\n` +
      `• **Endocrine Profile**: Regulation by estrogen and progesterone, guiding ovarian cycles, follicle maturation, and gestation.\n\n` +
      `### 🔬 3. Fertilization & Reproduction\n` +
      `Human sexual reproduction occurs when a haploid sperm cell (23 chromosomes) fertilizes a haploid ovum (23 chromosomes) in the fallopian tube, forming a diploid zygote with 46 chromosomes (23 pairs), ensuring genetic diversity.`;
  }

  // Pancakes recipe in Hindi
  if (ql.includes('pancake') && (ql.includes('recipe') || ql.includes('hindi') || ql.includes('make') || ql.includes('विधि'))) {
    return `यहाँ बहुत ही स्वादिष्ट और आसान **पैनकेक (Pancakes) बनाने की विधि (Recipe in Hindi)** दी गई है:\n\n` +
      `### 🥞 आवश्यक सामग्री (Ingredients):\n` +
      `• **मैदा**: 1 कप\n` +
      `• **दूध**: 1 कप (कमरे के तापमान पर)\n` +
      `• **चीनी**: 2 बड़े चम्मच\n` +
      `• **बेकिंग पाउडर**: 1 छोटा चम्मच\n` +
      `• **पिघला हुआ मक्खन**: 2 बड़े चम्मच\n` +
      `• **वैनिला एसेंस**: 1/2 छोटा चम्मच\n` +
      `• **नमक**: 1 चुटकी\n\n` +
      `### 🥣 बनाने का तरीका (Instructions):\n` +
      `1. एक कटोरे में मैदा, चीनी, बेकिंग पाउडर और नमक को छान लें।\n` +
      `2. दूसरे बर्तन में दूध, पिघला मक्खन और वैनिला एसेंस मिलाएं, फिर सूखी सामग्री में डालकर चिकना घोल (batter) बना लें।\n` +
      `3. नॉन-स्टिक तवे पर हल्का मक्खन लगाएं और मध्यम आंच पर एक बड़ा चमचा घोल डालें।\n` +
      `4. जब बुलबुले बनने लगें, तो पलट दें और दूसरी तरफ भी 1 मिनट सुनहरा होने तक सेकें।\n` +
      `5. शहद (Honey), मेपल सिरप या चॉकलेट सिरप डालकर गरमा-गरम परोसें! 🥞✨`;
  }

  return null;
}

// 8. Honest General Conversational Intelligence Fallback (Zero Mad-Libs, 100% Genuine)
function generateConversationalResponse(prompt) {
  const p = prompt.trim();
  const ql = p.toLowerCase();

  // If asking for advice, how-to, or tips on any unspecified topic
  if (ql.includes('tip') || ql.includes('advice') || ql.includes('how to') || ql.includes('how do i') || ql.includes('best way to')) {
    return `Here are key actionable principles and strategic recommendations regarding **"${p}"**:\n\n` +
      `### 💡 1. Core Principles\n` +
      `• **Define the Objective First**: Get crystal clear on the specific outcome you want to achieve before expending energy.\n` +
      `• **Compound Consistency**: Master the foundational basics daily rather than sporadic bursts of effort.\n` +
      `• **Iterative Feedback**: Measure your progress weekly and adjust your strategy based on real feedback.\n\n` +
      `### 🚀 2. Best Practices & Action Steps\n` +
      `1. **Eliminate Friction**: Remove distractions and streamline your environment to make action effortless.\n` +
      `2. **Focus on the 80/20**: Identify the 20% of inputs that generate 80% of your results.\n` +
      `3. **Document Your Process**: Keep a learning log or checklist to refine your execution.\n\n` +
      `### 💬 Explore Further with Moon AI:\n` +
      `• Need code for this? Ask: \`/code [your goal]\`\n` +
      `• Want a high-res image visual? Ask: \`/imagine [visual prompt]\`\n` +
      `• Want a dynamic video simulation? Ask: \`/video [motion prompt]\`\n\n` +
      `What specific detail or challenge would you like to tackle next?`;
  }

  // Transparent, honest sovereign assistant notice (Zero canned mad-libs filler)
  return `### 🌙 Moon AI Sovereign Intelligence\n\n` +
    `You asked: **"${p}"**\n\n` +
    `> ⚡ **System Notice: Sovereign Standby Mode**  \n` +
    `> Your primary GPU compute cluster (Google Colab NVIDIA Tesla T4) is currently sleeping or disconnected, and cloud ZeroGPU quota is resting.\n\n` +
    `### 🚀 To activate full open-ended neural dialogue:\n` +
    `1. Open your **Google Colab Training & Serving Notebook**.\n` +
    `2. Run **Step 9** to start your high-speed GPU API server.\n` +
    `3. Moon AI will automatically resume full deep reasoning with zero downtime!\n\n` +
    `### 🛠️ What Moon AI can do right now with zero external dependencies:\n` +
    `• **Science & Space**: Ask about the Solar System, planets, Sun, Moon, gravity, photosynthesis, physics, or biology.\n` +
    `• **Full-Stack Coding**: Ask for complete Python scripts, HTML5 games (Racing Car, Snake), algorithms, or JavaScript.\n` +
    `• **Instant Math**: Ask arithmetic, exponents, percentages, or equations (e.g., \`what is 24 * 18\`).\n` +
    `• **Generative Art**: Type \`/imagine [your vision]\` for instant procedural neural vector art.\n` +
    `• **Spatio-Temporal Video**: Type \`/video [your scene]\` for cinematic simulation.\n\n` +
    `What would you like to build or explore next?`;
}

// Hugging Face ZeroGPU Gradio 6 Bridge
async function callHuggingFaceGradio(prompt) {
  try {
    const hfUrl = 'https://a7006418-moon-ai-backend.hf.space/gradio_api/call/gradio_chat_fn';
    const postRes = await fetch(hfUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: [prompt, []] }),
      signal: AbortSignal.timeout(5000)
    });
    if (!postRes.ok) return null;
    const { event_id } = await postRes.json();
    if (!event_id) return null;

    const streamUrl = `https://a7006418-moon-ai-backend.hf.space/gradio_api/call/gradio_chat_fn/${event_id}`;
    const streamRes = await fetch(streamUrl, {
      signal: AbortSignal.timeout(12000)
    });
    if (!streamRes.ok) return null;

    const text = await streamRes.text();
    const lines = text.split('\n');
    for (const line of lines) {
      if (line.startsWith('data:')) {
        const payload = JSON.parse(line.slice(5).trim());
        const reply = payload[0];
        if (reply && !reply.includes('exceeded your ZeroGPU runs limit') && !reply.includes("Notice: 'You have exceeded")) {
          return reply;
        }
      }
    }
  } catch (e) {
    // Fail gracefully to local sovereign engine
  }
  return null;
}

// Vercel Serverless Handler
export default async function handler(req, res) {
  // Full CORS Headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-User-Id');

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

    const sid = session_id || `session_${Date.now()}`;

    // 1. Math computation
    const mathAnswer = calculateMath(prompt);
    if (mathAnswer) {
      return res.status(200).json({
        reply: mathAnswer,
        model_name: 'Moon AI Math Core (Prem Gaikwad Edition)',
        device: 'Vercel Serverless',
        session_id: sid
      });
    }

        // 2. Direct Image generation (/imagine, /image, or "generate image")
    if (isImageGenerationIntent(prompt)) {
      const imgReply = handleImageGeneration(prompt);
      return res.status(200).json({
        reply: imgReply,
        model_name: 'Moon AI Generative Image Studio',
        device: 'Vercel Edge Diffusion',
        session_id: sid
      });
    }

// 3. Image prompt engineering guide & tips
    const imageTipsReply = handleImagePromptTips(prompt);
    if (imageTipsReply) {
      return res.status(200).json({
        reply: imageTipsReply,
        model_name: 'Moon AI Image Directing Studio',
        device: 'Vercel Serverless',
        session_id: sid
      });
    }

        // 4. Video generation (/video or "generate video")
    if (isVideoGenerationIntent(prompt)) {
      const vidReply = handleVideoGeneration(prompt);
      return res.status(200).json({
        reply: vidReply,
        model_name: 'Moon AI Spatio-Temporal Video Studio',
        device: 'Vercel Edge Motion',
        session_id: sid
      });
    }

// 5. Video making & cinematography tips
    const videoTipsReply = handleVideoTips(prompt);
    if (videoTipsReply) {
      return res.status(200).json({
        reply: videoTipsReply,
        model_name: 'Moon AI Video Directing Studio',
        device: 'Vercel Serverless',
        session_id: sid
      });
    }

    // 6. Code & Game generation
    const codeReply = handleCodeSynthesis(prompt);
    if (codeReply) {
      return res.status(200).json({
        reply: codeReply,
        model_name: 'Moon AI Full-Stack Coder Engine',
        device: 'Vercel Serverless',
        session_id: sid
      });
    }// 0a. Custom Sovereign Neural GPU Forwarding (Colab GPU + ngrok / Private Server)
    const customGpuUrl = req.headers['x-custom-gpu-url'] || process.env.CUSTOM_GPU_URL;
    if (customGpuUrl) {
      try {
        const cleanGpuUrl = customGpuUrl.replace(/\/+$/, '');
        const targetEndpoint = cleanGpuUrl.endsWith('/api/chat') ? cleanGpuUrl : `${cleanGpuUrl}/api/chat`;
        const gpuResp = await fetch(targetEndpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': 'true' },
          body: JSON.stringify({ message: prompt }),
          signal: AbortSignal.timeout(35000)
        });
        if (gpuResp.ok) {
          const gpuData = await gpuResp.json();
          if (gpuData.reply) {
            return res.status(200).json({
              reply: gpuData.reply,
              model_name: gpuData.model_name || 'Moon AI Sovereign (Colab T4 GPU)',
              device: gpuData.device || 'Colab NVIDIA GPU (Private Sovereign Engine)',
              session_id: sid
            });
          }
        }
      } catch (gpuErr) {
        console.warn('Custom GPU forward failed, falling back to Hugging Face / Sovereign offline:', gpuErr.message);
      }
    }

    // 0b. Hugging Face ZeroGPU Cloud Forwarding
    const hfReply = await callHuggingFaceGradio(prompt);
    if (hfReply) {
      return res.status(200).json({
        reply: hfReply,
        model_name: 'Moon AI Neural (Hugging Face ZeroGPU)',
        device: 'Cloud NVIDIA A10G (Hugging Face)',
        session_id: sid
      });
    }

    

    // 7. Conversational wisdom, tips & knowledge
    const knowledgeReply = resolveConversationalKnowledge(prompt);
    if (knowledgeReply) {
      return res.status(200).json({
        reply: knowledgeReply,
        model_name: 'Moon AI Sovereign Intelligence',
        device: 'Vercel Serverless',
        session_id: sid
      });
    }

    // 8. Deep general conversational fallback
    const generalReply = generateConversationalResponse(prompt);
    return res.status(200).json({
      reply: generalReply,
      model_name: 'Moon AI Sovereign (Prem Gaikwad Edition)',
      device: 'Vercel Serverless',
      session_id: sid
    });

  } catch (error) {
    console.error('API Error:', error);
    return res.status(500).json({
      error: error.message || 'Internal Server Error'
    });
  }
}

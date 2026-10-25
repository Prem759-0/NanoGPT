import React, { useState, useEffect, useRef } from 'react';
import {
  ClerkProvider,
  SignedIn,
  SignedOut,
  UserButton,
  SignInButton,
  useUser
} from '@clerk/clerk-react';
import {
  Send,
  Code2,
  Gamepad2,
  Image as ImageIcon,
  Video,
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  Download,
  Play,
  Copy,
  Check,
  X,
  Plus,
  Trash2,
  RefreshCw,
  Cpu,
  Layers,
  Flame,
  Brain,
  Search,
  Sparkles,
  Terminal,
  ExternalLink,
  Key
} from 'lucide-react';
import './App.css';

// Read Clerk key from Vite env or localStorage
const DEFAULT_CLERK_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY || localStorage.getItem('clerk_publishable_key') || '';

// Read API base URL for deployment (defaults to live 24/7 Hugging Face Cloud GPU backend)
const API_BASE = (import.meta.env.VITE_API_URL || 'https://a7006418-moon-ai-backend.hf.space').replace(/\/$/, '');
const resolveMediaUrl = (url) => {
  if (!url) return '';
  if (url.startsWith('http://') || url.startsWith('https://')) return url;
  return `${API_BASE}${url.startsWith('/') ? '' : '/'}${url}`;
};

// Safe API fetch helper that routes directly to Vercel same-origin and sends Gemini Key
const apiFetch = (path, options = {}) => {
  const geminiKey = (typeof localStorage !== 'undefined' ? localStorage.getItem('gemini_api_key') : '') || '';
  const headers = {
    'ngrok-skip-browser-warning': 'true',
    ...(geminiKey ? { 'X-Gemini-Key': geminiKey } : {}),
    ...(options.headers || {})
  };
  const isVercelOrLocal = typeof window !== 'undefined' && (window.location.hostname.includes('vercel.app') || window.location.hostname === 'localhost');
  const targetUrl = (path.startsWith('/api') && isVercelOrLocal) ? path : `${API_BASE}${path}`;
  return fetch(targetUrl, {
    ...options,
    headers
  });
};

// Default high-performance cloud models (so dropdown is never empty)
const DEFAULT_MODELS = [
  {
    id: 'moon-gemini-pro',
    name: '🌙 Moon AI Sovereign (Gemini Pro Engine — By Prem Gaikwad)',
    type: 'cloud_frontier',
    description: 'Frontier reasoning, general knowledge, math, and full-stack coding running 24/7 on Vercel.',
    active: true,
    parameters: 'Frontier Scale',
    available: true
  },
  {
    id: 'qwen-2.5-coder-7b',
    name: '⚡ Moon AI Sovereign (7.6B Cloud — ZeroGPU)',
    type: 'cloud_zerogpu',
    description: 'Hugging Face Space GPU fallback engine.',
    active: false,
    parameters: '7.61B',
    available: true
  }
];

// Direct ZeroGPU generator using Gradio 6's verified native API endpoint
const callGradioChatFn = async (prompt) => {
  const hfBase = API_BASE.includes('hf.space') ? API_BASE : 'https://a7006418-moon-ai-backend.hf.space';
  const postUrl = `${hfBase}/gradio_api/call/v2/gradio_chat_fn`;
  const postRes = await fetch(postUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: prompt })
  });

  if (!postRes.ok) {
    throw new Error(`Cloud engine HTTP ${postRes.status}`);
  }

  const { event_id } = await postRes.json();
  if (!event_id) {
    throw new Error('Cloud engine did not provide an event ID');
  }

  // Gradio 6 official stream GET URL has NO /v2/ in path:
  const streamUrl = `${hfBase}/gradio_api/call/gradio_chat_fn/${event_id}`;
  const streamRes = await fetch(streamUrl);
  if (!streamRes.ok) {
    throw new Error(`Failed to read response stream (HTTP ${streamRes.status})`);
  }

  const text = await streamRes.text();
  const lines = text.split('\n');
  for (const line of lines) {
    if (line.startsWith('data:')) {
      try {
        const parsed = JSON.parse(line.slice(5).trim());
        if (Array.isArray(parsed) && parsed[0]) {
          const content = String(parsed[0]);
          if (content.includes('ZeroGPU runs limit') || content.includes('quota')) {
            throw new Error('ZeroGPU quota limit reached');
          }
          return content;
        }
      } catch (e) {
        if (e.message && e.message.includes('ZeroGPU')) throw e;
      }
    }
  }

  const match = text.match(/data:\s*(\[[\s\S]*?\])/);
  if (match) {
    const arr = JSON.parse(match[1]);
    if (arr[0]) {
      const content = String(arr[0]);
      if (content.includes('ZeroGPU runs limit') || content.includes('quota')) {
        throw new Error('ZeroGPU quota limit reached');
      }
      return content;
    }
  }

  throw new Error('Incomplete response received from cloud engine');
};

export default function App() {
  const [clerkKey, setClerkKey] = useState(DEFAULT_CLERK_KEY);

  if (clerkKey && clerkKey.startsWith('pk_')) {
    return (
      <ClerkProvider publishableKey={clerkKey}>
        <GrokWorkspace clerkKey={clerkKey} onUpdateKey={setClerkKey} />
      </ClerkProvider>
    );
  }

  return <GrokWorkspace clerkKey={clerkKey} onUpdateKey={setClerkKey} isGuestMode={true} />;
}

function GrokWorkspace({ clerkKey, onUpdateKey, isGuestMode }) {
  // Clerk user hook (safe fallback when in guest mode)
  let clerkUser = null;
  try {
    if (!isGuestMode) {
      const auth = useUser();
      clerkUser = auth.user;
    }
  } catch (e) {
    clerkUser = null;
  }

  // Current User ID
  const activeUserId = clerkUser ? clerkUser.id : (localStorage.getItem('nanogpt_user_id') || 'guest_user');

  // State
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(localStorage.getItem('nanogpt_session_id') || null);
  const [messages, setMessages] = useState([]);
  const [inputPrompt, setInputPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationSeconds, setGenerationSeconds] = useState(0);
  const [availableModels, setAvailableModels] = useState(DEFAULT_MODELS);
  const [activeModel, setActiveModel] = useState('moon-gemini-pro');
  const [modelSpecs, setModelSpecs] = useState(null);
  const [geminiKey, setGeminiKey] = useState(localStorage.getItem('gemini_api_key') || '');

  // Grok Signature Modes: 'normal', 'fun' (witty/unhinged), 'think' (deep reasoning)
  const [grokMode, setGrokMode] = useState('normal');

  // Live timer during token generation
  useEffect(() => {
    let interval = null;
    if (isGenerating) {
      interval = setInterval(() => {
        setGenerationSeconds(s => s + 1);
      }, 1000);
    } else {
      setGenerationSeconds(0);
    }
    return () => clearInterval(interval);
  }, [isGenerating]);

  // Artifacts Split-Screen Canvas State
  const [canvasDrawerOpen, setCanvasDrawerOpen] = useState(false);
  const [canvasData, setCanvasData] = useState({ title: 'Grok Canvas', code: '', lang: 'html', tab: 'code' });
  const [codeCopiedIndex, setCodeCopiedIndex] = useState(null);

  // Voice & Audio
  const [isRecording, setIsRecording] = useState(false);
  const [autoSpeak, setAutoSpeak] = useState(false);
  const [speakingIndex, setSpeakingIndex] = useState(null);
  const recognitionRef = useRef(null);
  const isRecordingRef = useRef(false);
  const basePromptRef = useRef('');
  const speechQueueRef = useRef([]);
  const isSpeakingRef = useRef(false);
  const currentUtteranceRef = useRef(null);

  // Initialize SpeechSynthesis Voices & Clean up listeners
  useEffect(() => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.getVoices();
      window.speechSynthesis.onvoiceschanged = () => {
        window.speechSynthesis.getVoices();
      };
    }
    return () => {
      stopSpeaking();
      if (recognitionRef.current) {
        try { recognitionRef.current.stop(); } catch (e) { }
      }
    };
  }, []);

  // References
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isGenerating]);

  // Initial fetch: models, health, and sessions
  useEffect(() => {
    fetchModels();
    fetchModelInfo();
    fetchSessions();
  }, [activeUserId]);

  // Load session messages when active session changes
  useEffect(() => {
    if (currentSessionId) {
      fetchSessionMessages(currentSessionId);
    }
  }, [currentSessionId]);

  // Fetch Available Models
  const fetchModels = async () => {
    try {
      const res = await apiFetch('/api/models');
      if (res.ok) {
        const ct = res.headers.get('content-type') || '';
        if (ct.includes('application/json')) {
          const data = await res.json();
          if (Array.isArray(data) && data.length > 0) {
            setAvailableModels(data);
            const active = data.find(m => m.active);
            if (active) setActiveModel(active.id);
            return;
          }
        }
      }
    } catch (e) {
      console.warn('Failed to fetch models from server:', e);
    }
  };

  // Fetch Model Specs
  const fetchModelInfo = async () => {
    try {
      const res = await apiFetch('/api/model-info');
      if (res.ok) {
        const data = await res.json();
        setModelSpecs(data);
      }
    } catch (e) {
      console.warn('Failed to fetch model info:', e);
    }
  };

  // Fetch Chat Sessions
  const fetchSessions = async () => {
    try {
      const res = await apiFetch(`/api/sessions?user_id=${encodeURIComponent(activeUserId)}`, {
        headers: { 'X-User-Id': activeUserId }
      });
      if (res.ok) {
        const ct = res.headers.get('content-type') || '';
        if (ct.includes('application/json')) {
          const data = await res.json();
          if (Array.isArray(data) && data.length > 0) {
            setSessions(data);
            try { localStorage.setItem(`moon_sessions_${activeUserId}`, JSON.stringify(data)); } catch (e) {}
            if (!currentSessionId) setCurrentSessionId(data[0].id);
            return;
          }
        }
      }
    } catch (e) {
      console.warn('Failed to fetch sessions from server:', e);
    }
    // Fallback to local storage so sidebar is never blank
    try {
      const local = JSON.parse(localStorage.getItem(`moon_sessions_${activeUserId}`) || '[]');
      if (local && local.length > 0) {
        setSessions(local);
        if (!currentSessionId) setCurrentSessionId(local[0].id);
      }
    } catch (e) {}
  };

  // Fetch Messages for a Session
  const fetchSessionMessages = async (sessionId) => {
    if (isGenerating) return; // Guard against overwriting optimistic user prompt while generating
    try {
      const res = await apiFetch(`/api/sessions/${sessionId}`);
      if (res.ok) {
        const ct = res.headers.get('content-type') || '';
        if (ct.includes('application/json')) {
          const data = await res.json();
          if (data.messages && data.messages.length > 0) {
            setMessages(data.messages);
            try { localStorage.setItem(`moon_msgs_${sessionId}`, JSON.stringify(data.messages)); } catch (e) {}
            return;
          }
        }
      }
    } catch (e) {
      console.warn('Failed to load session messages from server:', e);
    }
    // Fallback to local storage
    try {
      const localMsgs = JSON.parse(localStorage.getItem(`moon_msgs_${sessionId}`) || '[]');
      if (localMsgs.length > 0) {
        setMessages(localMsgs);
      }
    } catch (e) {}
  };

  // Switch Active Model
  const handleModelChange = async (newModelId) => {
    setActiveModel(newModelId);
    try {
      const res = await apiFetch('/api/switch-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: newModelId })
      });
      if (res.ok) {
        await fetchModelInfo();
        await fetchModels();
      }
    } catch (e) {
      console.error('Failed to switch model:', e);
    }
  };

  // Start New Chat Session
  const handleNewSession = async () => {
    const newSessionId = `session_${Date.now()}`;
    const newSession = {
      id: newSessionId,
      title: 'New Chat',
      created_at: new Date().toISOString()
    };
    setCurrentSessionId(newSessionId);
    localStorage.setItem('nanogpt_session_id', newSessionId);
    setMessages([]);
    setSessions(prev => {
      const updated = [newSession, ...prev.filter(s => s.id !== newSessionId)];
      try { localStorage.setItem(`moon_sessions_${activeUserId}`, JSON.stringify(updated)); } catch (e) {}
      return updated;
    });

    try {
      await apiFetch('/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-User-Id': activeUserId },
        body: JSON.stringify({ id: newSessionId, title: 'New Chat', user_id: activeUserId })
      });
    } catch (e) {
      console.warn('Could not sync new session to server:', e);
    }
  };

  // Delete Session
  const handleDeleteSession = async (sessionId, e) => {
    e.stopPropagation();
    try {
      await apiFetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
    } catch (e) {
      console.warn('Failed to delete session on server:', e);
    }
    const remaining = sessions.filter(s => s.id !== sessionId);
    setSessions(remaining);
    try { localStorage.setItem(`moon_sessions_${activeUserId}`, JSON.stringify(remaining)); } catch (e) {}
    try { localStorage.removeItem(`moon_msgs_${sessionId}`); } catch (e) {}
    if (currentSessionId === sessionId) {
      if (remaining.length > 0) {
        setCurrentSessionId(remaining[0].id);
        localStorage.setItem('nanogpt_session_id', remaining[0].id);
      } else {
        setCurrentSessionId(null);
        localStorage.removeItem('nanogpt_session_id');
        setMessages([]);
      }
    }
  };

  // --- Natural Text-to-Speech (TTS) Engine ---

  // Clean raw text, code blocks, URLs, and markdown symbols for natural speech synthesis
  const cleanSpeechText = (rawText) => {
    if (!rawText) return '';
    return rawText
      // Replace code blocks with brief spoken indicator
      .replace(/```[\s\S]*?```/g, ' [code block omitted] ')
      // Remove inline code ticks
      .replace(/`([^`]+)`/g, '$1')
      // Clean URLs
      .replace(/https?:\/\/\S+/g, ' link ')
      // Clean markdown links [text](url) -> text
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      // Remove headers (# Title)
      .replace(/^#+\s+/gm, '')
      // Remove bold/italics
      .replace(/[*_~>]/g, '')
      // Remove bullet points
      .replace(/^[-*•]\s+/gm, '')
      // Replace emojis/special symbols that cause synthesizer halts
      .replace(/[^\x00-\x7F]/g, ' ')
      // Normalize whitespace
      .replace(/\s+/g, ' ')
      .trim();
  };

  // Chunk text into sentences to bypass Chrome 15s TTS cutoff bug
  const chunkTextIntoSentences = (text, maxChunkLen = 140) => {
    const rawSentences = text.match(/[^.!?\n]+[.!?\n]+/g) || [text];
    const chunks = [];
    let currentChunk = '';

    for (const s of rawSentences) {
      const trimmed = s.trim();
      if (!trimmed) continue;
      if ((currentChunk + ' ' + trimmed).length <= maxChunkLen) {
        currentChunk = currentChunk ? currentChunk + ' ' + trimmed : trimmed;
      } else {
        if (currentChunk) chunks.push(currentChunk);
        if (trimmed.length > maxChunkLen) {
          const words = trimmed.split(' ');
          let sub = '';
          for (const w of words) {
            if ((sub + ' ' + w).length <= maxChunkLen) {
              sub = sub ? sub + ' ' + w : w;
            } else {
              if (sub) chunks.push(sub);
              sub = w;
            }
          }
          if (sub) chunks.push(sub);
          currentChunk = '';
        } else {
          currentChunk = trimmed;
        }
      }
    }
    if (currentChunk) chunks.push(currentChunk);
    return chunks;
  };

  // Stop any active speech playback
  const stopSpeaking = () => {
    speechQueueRef.current = [];
    isSpeakingRef.current = false;
    currentUtteranceRef.current = null;
    setSpeakingIndex(null);
    if ('speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
      } catch (e) {
        console.error(e);
      }
    }
  };

  // Play next chunk in speech queue
  const playNextSpeechChunk = () => {
    if (!isSpeakingRef.current || speechQueueRef.current.length === 0) {
      stopSpeaking();
      return;
    }

    const nextChunk = speechQueueRef.current.shift();
    if (!nextChunk || !nextChunk.trim()) {
      playNextSpeechChunk();
      return;
    }

    const utterance = new SpeechSynthesisUtterance(nextChunk.trim());
    currentUtteranceRef.current = utterance; // Keep reference to prevent GC in Chrome

    // Select natural English voice if available
    try {
      const voices = window.speechSynthesis.getVoices() || [];
      const englishVoice =
        voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha') || v.name.includes('David') || v.name.includes('Zira'))) ||
        voices.find(v => v.lang.startsWith('en')) ||
        voices[0];
      if (englishVoice) {
        utterance.voice = englishVoice;
      }
    } catch (e) {
      console.warn('Voice picker fallback:', e);
    }

    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    utterance.onend = () => {
      if (isSpeakingRef.current) {
        playNextSpeechChunk();
      }
    };

    utterance.onerror = (e) => {
      console.warn('Speech chunk error:', e);
      if (isSpeakingRef.current) {
        playNextSpeechChunk();
      }
    };

    window.speechSynthesis.speak(utterance);
  };

  // Speak text with message tracking and sentence streaming
  const speakText = (text, messageIndex = null) => {
    if (!('speechSynthesis' in window)) {
      alert('Text-to-speech is not supported by your browser.');
      return;
    }

    // Toggle off if already speaking this message
    if (speakingIndex === messageIndex && messageIndex !== null) {
      stopSpeaking();
      return;
    }

    stopSpeaking();

    const clean = cleanSpeechText(text);
    if (!clean) return;

    const chunks = chunkTextIntoSentences(clean);
    if (chunks.length === 0) return;

    speechQueueRef.current = chunks;
    isSpeakingRef.current = true;
    setSpeakingIndex(messageIndex !== null ? messageIndex : 'auto');

    // Pre-cancel to clear frozen browser state
    try {
      window.speechSynthesis.cancel();
    } catch (e) { }

    playNextSpeechChunk();
  };

  // --- Real-time Continuous Speech-to-Text (STT) Voice Typing ---
  const toggleRecording = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert('Speech recognition is not supported in this browser. Please use modern Google Chrome, Microsoft Edge, or Safari.');
      return;
    }

    if (isRecording || isRecordingRef.current) {
      try {
        recognitionRef.current?.stop();
      } catch (e) {
        console.error(e);
      }
      setIsRecording(false);
      isRecordingRef.current = false;
      return;
    }

    try {
      basePromptRef.current = inputPrompt.trim();
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        setIsRecording(true);
        isRecordingRef.current = true;
      };

      recognition.onresult = (event) => {
        let final = '';
        let interim = '';
        for (let i = 0; i < event.results.length; i++) {
          const item = event.results[i];
          if (item.isFinal) {
            final += item[0].transcript + ' ';
          } else {
            interim += item[0].transcript;
          }
        }
        const prefix = basePromptRef.current ? basePromptRef.current.trim() + ' ' : '';
        const currentText = (prefix + final + interim).replace(/\s+/g, ' ');
        setInputPrompt(currentText);
      };

      recognition.onerror = (event) => {
        console.warn('SpeechRecognition error:', event.error);
        if (event.error === 'not-allowed') {
          alert('🎤 Microphone access was blocked. Please click the permissions/lock icon in your browser address bar and enable microphone access.');
        } else if (event.error === 'audio-capture') {
          alert('🎤 No microphone detected. Please plug in or enable a microphone.');
        } else if (event.error === 'network') {
          alert('⚠️ Speech recognition network service unavailable.');
        }
        setIsRecording(false);
        isRecordingRef.current = false;
      };

      recognition.onend = () => {
        setIsRecording(false);
        isRecordingRef.current = false;
      };

      recognitionRef.current = recognition;
      recognition.start();
    } catch (err) {
      console.error('Failed to start speech recognition:', err);
      setIsRecording(false);
      isRecordingRef.current = false;
    }
  };

  // Send Prompt to AI (supports direct call or form submission)
  const handleSubmit = async (e, directPrompt = null) => {
    if (e && e.preventDefault) e.preventDefault();
    let prompt = (directPrompt !== null ? directPrompt : inputPrompt).trim();
    if (!prompt || isGenerating) return;

    // Halt any active speech and speech-to-text recording
    stopSpeaking();
    if (isRecording || isRecordingRef.current) {
      try { recognitionRef.current?.stop(); } catch (e) { }
      setIsRecording(false);
      isRecordingRef.current = false;
    }

    setInputPrompt('');
    setIsGenerating(true);
    setGenerationSeconds(0);

    // Ensure we have a valid session ID
    let activeSessionId = currentSessionId;
    if (!activeSessionId) {
      activeSessionId = `session_${Date.now()}`;
      setCurrentSessionId(activeSessionId);
      localStorage.setItem('nanogpt_session_id', activeSessionId);
    }

    // Optimistically append user message
    const tempUserMsg = { role: 'user', content: prompt, timestamp: new Date().toISOString() };
    const updatedMessagesWithUser = [...messages, tempUserMsg];
    setMessages(updatedMessagesWithUser);

    // Update or create session title in sidebar
    const cleanTitle = prompt.length > 28 ? prompt.slice(0, 28) + '...' : prompt;
    setSessions(prev => {
      const existing = prev.find(s => s.id === activeSessionId);
      let nextSessions;
      if (existing) {
        nextSessions = prev.map(s => s.id === activeSessionId ? { ...s, title: s.title === 'New Chat' || !s.title ? cleanTitle : s.title } : s);
      } else {
        nextSessions = [{ id: activeSessionId, title: cleanTitle, created_at: new Date().toISOString() }, ...prev];
      }
      try { localStorage.setItem(`moon_sessions_${activeUserId}`, JSON.stringify(nextSessions)); } catch (e) {}
      return nextSessions;
    });

    const isCodeQuery = prompt.toLowerCase().includes('game') || prompt.toLowerCase().includes('code') || prompt.toLowerCase().includes('write') || prompt.startsWith('/code');
    const tokenBudget = isCodeQuery ? 400 : 160;

    // Grok Mode Persona Injection
    let outgoingMessage = prompt;
    if (grokMode === 'fun' && !prompt.startsWith('/')) {
      outgoingMessage = `[Grok Fun Mode: Be witty, razor-sharp, unfiltered, and creative]: ${prompt}`;
    } else if (grokMode === 'think' && !prompt.startsWith('/')) {
      outgoingMessage = `[Grok Deep Think Mode: Provide comprehensive architectural reasoning and step-by-step logic]: ${prompt}`;
    }

    try {
      let replyText = '';
      let meta = {
        model_name: 'Moon AI Sovereign (Gemini / Unified Vercel Engine)',
        tokens: 'Variable',
        device: 'Vercel Serverless'
      };

      let success = false;

      // 1. Unified Vercel Serverless Backend (Instant math, general knowledge, Gemini Pro — No ZeroGPU limits!)
      try {
        const res = await apiFetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-User-Id': activeUserId },
          body: JSON.stringify({
            message: outgoingMessage,
            session_id: activeSessionId,
            user_id: activeUserId,
            max_new_tokens: tokenBudget,
            temperature: grokMode === 'fun' ? 0.85 : 0.65
          })
        });

        if (res.ok) {
          const ct = res.headers.get('content-type') || '';
          if (ct.includes('application/json')) {
            const data = await res.json();
            if (data.reply) {
              replyText = data.reply;
              if (data.model_name) meta.model_name = data.model_name;
              if (data.tokens_generated) meta.tokens = data.tokens_generated;
              if (data.device) meta.device = data.device;
              if (data.session_id && data.session_id !== activeSessionId) {
                activeSessionId = data.session_id;
                setCurrentSessionId(data.session_id);
                localStorage.setItem('nanogpt_session_id', data.session_id);
              }
              success = true;
            }
          }
        }
      } catch (vercelErr) {
        console.warn('Vercel API attempt failed, trying fallback engines:', vercelErr);
      }

      // 2. Fallback to ZeroGPU Hugging Face Space if Vercel serverless wasn't reachable
      if (!success) {
        try {
          replyText = await callGradioChatFn(outgoingMessage);
          meta.model_name = 'Moon AI Sovereign (7.6B Cloud — ZeroGPU)';
          meta.device = 'Cloud NVIDIA GPU';
          success = true;
        } catch (gradioErr) {
          console.warn('Direct Gradio call failed, trying local port 8000:', gradioErr);
          // 3. Fallback to local development server (port 8000)
          const localRes = await fetch('http://localhost:8000/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-User-Id': activeUserId },
            body: JSON.stringify({
              message: outgoingMessage,
              session_id: activeSessionId,
              user_id: activeUserId,
              max_new_tokens: tokenBudget,
              temperature: grokMode === 'fun' ? 0.85 : 0.65
            })
          });

          if (!localRes.ok) {
            throw new Error(`Inference returned status ${localRes.status}`);
          }
          const localData = await localRes.json();
          replyText = localData.reply;
          if (localData.model_name) meta.model_name = localData.model_name;
          success = true;
        }
      }

      const assistantMsg = {
        role: 'assistant',
        content: replyText,
        meta,
        timestamp: new Date().toISOString()
      };

      const finalMessages = [...updatedMessagesWithUser, assistantMsg];
      setMessages(finalMessages);
      try { localStorage.setItem(`moon_msgs_${activeSessionId}`, JSON.stringify(finalMessages)); } catch (e) {}

      // If response includes code or interactive game, auto-open in Grok Canvas drawer
      checkForArtifacts(replyText);

      if (autoSpeak) {
        speakText(replyText);
      }
    } catch (err) {
      console.error('Generation error:', err);
      const errMsg = {
        role: 'assistant',
        content: `⚠️ Error during generation: ${err.message}. Please check connection to cloud GPU server.`,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errMsg]);
    } finally {
      setIsGenerating(false);
    }
  };

  // Detect code blocks to open in Grok Canvas Drawer
  const checkForArtifacts = (replyText) => {
    const codeMatch = replyText.match(/```([a-zA-Z0-9_\-]+)?\s*([\s\S]*?)```/);
    if (codeMatch) {
      const lang = (codeMatch[1] || 'code').toLowerCase();
      const code = codeMatch[2].trim();
      const isHtml = lang === 'html' || code.includes('<html') || code.includes('<canvas');
      setCanvasData({
        title: isHtml ? '🎮 Grok Interactive App' : '🐍 Python Code Studio',
        code: code,
        lang: isHtml ? 'html' : 'python',
        tab: isHtml ? 'preview' : 'code'
      });
      setCanvasDrawerOpen(true);
    }
  };

  // Open specific code snippet in Grok Canvas
  const openSnippetInCanvas = (code, lang = 'html', title = 'Interactive Canvas') => {
    const isHtml = lang === 'html' || code.includes('<html') || code.includes('<canvas');
    setCanvasData({
      title: title,
      code: code,
      lang: isHtml ? 'html' : lang,
      tab: isHtml ? 'preview' : 'code'
    });
    setCanvasDrawerOpen(true);
  };

  // Prompt Clerk Key Input Modal
  const handleConfigureClerk = () => {
    const key = prompt(
      '🔑 Setup Clerk User Authentication (Google / GitHub / Email Login):\n\n' +
      'Paste your Clerk Publishable Key (from https://dashboard.clerk.com):\n' +
      '(Example: pk_test_...)\n\n' +
      'Click Cancel to remain in Sovereign Guest mode.'
    );
    if (key && key.trim().startsWith('pk_')) {
      localStorage.setItem('clerk_publishable_key', key.trim());
      onUpdateKey(key.trim());
    }
  };

  // Prompt Google Gemini Pro Key Input Modal
  const handleConfigureGemini = () => {
    const current = localStorage.getItem('gemini_api_key') || '';
    const key = prompt(
      '🔑 Connect Google Gemini Pro / Flash (Frontier AI Engine):\n\n' +
      'Get your 100% free key with 1,500 requests/day from:\n' +
      '👉 https://aistudio.google.com/app/apikey\n\n' +
      'Paste your Gemini API Key below (starts with AIza...):',
      current
    );
    if (key !== null) {
      if (key.trim()) {
        localStorage.setItem('gemini_api_key', key.trim());
        setGeminiKey(key.trim());
        alert('✅ Gemini Pro Key saved! Moon AI will now run with frontier Gemini 1.5 Pro / Flash reasoning.');
      } else {
        localStorage.removeItem('gemini_api_key');
        setGeminiKey('');
        alert('Gemini key removed. Moon AI is using built-in reasoning engine.');
      }
    }
  };

  // Copy Code to Clipboard
  const handleCopyCode = (text, index) => {
    navigator.clipboard.writeText(text);
    setCodeCopiedIndex(index);
    setTimeout(() => setCodeCopiedIndex(null), 2000);
  };

  // Helper to parse bold-italic (***text***), bold (**bold**), inline code (`code`), links ([text](url)), and italics (*text*)
  const renderInlineMarkdown = (text, parentKey = 'tok') => {
    if (!text) return null;
    const tokens = [];
    const inlineRegex = /(\*\*\*([^*]+)\*\*\*)|(\*\*([^*]+)\*\*)|(`([^`]+)`)|(\[([^\]]+)\]\(([^)]+)\))|(\*([^*]+)\*)/g;
    let lastIdx = 0;
    let match;

    while ((match = inlineRegex.exec(text)) !== null) {
      if (match.index > lastIdx) {
        tokens.push(text.slice(lastIdx, match.index));
      }

      const tokKey = `${parentKey}-${match.index}`;
      if (match[1]) {
        // ***bold-italic***
        tokens.push(<strong key={tokKey} style={{ color: '#ffffff', fontWeight: 700 }}><em>{match[2]}</em></strong>);
      } else if (match[3]) {
        // **bold**
        tokens.push(<strong key={tokKey} style={{ color: '#ffffff', fontWeight: 700 }}>{match[4]}</strong>);
      } else if (match[5]) {
        // `code`
        tokens.push(<code key={tokKey} className="grok-inline-code">{match[6]}</code>);
      } else if (match[7]) {
        // [link](url)
        const label = match[8];
        const href = match[9];
        const isMedia = /\.(mp4|webm|mov|png|jpg|jpeg|webp|gif)(\?|$)/i.test(href);

        if (!isMedia) {
          tokens.push(
            <a
              key={tokKey}
              href={href}
              target="_blank"
              rel="noreferrer"
              className="grok-md-link"
            >
              {label}
            </a>
          );
        }
      } else if (match[10]) {
        // *italic*
        tokens.push(<em key={tokKey} style={{ color: '#e4e4e7' }}>{match[11]}</em>);
      }

      lastIdx = match.index + match[0].length;
    }

    if (lastIdx < text.length) {
      tokens.push(text.slice(lastIdx));
    }

    return tokens;
  };

  // Helper to render text with line breaks, bullets, and embedded dynamic media
  const renderTextWithMedia = (rawText, blockKey) => {
    // 1. Detect video URLs
    const videoMatches = [...rawText.matchAll(/\[(?:[^\]]*)\]\(([^)]+\.(?:mp4|webm|mov)[^)]*)\)|(?:https?:\/\/[^\s)]+|\/static\/[^\s)]+)\.(?:mp4|webm|mov)/gi)];
    const videoUrls = [...new Set(videoMatches.map(m => m[1] || m[0]))];

    // 2. Detect image URLs
    const imageMatches = [...rawText.matchAll(/!\[(?:[^\]]*)\]\(([^)]+)\)|\[(?:[^\]]*)\]\(([^)]+\.(?:png|jpg|jpeg|webp|gif)[^)]*)\)/gi)];
    const imageUrls = [...new Set(imageMatches.map(m => m[1] || m[2]))];

    // 3. Clean raw lines (strip redundant markdown links that are rendered as media players)
    const lines = rawText.split('\n');
    const filteredLines = lines.filter(line => {
      const trimmed = line.trim();
      if (/🎬\s*(\*\*)?Video Clip(\*\*)?:/i.test(trimmed) ||
        /\[Download Full MP4 Video\]/i.test(trimmed) ||
        /^!\[.*?\]\(.*?\)$/i.test(trimmed) ||
        /\[Download Full Resolution Image/i.test(trimmed) ||
        /^\[.*?\.mp4\]\(\/static\/generated\/.*?\.mp4\)$/i.test(trimmed)) {
        return false;
      }
      return true;
    });

    return (
      <div key={blockKey} style={{ lineHeight: 1.68 }}>
        {filteredLines.map((line, lIdx) => {
          const trimmed = line.trim();
          if (!trimmed) {
            return <div key={lIdx} style={{ height: 6 }} />;
          }

          // Bullet items
          if (trimmed.startsWith('•') || trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
            const bulletText = trimmed.replace(/^[•\-*]\s*/, '');
            return (
              <div key={lIdx} className="grok-bullet-row">
                <span className="grok-bullet-symbol">•</span>
                <span style={{ flex: 1 }}>{renderInlineMarkdown(bulletText)}</span>
              </div>
            );
          }

          // Numbered items e.g. "1. ..."
          const numMatch = trimmed.match(/^(\d+\.)\s+(.+)$/);
          if (numMatch) {
            return (
              <div key={lIdx} className="grok-bullet-row">
                <span className="grok-bullet-symbol" style={{ minWidth: 20 }}>{numMatch[1]}</span>
                <span style={{ flex: 1 }}>{renderInlineMarkdown(numMatch[2])}</span>
              </div>
            );
          }

          return (
            <div key={lIdx} style={{ margin: '3px 0' }}>
              {renderInlineMarkdown(line)}
            </div>
          );
        })}

        {/* Embedded Dynamic Video Players */}
        {videoUrls.map((vUrl, vIdx) => {
          const resolvedVUrl = resolveMediaUrl(vUrl);
          return (
            <div key={`video-${vIdx}`} className="grok-media-container">
              <video
                src={resolvedVUrl}
                controls
                playsInline
                preload="metadata"
                className="grok-media-video"
              />
              <div className="grok-media-actions">
                <a href={resolvedVUrl} download className="grok-action-chip" target="_blank" rel="noreferrer">
                  <Download size={12} />
                  <span>Download MP4</span>
                </a>
                <a href={resolvedVUrl} className="grok-action-chip" target="_blank" rel="noreferrer">
                  <ExternalLink size={12} />
                  <span>Full Window</span>
                </a>
              </div>
            </div>
          );
        })}

        {/* Embedded Generated Images */}
        {imageUrls.map((imgUrl, iIdx) => {
          const resolvedImgUrl = resolveMediaUrl(imgUrl);
          return (
            <div key={`image-${iIdx}`} className="grok-media-container">
              <img
                src={resolvedImgUrl}
                alt="Generated visual"
                className="grok-media-image"
                onClick={() => window.open(resolvedImgUrl, '_blank')}
                title="Click to view full resolution"
              />
              <div className="grok-media-actions">
                <a href={resolvedImgUrl} download className="grok-action-chip" target="_blank" rel="noreferrer">
                  <Download size={12} />
                  <span>Download Image</span>
                </a>
                <a href={resolvedImgUrl} className="grok-action-chip" target="_blank" rel="noreferrer">
                  <ExternalLink size={12} />
                  <span>Full Resolution</span>
                </a>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // Render markdown with code block cards, rich media, and live preview triggers
  const renderMessageContent = (content, msgIndex) => {
    const codeBlockRegex = /```([a-zA-Z0-9_\-]+)?\s*([\s\S]*?)```/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = codeBlockRegex.exec(content)) !== null) {
      if (match.index > lastIndex) {
        parts.push({
          type: 'text',
          content: content.slice(lastIndex, match.index)
        });
      }

      const lang = (match[1] || 'code').toLowerCase();
      const code = match[2].trim();
      const isHtml = lang === 'html' || code.includes('<html') || code.includes('<canvas');

      parts.push({
        type: 'code',
        lang: lang,
        code: code,
        isHtml: isHtml
      });

      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < content.length) {
      parts.push({
        type: 'text',
        content: content.slice(lastIndex)
      });
    }

    return (
      <div>
        {parts.map((p, idx) => {
          if (p.type === 'text') {
            return renderTextWithMedia(p.content, `${msgIndex}-txt-${idx}`);
          }

          const blockId = `${msgIndex}-${idx}`;
          return (
            <div key={idx} className="grok-code-block">
              <div className="grok-code-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Terminal size={14} color="#a1a1aa" />
                  <span style={{ fontWeight: 600, textTransform: 'uppercase' }}>{p.lang}</span>
                </div>
                <div className="grok-code-actions">
                  {p.isHtml && (
                    <button
                      className="grok-action-chip"
                      onClick={() => openSnippetInCanvas(p.code, 'html', '🎮 Interactive App')}
                      title="Launch in Grok Canvas"
                    >
                      <Play size={12} color="#22c55e" />
                      <span>Live Preview</span>
                    </button>
                  )}
                  <button
                    className="grok-action-chip"
                    onClick={() => handleCopyCode(p.code, blockId)}
                  >
                    {codeCopiedIndex === blockId ? (
                      <>
                        <Check size={12} color="#22c55e" />
                        <span>Copied</span>
                      </>
                    ) : (
                      <>
                        <Copy size={12} />
                        <span>Copy</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
              <pre className="grok-code-content">
                <code>{p.code}</code>
              </pre>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="app-container">
      {/* Grok Left Sidebar */}
      <aside className="grok-sidebar">
        <div className="sidebar-header">
          <a href="#" className="grok-brand">
            <div className="grok-slash-icon">🌙</div>
            <div>
              <div className="grok-brand-title">MOON AI</div>
              <div className="grok-brand-sub">PREM GAIKWAD EDITION</div>
            </div>
          </a>
        </div>

        {/* New Chat Button */}
        <button className="grok-new-chat-btn" onClick={handleNewSession}>
          <Plus size={16} />
          <span>New Chat</span>
        </button>

        {/* Conversation History */}
        <div className="sidebar-content">
          <div>
            <div className="sidebar-section-title">Active Engine</div>
            <select
              className="grok-model-pill"
              style={{ width: '100%', outline: 'none' }}
              value={activeModel}
              onChange={(e) => handleModelChange(e.target.value)}
            >
              {availableModels.map(m => (
                <option key={m.id} value={m.id} style={{ background: '#09090b', color: '#ffffff' }}>
                  {m.name}
                </option>
              ))}
            </select>
          </div>

          <div style={{ flex: 1 }}>
            <div className="sidebar-section-title">Conversations</div>
            <div className="sessions-list">
              {sessions.length === 0 ? (
                <div style={{ padding: '8px 12px', fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                  No previous sessions
                </div>
              ) : (
                sessions.map(s => (
                  <div
                    key={s.id}
                    className={`session-item ${s.id === currentSessionId ? 'active' : ''}`}
                    onClick={() => {
                      setCurrentSessionId(s.id);
                      localStorage.setItem('nanogpt_session_id', s.id);
                    }}
                  >
                    <span className="session-title-text">{s.title || 'Conversation'}</span>
                    <button className="session-del-btn" onClick={(e) => handleDeleteSession(s.id, e)}>
                      <Trash2 size={13} />
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Model Hardware Specs Card */}
          <div className="grok-specs-card">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
              <span style={{ fontWeight: 700, color: '#ffffff' }}>System Architecture</span>
              <span className="grok-badge-green">
                <span className="grok-badge-dot"></span>
                <span>ONLINE</span>
              </span>
            </div>
            <div>Model: {modelSpecs?.model_name || modelSpecs?.model_type || 'Moon AI Sovereign (7.6B)'}</div>
            <div>Parameters: 7.61 Billion (4-Bit GGUF)</div>
            <div>Storage: Neon PostgreSQL / SQLite</div>
            <div>Cost: $0.00 / Zero Third-Party APIs</div>
          </div>
        </div>
      </aside>

      {/* Main Grok Chat Area */}
      <main className="grok-main-area">
        {/* Topbar */}
        <header className="grok-topbar">
          <div className="topbar-left">
            <div className="grok-model-pill">
              <Sparkles size={14} color="#38bdf8" />
              <span>{activeModel.includes('qwen') ? '🌙 Moon AI Sovereign (7B)' : '⚡ Moon AI Fast (1.2B)'}</span>
            </div>

            {/* Grok Fun Mode & Think Mode Toggles */}
            <button
              className={`grok-mode-toggle ${grokMode === 'fun' ? 'active-fun' : ''}`}
              onClick={() => setGrokMode(prev => prev === 'fun' ? 'normal' : 'fun')}
              title="Toggle Grok Fun / Witty Mode"
            >
              <Flame size={13} />
              <span>Fun Mode</span>
            </button>

            <button
              className={`grok-mode-toggle ${grokMode === 'think' ? 'active-think' : ''}`}
              onClick={() => setGrokMode(prev => prev === 'think' ? 'normal' : 'think')}
              title="Toggle Deep Reasoning Mode"
            >
              <Brain size={13} />
              <span>Think Mode</span>
            </button>

            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.74rem', color: 'var(--text-secondary)', cursor: 'pointer', marginLeft: 6 }}>
              <input
                type="checkbox"
                checked={autoSpeak}
                onChange={(e) => {
                  setAutoSpeak(e.target.checked);
                  if (!e.target.checked) stopSpeaking();
                }}
              />
              <span>🔊 Auto-Speak</span>
            </label>

            {speakingIndex !== null && (
              <button
                className="grok-action-chip"
                onClick={stopSpeaking}
                style={{ borderColor: '#ef4444', color: '#ef4444', background: 'rgba(239, 68, 68, 0.15)', cursor: 'pointer' }}
                title="Stop Speech Playback"
              >
                <VolumeX size={12} />
                <span>Stop Audio</span>
              </button>
            )}
          </div>

          <div className="topbar-right">
            {/* Clerk Authentication Controls */}
            {isGuestMode ? (
              <button className="grok-btn-primary" onClick={handleConfigureClerk}>
                <span>Sign In with Clerk</span>
              </button>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <SignedIn>
                  <UserButton afterSignOutUrl="/" />
                </SignedIn>
                <SignedOut>
                  <SignInButton mode="modal">
                    <button className="grok-btn-primary">Sign In</button>
                  </SignInButton>
                </SignedOut>
              </div>
            )}

            {/* Google Gemini Pro API Key Button */}
            <button
              className="grok-btn-ghost"
              onClick={handleConfigureGemini}
              title="Connect Google Gemini Pro / Flash API Key (1,500 free requests/day)"
              style={geminiKey ? { borderColor: '#22c55e', color: '#22c55e', background: 'rgba(34, 197, 94, 0.1)' } : {}}
            >
              <Key size={13} color={geminiKey ? '#22c55e' : '#a1a1aa'} />
              <span>{geminiKey ? 'Gemini Pro ✓' : 'Gemini Key'}</span>
            </button>

            {/* Canvas Drawer Toggle */}
            <button
              className="grok-btn-ghost"
              onClick={() => setCanvasDrawerOpen(prev => !prev)}
              title="Toggle Grok Canvas / Artifacts Drawer"
            >
              <Layers size={14} />
              <span>Canvas</span>
            </button>
          </div>
        </header>

        {/* Workspace: Message Stream & Split Screen Canvas */}
        <div className="grok-workspace">
          <div className="grok-messages-panel">
            <div className="grok-messages-inner">
              {messages.length === 0 && !isGenerating ? (
                /* Moon AI Hero Screen */
                <div className="grok-hero">
                  <div className="grok-hero-logo">🌙</div>
                  <h1 className="grok-hero-title">What do you want to build with Moon AI?</h1>
                  <p className="grok-hero-desc">
                    Sovereign AI engine created and deployed by <strong>Prem Gaikwad</strong>. Powered by <strong>Moon AI (Qwen-2.5-Coder-7B)</strong> 100% privately on your machine with Clerk authentication, Neon Cloud PostgreSQL, and zero cloud token fees.
                  </p>

                  <div className="grok-hero-cards">
                    <div
                      className="grok-card"
                      onClick={() => handleSubmit(null, 'Write a complete, playable Snake Game in a single HTML, CSS, and JS file')}
                    >
                      <div className="grok-card-icon">🎮</div>
                      <div>
                        <div className="grok-card-title">Playable HTML5 Games</div>
                        <div className="grok-card-desc">Build playable Snake or Chess games with instant live canvas</div>
                      </div>
                    </div>

                    <div
                      className="grok-card"
                      onClick={() => handleSubmit(null, 'Write an ultra-fast Python palindrome verification function with docstrings and doctests')}
                    >
                      <div className="grok-card-icon">💻</div>
                      <div>
                        <div className="grok-card-title">Deep Code & Refactor</div>
                        <div className="grok-card-desc">Frontier algorithms, system architecture, and debugging</div>
                      </div>
                    </div>

                    <div
                      className="grok-card"
                      onClick={() => handleSubmit(null, '/imagine a futuristic cybernetic city at dusk with neon flying speeders, 8k cinematic')}
                    >
                      <div className="grok-card-icon">🎨</div>
                      <div>
                        <div className="grok-card-title">Generative Art Studio</div>
                        <div className="grok-card-desc">Synthesize procedural neural artwork and wallpapers</div>
                      </div>
                    </div>

                    <div
                      className="grok-card"
                      onClick={() => handleSubmit(null, '/video a cinematic drone flight over glowing neon skyscrapers in the rain')}
                    >
                      <div className="grok-card-icon">🎬</div>
                      <div>
                        <div className="grok-card-title">Video Motion Studio</div>
                        <div className="grok-card-desc">Synthesize dynamic MP4 spatio-temporal video clips</div>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                /* Messages Stream */
                messages.map((msg, i) => (
                  <div key={i} className={`grok-msg-row ${msg.role}`}>
                    {msg.role === 'assistant' && (
                      <div className="grok-avatar">🌙</div>
                    )}
                    <div className={msg.role === 'user' ? 'grok-bubble-user' : 'grok-bubble-assistant'}>
                      {renderMessageContent(msg.content, i)}

                      {msg.role === 'assistant' && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 10 }}>
                          <button
                            className="grok-action-chip"
                            onClick={() => speakText(msg.content, i)}
                            title={speakingIndex === i ? "Stop playback" : "Listen to answer (Voice Speech)"}
                            style={speakingIndex === i ? { borderColor: '#38bdf8', color: '#38bdf8', background: 'rgba(56, 189, 248, 0.15)' } : {}}
                          >
                            {speakingIndex === i ? (
                              <>
                                <VolumeX size={12} />
                                <span>Stop Audio</span>
                              </>
                            ) : (
                              <>
                                <Volume2 size={12} />
                                <span>Listen</span>
                              </>
                            )}
                          </button>

                          <button
                            className="grok-action-chip"
                            onClick={() => handleCopyCode(msg.content, `msg-${i}`)}
                            title="Copy reply text"
                          >
                            {codeCopiedIndex === `msg-${i}` ? (
                              <>
                                <Check size={12} color="#22c55e" />
                                <span>Copied</span>
                              </>
                            ) : (
                              <>
                                <Copy size={12} />
                                <span>Copy</span>
                              </>
                            )}
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}

              {/* Generating Spinner */}
              {isGenerating && (
                <div className="grok-msg-row assistant">
                  <div className="grok-avatar">🌙</div>
                  <div className="grok-bubble-assistant" style={{ color: 'var(--color-grok-cyan)', display: 'flex', alignItems: 'center', gap: 10, flexDirection: 'column', alignItems: 'flex-start' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <RefreshCw size={14} className="animate-spin" />
                      <span>
                        {generationSeconds < 5
                          ? 'Moon AI is thinking...'
                          : generationSeconds < 30
                            ? `Moon AI is thinking... (${generationSeconds}s)`
                            : generationSeconds < 90
                              ? `⏳ Loading 7B model into memory... (${generationSeconds}s — first load takes ~60s)`
                              : `🔄 Generating response... (${generationSeconds}s — 7B CPU inference is slow, please wait)`
                        }
                      </span>
                    </div>
                    {generationSeconds >= 30 && (
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', paddingLeft: 24 }}>
                        💡 Tip: For faster replies, ask simple questions. Code generation takes longer.
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Grok Floating Pill Input Bar */}
          <div className="grok-input-container">
            <div
              className="grok-input-card"
              style={isRecording ? { borderColor: '#ef4444', boxShadow: '0 0 18px rgba(239, 68, 68, 0.4)' } : {}}
            >
              <textarea
                ref={textareaRef}
                className="grok-textarea"
                rows={2}
                placeholder={
                  isRecording
                    ? "🎙️ Listening... Speak into your microphone now (click Voice to finish)..."
                    : (grokMode === 'fun'
                      ? "Ask Moon AI anything (Fun Mode 🌶️ is active — expect wit & punch)..."
                      : "Ask Moon AI to code, design a game, or /imagine an image...")
                }
                value={inputPrompt}
                onChange={(e) => setInputPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
              />

              <div className="grok-toolbar">
                <div className="toolbar-left">
                  <button
                    className="toolbar-pill"
                    onClick={() => {
                      setInputPrompt('/code ');
                      textareaRef.current?.focus();
                    }}
                  >
                    <Code2 size={13} />
                    <span>Code</span>
                  </button>

                  <button
                    className="toolbar-pill"
                    onClick={() => {
                      setInputPrompt('/imagine ');
                      textareaRef.current?.focus();
                    }}
                  >
                    <ImageIcon size={13} />
                    <span>Image</span>
                  </button>

                  <button
                    className="toolbar-pill"
                    onClick={() => {
                      setInputPrompt('/video ');
                      textareaRef.current?.focus();
                    }}
                  >
                    <Video size={13} />
                    <span>Video</span>
                  </button>

                  <button
                    className="toolbar-pill"
                    onClick={toggleRecording}
                    style={isRecording ? { borderColor: '#ef4444', color: '#ef4444', background: 'rgba(239, 68, 68, 0.15)' } : {}}
                    title={isRecording ? "Click to stop recording" : "Voice Typing: Click and speak into microphone"}
                  >
                    {isRecording ? <MicOff size={13} /> : <Mic size={13} />}
                    <span>{isRecording ? 'Listening...' : 'Voice'}</span>
                  </button>
                </div>

                <div className="toolbar-right">
                  <button
                    className="grok-send-btn"
                    onClick={handleSubmit}
                    disabled={!inputPrompt.trim() || isGenerating}
                    title="Send to Grok (Enter)"
                  >
                    <Send size={15} />
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Split-Screen Grok Canvas / Artifacts Drawer */}
          {canvasDrawerOpen && (
            <aside className="grok-canvas-drawer">
              <div className="canvas-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: '0.68rem', fontWeight: 700, padding: '2px 8px', background: '#ffffff', color: '#000000', borderRadius: 4 }}>
                    CANVAS
                  </span>
                  <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#ffffff' }}>{canvasData.title}</span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div className="canvas-tabs">
                    <button
                      className={`canvas-tab ${canvasData.tab === 'code' ? 'active' : ''}`}
                      onClick={() => setCanvasData(prev => ({ ...prev, tab: 'code' }))}
                    >
                      Code
                    </button>
                    {canvasData.lang === 'html' && (
                      <button
                        className={`canvas-tab ${canvasData.tab === 'preview' ? 'active' : ''}`}
                        onClick={() => setCanvasData(prev => ({ ...prev, tab: 'preview' }))}
                      >
                        Live Preview
                      </button>
                    )}
                  </div>

                  <button
                    className="grok-action-chip"
                    onClick={() => handleCopyCode(canvasData.code, 'drawer')}
                    title="Copy code"
                  >
                    {codeCopiedIndex === 'drawer' ? <Check size={13} color="#22c55e" /> : <Copy size={13} />}
                  </button>

                  <button
                    className="grok-action-chip"
                    onClick={() => setCanvasDrawerOpen(false)}
                    title="Close Canvas"
                  >
                    <X size={14} />
                  </button>
                </div>
              </div>

              <div className="canvas-body">
                {canvasData.tab === 'preview' && canvasData.lang === 'html' ? (
                  <iframe
                    className="canvas-iframe"
                    title="Grok Live Preview"
                    sandbox="allow-scripts allow-modals"
                    srcDoc={canvasData.code}
                  />
                ) : (
                  <pre className="canvas-code-view">
                    <code>{canvasData.code}</code>
                  </pre>
                )}
              </div>
            </aside>
          )}
        </div>
      </main>
    </div>
  );
}

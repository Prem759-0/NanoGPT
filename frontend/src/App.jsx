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
  ExternalLink
} from 'lucide-react';
import './App.css';

// Read Clerk key from Vite env or localStorage
const DEFAULT_CLERK_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY || localStorage.getItem('clerk_publishable_key') || '';

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
  const [availableModels, setAvailableModels] = useState([]);
  const [activeModel, setActiveModel] = useState('qwen-2.5-coder-7b');
  const [modelSpecs, setModelSpecs] = useState(null);

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
  const recognitionRef = useRef(null);

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
      const res = await fetch('/api/models');
      if (res.ok) {
        const data = await res.json();
        setAvailableModels(data);
        const active = data.find(m => m.active);
        if (active) setActiveModel(active.id);
      }
    } catch (e) {
      console.warn('Failed to fetch models:', e);
    }
  };

  // Fetch Model Specs
  const fetchModelInfo = async () => {
    try {
      const res = await fetch('/api/model-info');
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
      const res = await fetch(`/api/sessions?user_id=${encodeURIComponent(activeUserId)}`, {
        headers: { 'X-User-Id': activeUserId }
      });
      if (res.ok) {
        const data = await res.json();
        setSessions(data);
        if (!currentSessionId && data.length > 0) {
          setCurrentSessionId(data[0].id);
        }
      }
    } catch (e) {
      console.warn('Failed to fetch sessions:', e);
    }
  };

  // Fetch Messages for a Session
  const fetchSessionMessages = async (sessionId) => {
    if (isGenerating) return; // Guard against overwriting optimistic user prompt while generating
    try {
      const res = await fetch(`/api/sessions/${sessionId}`);
      if (res.ok) {
        const data = await res.json();
        setMessages(data.messages || []);
      }
    } catch (e) {
      console.warn('Failed to load session messages:', e);
    }
  };

  // Switch Active Model
  const handleModelChange = async (newModelId) => {
    setActiveModel(newModelId);
    try {
      const res = await fetch('/api/switch-model', {
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
    try {
      const res = await fetch('/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-User-Id': activeUserId },
        body: JSON.stringify({ title: 'New Chat', user_id: activeUserId })
      });
      if (res.ok) {
        const data = await res.json();
        setCurrentSessionId(data.session_id);
        localStorage.setItem('nanogpt_session_id', data.session_id);
        setMessages([]);
        fetchSessions();
      }
    } catch (e) {
      console.error('Failed to create session:', e);
    }
  };

  // Delete Session
  const handleDeleteSession = async (sessionId, e) => {
    e.stopPropagation();
    try {
      await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
      const remaining = sessions.filter(s => s.id !== sessionId);
      setSessions(remaining);
      if (currentSessionId === sessionId) {
        if (remaining.length > 0) {
          setCurrentSessionId(remaining[0].id);
        } else {
          setCurrentSessionId(null);
          setMessages([]);
        }
      }
    } catch (e) {
      console.error('Failed to delete session:', e);
    }
  };

  // Handle Speech-to-Text Voice Input
  const toggleRecording = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert('Speech recognition is supported in modern Chrome & Edge browsers.');
      return;
    }

    if (isRecording) {
      recognitionRef.current?.stop();
      setIsRecording(false);
    } else {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onstart = () => setIsRecording(true);
      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setInputPrompt(prev => (prev ? prev + ' ' : '') + transcript);
      };
      recognition.onerror = () => setIsRecording(false);
      recognition.onend = () => setIsRecording(false);

      recognitionRef.current = recognition;
      recognition.start();
    }
  };

  // Handle Text-to-Speech Output
  const speakText = (text) => {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const clean = text.replace(/[*_#`]/g, '').trim();
    if (!clean) return;
    const utterance = new SpeechSynthesisUtterance(clean);
    utterance.rate = 1.0;
    window.speechSynthesis.speak(utterance);
  };

  // Send Prompt to AI (supports direct call or form submission)
  const handleSubmit = async (e, directPrompt = null) => {
    if (e && e.preventDefault) e.preventDefault();
    let prompt = (directPrompt !== null ? directPrompt : inputPrompt).trim();
    if (!prompt || isGenerating) return;

    setInputPrompt('');
    setIsGenerating(true);
    setGenerationSeconds(0);

    // Optimistically append user message
    const tempUserMsg = { role: 'user', content: prompt, timestamp: new Date().toISOString() };
    setMessages(prev => [...prev, tempUserMsg]);

    const isCodeQuery = prompt.toLowerCase().includes('game') || prompt.toLowerCase().includes('code') || prompt.toLowerCase().includes('write') || prompt.startsWith('/code');
    const tokenBudget = isCodeQuery ? 750 : 320;

    // Grok Mode Persona Injection
    let outgoingMessage = prompt;
    if (grokMode === 'fun' && !prompt.startsWith('/')) {
      outgoingMessage = `[Grok Fun Mode: Be witty, razor-sharp, unfiltered, and creative]: ${prompt}`;
    } else if (grokMode === 'think' && !prompt.startsWith('/')) {
      outgoingMessage = `[Grok Deep Think Mode: Provide comprehensive architectural reasoning and step-by-step logic]: ${prompt}`;
    }

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-User-Id': activeUserId },
        body: JSON.stringify({
          message: outgoingMessage,
          session_id: currentSessionId,
          user_id: activeUserId,
          max_new_tokens: tokenBudget,
          temperature: grokMode === 'fun' ? 0.85 : 0.65
        })
      });

      if (!res.ok) {
        throw new Error(`Inference returned status ${res.status}`);
      }

      const data = await res.json();
      if (data.session_id && data.session_id !== currentSessionId) {
        setCurrentSessionId(data.session_id);
        localStorage.setItem('nanogpt_session_id', data.session_id);
        fetchSessions();
      }

      const assistantMsg = {
        role: 'assistant',
        content: data.reply,
        meta: {
          model_name: data.model_name,
          tokens: data.tokens_generated,
          device: data.device
        },
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, assistantMsg]);

      // If response includes code or interactive game, auto-open in Grok Canvas drawer
      checkForArtifacts(data.reply);

      if (autoSpeak) {
        speakText(data.reply);
      }
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: `⚠️ Error during generation: ${err.message}. Check that the local server is running.`,
          timestamp: new Date().toISOString()
        }
      ]);
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

  // Copy Code to Clipboard
  const handleCopyCode = (text, index) => {
    navigator.clipboard.writeText(text);
    setCodeCopiedIndex(index);
    setTimeout(() => setCodeCopiedIndex(null), 2000);
  };

  // Render markdown with code block cards and live preview triggers
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
            return (
              <div key={idx} style={{ whiteSpace: 'pre-wrap', lineHeight: 1.65 }}>
                {p.content}
              </div>
            );
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
            <div className="grok-slash-icon">//</div>
            <div>
              <div className="grok-brand-title">GROK AI</div>
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
                  {m.id.includes('qwen') ? '👑 Qwen-2.5-Coder-7B' : '⚡ NanoGPT-3.2'}
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
            <div>Model: {modelSpecs?.model_type || 'Qwen-2.5-Coder-7B'}</div>
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
              <span>{activeModel.includes('qwen') ? 'Grok-Coder (Qwen 7B)' : 'NanoGPT Local'}</span>
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
                onChange={(e) => setAutoSpeak(e.target.checked)} 
              />
              <span>🔊 Auto-Speak</span>
            </label>
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
                /* Grok Hero Screen */
                <div className="grok-hero">
                  <div className="grok-hero-logo">//</div>
                  <h1 className="grok-hero-title">What do you want to build today?</h1>
                  <p className="grok-hero-desc">
                    Sovereign AI engine trained and deployed by <strong>Prem Gaikwad</strong>. Running <strong>Qwen-2.5-Coder-7B</strong> 100% privately on your machine with Clerk authentication and zero cloud token fees.
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
                      <div className="grok-avatar">//</div>
                    )}
                    <div className={msg.role === 'user' ? 'grok-bubble-user' : 'grok-bubble-assistant'}>
                      {renderMessageContent(msg.content, i)}
                    </div>
                  </div>
                ))
              )}

              {/* Generating Spinner */}
              {isGenerating && (
                <div className="grok-msg-row assistant">
                  <div className="grok-avatar">//</div>
                  <div className="grok-bubble-assistant" style={{ color: 'var(--color-grok-cyan)', display: 'flex', alignItems: 'center', gap: 10 }}>
                    <RefreshCw size={14} className="animate-spin" />
                    <span>Grok-Coder is thinking... ({generationSeconds}s • CPU AVX2 Local Engine)</span>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Grok Floating Pill Input Bar */}
          <div className="grok-input-container">
            <div className="grok-input-card">
              <textarea
                ref={textareaRef}
                className="grok-textarea"
                rows={2}
                placeholder={
                  grokMode === 'fun' 
                    ? "Ask Grok anything (Fun Mode 🌶️ is active — expect wit & punch)..."
                    : "Ask Grok to code, design a game, or /imagine an image..."
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
                    style={isRecording ? { borderColor: '#ef4444', color: '#ef4444' } : {}}
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

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
  Sparkles, 
  Send, 
  Code2, 
  Gamepad2, 
  Image, 
  Video, 
  Mic, 
  MicOff, 
  Play, 
  Copy, 
  Check, 
  X, 
  PanelRightClose, 
  Plus, 
  Trash2, 
  RefreshCw,
  Cpu,
  Layers,
  Volume2
} from 'lucide-react';
import './App.css';

// Read Clerk key from Vite env or localStorage
const DEFAULT_CLERK_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY || localStorage.getItem('clerk_publishable_key') || '';

export default function App() {
  const [clerkKey, setClerkKey] = useState(DEFAULT_CLERK_KEY);

  if (clerkKey && clerkKey.startsWith('pk_')) {
    return (
      <ClerkProvider publishableKey={clerkKey}>
        <ClaudeWorkspace clerkKey={clerkKey} onUpdateKey={setClerkKey} />
      </ClerkProvider>
    );
  }

  return <ClaudeWorkspace clerkKey={clerkKey} onUpdateKey={setClerkKey} isGuestMode={true} />;
}

function ClaudeWorkspace({ clerkKey, onUpdateKey, isGuestMode }) {
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
  
  // Artifacts Split-Screen Drawer State
  const [artifactDrawerOpen, setArtifactDrawerOpen] = useState(false);
  const [artifactData, setArtifactData] = useState({ title: 'Interactive App', code: '', lang: 'html', tab: 'code' });
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
      alert('Speech recognition is supported in Chrome & Edge browsers.');
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
    const prompt = (directPrompt !== null ? directPrompt : inputPrompt).trim();
    if (!prompt || isGenerating) return;

    setInputPrompt('');
    setIsGenerating(true);
    setGenerationSeconds(0);

    // Optimistically append user message
    const tempUserMsg = { role: 'user', content: prompt, timestamp: new Date().toISOString() };
    setMessages(prev => [...prev, tempUserMsg]);

    const isCodeQuery = prompt.toLowerCase().includes('game') || prompt.toLowerCase().includes('code') || prompt.toLowerCase().includes('write') || prompt.startsWith('/code');
    const tokenBudget = isCodeQuery ? 750 : 300;

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-User-Id': activeUserId },
        body: JSON.stringify({
          message: prompt,
          session_id: currentSessionId,
          user_id: activeUserId,
          max_new_tokens: tokenBudget,
          temperature: 0.7
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

      // If response includes code or interactive game, auto-open in artifact drawer
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

  // Detect code blocks to open in Claude Artifact Drawer
  const checkForArtifacts = (replyText) => {
    const codeMatch = replyText.match(/```([a-zA-Z0-9_\-]+)?\s*([\s\S]*?)```/);
    if (codeMatch) {
      const lang = (codeMatch[1] || 'code').toLowerCase();
      const code = codeMatch[2].trim();
      const isHtml = lang === 'html' || code.includes('<html') || code.includes('<canvas');
      setArtifactData({
        title: isHtml ? '🎮 Interactive Web App' : '🐍 Python Code Studio',
        code: code,
        lang: isHtml ? 'html' : 'python',
        tab: isHtml ? 'preview' : 'code'
      });
      setArtifactDrawerOpen(true);
    }
  };

  // Open specific code snippet in Artifact Drawer
  const openSnippetInArtifact = (code, lang = 'html', title = 'Interactive Artifact') => {
    const isHtml = lang === 'html' || code.includes('<html') || code.includes('<canvas');
    setArtifactData({
      title: title,
      code: code,
      lang: isHtml ? 'html' : lang,
      tab: isHtml ? 'preview' : 'code'
    });
    setArtifactDrawerOpen(true);
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

  // Helper to render message content with rich code cards
  const renderMessageContent = (text, msgIdx) => {
    const codeBlockRegex = /```([a-zA-Z0-9_\-]+)?\s*([\s\S]*?)```/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = codeBlockRegex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        parts.push({ type: 'text', content: text.substring(lastIndex, match.index) });
      }
      parts.push({
        type: 'code',
        lang: (match[1] || 'code').toLowerCase(),
        code: match[2].trim()
      });
      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < text.length) {
      parts.push({ type: 'text', content: text.substring(lastIndex) });
    }

    if (parts.length === 0) {
      return <div className="message-text-content">{text}</div>;
    }

    return (
      <div className="message-content-wrapper">
        {parts.map((p, idx) => {
          if (p.type === 'text') {
            return (
              <div key={idx} className="message-text-paragraph">
                {p.content}
              </div>
            );
          }
          const isHtml = p.lang === 'html' || p.code.includes('<html') || p.code.includes('<canvas');
          return (
            <div key={idx} className="code-card">
              <div className="code-card-header">
                <span>{isHtml ? '🎮 HTML5 Live Sandbox' : `🐍 ${p.lang.toUpperCase()}`}</span>
                <div className="code-actions">
                  <button 
                    className="code-action-btn"
                    onClick={() => openSnippetInArtifact(p.code, p.lang, isHtml ? 'Playable HTML App' : 'Code Artifact')}
                  >
                    ✨ View in Artifact Drawer
                  </button>
                  <button 
                    className="code-action-btn"
                    onClick={() => {
                      navigator.clipboard.writeText(p.code);
                      setCodeCopiedIndex(`${msgIdx}-${idx}`);
                      setTimeout(() => setCodeCopiedIndex(null), 2000);
                    }}
                  >
                    {codeCopiedIndex === `${msgIdx}-${idx}` ? '✓ Copied' : '📋 Copy'}
                  </button>
                </div>
              </div>
              <pre><code>{p.code}</code></pre>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="claude-sidebar">
        <div className="sidebar-top">
          <div className="brand-badge">
            <div className="claude-logo-icon">C</div>
            <div>
              <h1 className="brand-title">Claude AI</h1>
              <span className="brand-subtitle">Prem Gaikwad Edition</span>
            </div>
          </div>
        </div>

        <button className="new-chat-btn" onClick={handleNewSession}>
          <Plus size={15} />
          <span>New Chat</span>
        </button>

        <div className="sidebar-scrollable">
          {/* Active Model Switcher */}
          <div>
            <div className="sidebar-section-title">Active AI Architecture</div>
            <div className="model-picker-card">
              <select 
                className="model-picker-select" 
                value={activeModel} 
                onChange={(e) => handleModelChange(e.target.value)}
              >
                {availableModels.map(m => (
                  <option key={m.id} value={m.id}>{m.name}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Chat Sessions History */}
          <div style={{ flex: 1 }}>
            <div className="sidebar-section-title">Recent Conversations</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
              {sessions.length === 0 ? (
                <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontStyle: 'italic', padding: '6px 8px' }}>
                  No past conversations yet.
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

          {/* Model Specs Card */}
          <div className="model-picker-card" style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
            <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: 6 }}>Model Details:</div>
            <div>Architecture: {modelSpecs?.model_type || 'Qwen-2.5-Coder-7B'}</div>
            <div>Parameters: 7.61 Billion (4-Bit)</div>
            <div>Context: 2,048 Tokens</div>
            <div>Hardware: CPU AVX2 Local Engine</div>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="main-chat-area">
        {/* Topbar */}
        <header className="claude-topbar">
          <div className="topbar-left">
            <div className="topbar-model-badge">
              <Sparkles size={14} color="#d97757" />
              <span>{activeModel.includes('qwen') ? 'Qwen-2.5-Coder-7B' : 'NanoGPT-3.2'}</span>
            </div>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.74rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>
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
              <button className="claude-pill-btn auth-btn" onClick={handleConfigureClerk}>
                <span>Sign In with Clerk</span>
              </button>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <SignedIn>
                  <UserButton afterSignOutUrl="/" />
                </SignedIn>
                <SignedOut>
                  <SignInButton mode="modal">
                    <button className="claude-pill-btn auth-btn">Sign In</button>
                  </SignInButton>
                </SignedOut>
              </div>
            )}

            {/* Split Screen Artifact Drawer Toggle */}
            <button 
              className="claude-pill-btn" 
              onClick={() => setArtifactDrawerOpen(prev => !prev)}
              title="Toggle Claude Code Artifacts Drawer"
            >
              <Layers size={14} />
              <span>Artifacts</span>
            </button>
          </div>
        </header>

        {/* Workspace: Chat Messages Stream + Split-Screen Artifacts Drawer */}
        <div className="chat-workspace">
          <div className="messages-scroll-panel">
            <div className="messages-inner">
              {messages.length === 0 && !isGenerating ? (
                /* Claude Welcome Hero */
                <div className="claude-welcome-hero">
                  <div className="welcome-badge">CLAUDE SOVEREIGN AI • PREM GAIKWAD EDITION</div>
                  <h2 className="claude-hero-title">How can I help you program, analyze, or create today?</h2>
                  <p className="claude-hero-desc">
                    Running <strong>Qwen-2.5-Coder-7B</strong> & <strong>NanoGPT</strong> locally on your CPU with Clerk multi-user authentication and zero third-party cloud API costs.
                  </p>

                  <div className="quick-prompts-grid">
                    <div 
                      className="quick-prompt-card"
                      onClick={() => handleSubmit(null, 'Write a complete, playable Snake Game in a single HTML, CSS, and JS file')}
                    >
                      <div className="quick-prompt-icon">🎮</div>
                      <div>
                        <div className="quick-prompt-title">Playable HTML5 Games</div>
                        <div className="quick-prompt-sub">Build a complete Snake or Chess game in 1 file</div>
                      </div>
                    </div>

                    <div 
                      className="quick-prompt-card"
                      onClick={() => handleSubmit(null, 'Write a Python function to check if a string is a palindrome with docstring and unit tests')}
                    >
                      <div className="quick-prompt-icon">💻</div>
                      <div>
                        <div className="quick-prompt-title">Codex & Claude Coding</div>
                        <div className="quick-prompt-sub">Write algorithms, optimize code, and debug</div>
                      </div>
                    </div>

                    <div 
                      className="quick-prompt-card"
                      onClick={() => handleSubmit(null, '/imagine an ultra-detailed cyberpunk street in Tokyo at midnight, neon reflections, 8k')}
                    >
                      <div className="quick-prompt-icon">🎨</div>
                      <div>
                        <div className="quick-prompt-title">Midjourney Image Studio</div>
                        <div className="quick-prompt-sub">Synthesize cinematic procedural neural artwork</div>
                      </div>
                    </div>

                    <div 
                      className="quick-prompt-card"
                      onClick={() => handleSubmit(null, '/video a cinematic drone flight over glowing neon skyscrapers in the rain')}
                    >
                      <div className="quick-prompt-icon">🎬</div>
                      <div>
                        <div className="quick-prompt-title">Runway Video Studio</div>
                        <div className="quick-prompt-sub">Generate dynamic spatio-temporal MP4 video clips</div>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                /* Chat Messages */
                messages.map((msg, i) => (
                  <div key={i} className={`message-row ${msg.role}`}>
                    {msg.role === 'assistant' && (
                      <div className="assistant-avatar">
                        <Sparkles size={16} />
                      </div>
                    )}
                    <div className="message-bubble">
                      {renderMessageContent(msg.content, i)}
                    </div>
                  </div>
                ))
              )}

              {isGenerating && (
                <div className="message-row assistant">
                  <div className="assistant-avatar">
                    <Sparkles size={16} />
                  </div>
                  <div className="message-bubble" style={{ color: 'var(--color-accent)', display: 'flex', alignItems: 'center', gap: 10 }}>
                    <RefreshCw size={14} className="animate-spin" />
                    <span>Qwen-2.5-Coder-7B is thinking... ({generationSeconds}s • CPU AVX2 Local Engine)</span>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Split-Screen Claude Artifacts Drawer */}
          {artifactDrawerOpen && (
            <aside className="artifacts-drawer">
              <div className="artifacts-drawer-header">
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <span className="artifact-badge">ARTIFACT</span>
                  <span className="artifact-title-text">{artifactData.title}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div className="artifact-tabs">
                    <button 
                      className={`artifact-tab-btn ${artifactData.tab === 'code' ? 'active' : ''}`}
                      onClick={() => setArtifactData(prev => ({ ...prev, tab: 'code' }))}
                    >
                      Code
                    </button>
                    {artifactData.lang === 'html' && (
                      <button 
                        className={`artifact-tab-btn ${artifactData.tab === 'preview' ? 'active' : ''}`}
                        onClick={() => setArtifactData(prev => ({ ...prev, tab: 'preview' }))}
                      >
                        Live Preview 🎮
                      </button>
                    )}
                  </div>
                  <button 
                    className="claude-pill-btn" 
                    style={{ padding: '4px 8px' }}
                    onClick={() => setArtifactDrawerOpen(false)}
                  >
                    <X size={14} />
                  </button>
                </div>
              </div>

              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                {artifactData.tab === 'code' ? (
                  <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: '#141312' }}>
                    <div style={{ padding: '8px 16px', background: 'var(--bg-tertiary)', display: 'flex', gap: 8, borderBottom: '1px solid var(--border-subtle)' }}>
                      <button 
                        className="claude-pill-btn"
                        onClick={() => {
                          navigator.clipboard.writeText(artifactData.code);
                          alert('Code copied to clipboard!');
                        }}
                      >
                        <Copy size={13} />
                        <span>Copy Code</span>
                      </button>
                    </div>
                    <pre style={{ flex: 1, margin: 0, padding: 16, overflow: 'auto', fontFamily: 'var(--font-mono)', fontSize: '0.82rem', color: '#e8e6df' }}>
                      <code>{artifactData.code}</code>
                    </pre>
                  </div>
                ) : (
                  <iframe 
                    className="artifact-preview-iframe"
                    srcDoc={artifactData.code}
                    sandbox="allow-scripts allow-modals allow-same-origin"
                    title="Live App Preview"
                  />
                )}
              </div>
            </aside>
          )}
        </div>

        {/* Floating Claude Input Card */}
        <div className="claude-input-container">
          <form className="claude-input-card" onSubmit={handleSubmit}>
            <textarea
              ref={textareaRef}
              className="claude-textarea"
              placeholder="Ask Claude / Qwen-2.5-Coder to code, design a game, or /imagine an image..."
              rows={2}
              value={inputPrompt}
              onChange={(e) => setInputPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
            />
            <div className="claude-input-toolbar">
              <div className="toolbar-chips">
                <button 
                  type="button" 
                  className="tool-chip"
                  onClick={() => setInputPrompt('/code ')}
                >
                  <Code2 size={13} />
                  <span>Code</span>
                </button>
                <button 
                  type="button" 
                  className="tool-chip"
                  onClick={() => setInputPrompt('/imagine ')}
                >
                  <Image size={13} />
                  <span>Image</span>
                </button>
                <button 
                  type="button" 
                  className="tool-chip"
                  onClick={() => setInputPrompt('/video ')}
                >
                  <Video size={13} />
                  <span>Video</span>
                </button>
                <button 
                  type="button" 
                  className={`tool-chip ${isRecording ? 'active' : ''}`}
                  onClick={toggleRecording}
                  title="Voice Input (Speech-to-Text)"
                >
                  {isRecording ? <MicOff size={13} /> : <Mic size={13} />}
                  <span>{isRecording ? 'Listening...' : 'Mic'}</span>
                </button>
              </div>

              <button 
                type="submit" 
                className="send-action-btn"
                disabled={!inputPrompt.trim() || isGenerating}
                title="Send Message"
              >
                <Send size={15} />
              </button>
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}

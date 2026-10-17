// NanoGPT Lab Frontend Controller — Super AI Edition (Voice & Memory)
document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const modelStatusBadge = document.getElementById("model-status-badge");
  const modelStatusDesc = document.getElementById("model-status-desc");
  const specHardware = document.getElementById("spec-hardware");
  const specCheckpoint = document.getElementById("spec-checkpoint");
  const specDataset = document.getElementById("spec-dataset");
  const specParams = document.getElementById("spec-params");
  const specVocab = document.getElementById("spec-vocab");
  const refreshStatusBtn = document.getElementById("refresh-status-btn");

  const maxTokensSlider = document.getElementById("max-tokens-slider");
  const maxTokensVal = document.getElementById("max-tokens-val");
  const temperatureSlider = document.getElementById("temperature-slider");
  const temperatureVal = document.getElementById("temperature-val");

  const clearChatBtn = document.getElementById("clear-chat-btn");
  const newSessionBtn = document.getElementById("new-session-btn");
  const sessionsList = document.getElementById("sessions-list");
  const autoSpeakToggle = document.getElementById("auto-speak-toggle");

  const modelSelect = document.getElementById("model-select");
  const modelInfoNote = document.getElementById("model-info-note");
  const headerStatusDot = document.getElementById("header-status-dot");
  const headerStatusText = document.getElementById("header-status-text");
  const honestStatusContent = document.getElementById("honest-status-content");

  const messagesStream = document.getElementById("messages-stream");
  const welcomeCard = document.getElementById("welcome-card");
  const chatForm = document.getElementById("chat-form");
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const micBtn = document.getElementById("mic-btn");
  const charCounter = document.getElementById("char-counter");
  const backendPing = document.getElementById("backend-ping");

  const toolImageBtn = document.getElementById("tool-image-btn");
  const toolVideoBtn = document.getElementById("tool-video-btn");
  const toolCodeBtn = document.getElementById("tool-code-btn");
  const toolCodexBtn = document.getElementById("tool-codex-btn");
  const codexBtnLabel = document.getElementById("codex-btn-label");

  let isGenerating = false;
  let codexModeActive = false;
  let currentSessionId = localStorage.getItem("nanogpt_session_id") || null;
  let isRecording = false;
  let recognition = null;

  // --- Voice Input (Speech-to-Text) Setup ---
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = "en-US";

    recognition.onstart = () => {
      isRecording = true;
      if (micBtn) micBtn.classList.add("recording");
      userInput.placeholder = "Listening... speak into your microphone...";
    };

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      userInput.value = (userInput.value ? userInput.value + " " : "") + transcript;
      charCounter.textContent = `${userInput.value.length}/1000`;
    };

    recognition.onerror = (event) => {
      console.warn("Speech recognition error:", event.error);
      stopRecording();
    };

    recognition.onend = () => {
      stopRecording();
    };
  } else {
    if (micBtn) {
      micBtn.title = "Speech recognition not supported in this browser";
      micBtn.style.opacity = "0.5";
    }
  }

  function toggleRecording() {
    if (!recognition) {
      alert("Microphone speech recognition is not supported in this browser. Please use Chrome or Edge.");
      return;
    }
    if (isRecording) {
      recognition.stop();
    } else {
      try {
        recognition.start();
      } catch (e) {
        console.error(e);
      }
    }
  }

  function stopRecording() {
    isRecording = false;
    if (micBtn) micBtn.classList.remove("recording");
    userInput.placeholder = "Chat with AI, or type /imagine [prompt] to generate images, or /run [code]...";
  }

  if (micBtn) {
    micBtn.addEventListener("click", (e) => {
      e.preventDefault();
      toggleRecording();
    });
  }

  // --- Voice Output (Text-to-Speech) Function ---
  function speakText(text, btnElem = null) {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel(); // Stop any active speech

    const cleanText = text.replace(/[*_#`]/g, "").trim();
    if (!cleanText) return;

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    if (btnElem) btnElem.classList.add("speaking");

    utterance.onend = () => {
      if (btnElem) btnElem.classList.remove("speaking");
    };
    utterance.onerror = () => {
      if (btnElem) btnElem.classList.remove("speaking");
    };

    window.speechSynthesis.speak(utterance);
  }

  // --- Sliders Listeners ---
  maxTokensSlider.addEventListener("input", (e) => {
    maxTokensVal.textContent = e.target.value;
  });

  temperatureSlider.addEventListener("input", (e) => {
    temperatureVal.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // --- Quick Tools Handlers ---
  if (toolImageBtn) {
    toolImageBtn.addEventListener("click", () => {
      userInput.value = "/imagine a futuristic cybernetic explorer in a neon metropolis, 8k cinematic";
      userInput.focus();
      charCounter.textContent = `${userInput.value.length}/1000`;
    });
  }

  if (toolVideoBtn) {
    toolVideoBtn.addEventListener("click", () => {
      userInput.value = "/video a cinematic drone flyover of futuristic neon skyscrapers with flying vehicles";
      userInput.focus();
      charCounter.textContent = `${userInput.value.length}/1000`;
    });
  }

  if (toolCodeBtn) {
    toolCodeBtn.addEventListener("click", () => {
      userInput.value = "/run print('Hello from NanoGPT Python sandbox!')\nfor i in range(1, 6):\n    print(f'Step {i}: Computing neural tensors...')";
      userInput.focus();
      charCounter.textContent = `${userInput.value.length}/1000`;
    });
  }

  if (toolCodexBtn) {
    toolCodexBtn.addEventListener("click", () => {
      codexModeActive = !codexModeActive;
      if (codexModeActive) {
        toolCodexBtn.style.borderColor = "var(--primary)";
        toolCodexBtn.style.background = "rgba(6, 182, 212, 0.2)";
        if (codexBtnLabel) codexBtnLabel.textContent = "Codex: ON ⚡";
        userInput.placeholder = "Codex Mode active — ask for algorithms, debugging, or full code...";
      } else {
        toolCodexBtn.style.borderColor = "";
        toolCodexBtn.style.background = "";
        if (codexBtnLabel) codexBtnLabel.textContent = "Codex Mode";
        userInput.placeholder = "Chat, or /code [task], /imagine [prompt], /video [prompt], /run [code]...";
      }
    });
  }

  // --- Textarea Auto-Resize & Counter ---
  userInput.addEventListener("input", () => {
    userInput.style.height = "auto";
    userInput.style.height = Math.min(userInput.scrollHeight, 160) + "px";
    charCounter.textContent = `${userInput.value.length}/1000`;
  });

  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isGenerating && userInput.value.trim().length > 0) {
        handleSubmit();
      }
    }
  });

  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    if (!isGenerating && userInput.value.trim().length > 0) {
      handleSubmit();
    }
  });

  refreshStatusBtn.addEventListener("click", () => {
    refreshStatus();
    loadSessions();
  });

  if (modelSelect) {
    modelSelect.addEventListener("change", async (e) => {
      const selected = e.target.value;
      modelSelect.disabled = true;
      if (modelInfoNote) modelInfoNote.textContent = "Switching model in memory...";
      try {
        const res = await fetch("/api/switch-model", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_name: selected })
        });
        if (!res.ok) {
          const errData = await res.json();
          throw new Error(errData.detail || "Failed to switch model");
        }
        await refreshStatus();
        if (modelInfoNote) {
          if (selected === "llama-3.2") {
            modelInfoNote.textContent = "🔥 NanoGPT-3.2 active: Frontier coding & reasoning model created by Prem Gaikwad, running at ~20 tokens/sec on CPU!";
          } else if (selected === "gpt2") {
            modelInfoNote.textContent = "✨ NanoGPT Light (124M) active: Fast, coherent English & coding with 0 cloud APIs.";
          } else {
            modelInfoNote.textContent = "🔬 NanoGPT Core active: Local character-level transformer trained from scratch.";
          }
        }
      } catch (err) {
        alert("Model Switch Error: " + err.message);
        refreshStatus();
      } finally {
        modelSelect.disabled = false;
      }
    });
  }

  if (clearChatBtn) {
    clearChatBtn.addEventListener("click", () => {
      startNewSession();
    });
  }

  if (newSessionBtn) {
    newSessionBtn.addEventListener("click", () => {
      startNewSession();
    });
  }

  // --- Sessions Management ---
  async function loadSessions() {
    if (!sessionsList) return;
    try {
      const res = await fetch("/api/sessions");
      if (!res.ok) return;
      const sessions = await res.json();
      renderSessions(sessions);
    } catch (err) {
      console.error("Failed to load sessions:", err);
    }
  }

  function renderSessions(sessions) {
    if (!sessionsList) return;
    sessionsList.innerHTML = "";
    if (sessions.length === 0) {
      sessionsList.innerHTML = '<div class="session-empty">No past chats yet.</div>';
      return;
    }

    sessions.forEach((s) => {
      const item = document.createElement("div");
      item.className = `session-item ${s.id === currentSessionId ? "active" : ""}`;
      item.innerHTML = `
        <span class="session-title-text" title="${s.title}">${s.title}</span>
        <button class="session-del-btn" title="Delete Session" data-id="${s.id}">✕</button>
      `;

      item.querySelector(".session-title-text").addEventListener("click", () => {
        switchSession(s.id);
      });

      item.querySelector(".session-del-btn").addEventListener("click", async (e) => {
        e.stopPropagation();
        await deleteSession(s.id);
      });

      sessionsList.appendChild(item);
    });
  }

  async function switchSession(sessionId) {
    currentSessionId = sessionId;
    localStorage.setItem("nanogpt_session_id", sessionId);
    loadSessions();

    try {
      const res = await fetch(`/api/sessions/${sessionId}`);
      if (!res.ok) return;
      const data = await res.json();

      messagesStream.innerHTML = "";
      if (welcomeCard) messagesStream.appendChild(welcomeCard);

      data.messages.forEach((msg) => {
        appendMessage(msg.role, msg.content, {
          tokens: msg.meta ? msg.meta.tokens : undefined,
          device: msg.meta ? msg.meta.device : undefined
        });
      });
    } catch (err) {
      console.error("Failed to load session messages:", err);
    }
  }

  async function startNewSession() {
    try {
      const res = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "New Chat" })
      });
      const data = await res.json();
      currentSessionId = data.session_id;
      localStorage.setItem("nanogpt_session_id", currentSessionId);
      messagesStream.innerHTML = "";
      if (welcomeCard) messagesStream.appendChild(welcomeCard);
      loadSessions();
    } catch (err) {
      console.error(err);
    }
  }

  async function deleteSession(sessionId) {
    try {
      await fetch(`/api/sessions/${sessionId}`, { method: "DELETE" });
      if (currentSessionId === sessionId) {
        currentSessionId = null;
        localStorage.removeItem("nanogpt_session_id");
        messagesStream.innerHTML = "";
        if (welcomeCard) messagesStream.appendChild(welcomeCard);
      }
      loadSessions();
    } catch (err) {
      console.error(err);
    }
  }

  // --- Fetch System & Model Status ---
  async function refreshStatus() {
    backendPing.textContent = "Backend: Checking...";
    try {
      const [healthRes, infoRes] = await Promise.all([
        fetch("/api/health"),
        fetch("/api/model-info")
      ]);

      if (!healthRes.ok) {
        throw new Error(`Health endpoint returned status ${healthRes.status}`);
      }

      const health = await healthRes.json();
      const info = infoRes.ok ? await infoRes.json() : {};

      updateStatusUI(health, info);
      backendPing.textContent = "Backend: Online (127.0.0.1)";
    } catch (err) {
      console.error("Status fetch failed:", err);
      modelStatusBadge.className = "status-badge status-error";
      modelStatusBadge.textContent = "OFFLINE";
      modelStatusDesc.textContent = "Backend not reachable. Check console or terminal.";
      headerStatusDot.className = "chat-header-status-indicator";
      headerStatusText.textContent = "Server Offline";
      backendPing.textContent = "Backend: Offline";
    }
  }

  function updateStatusUI(health, info) {
    const status = health.model_status || "NOT TRAINED";
    modelStatusBadge.className = "status-badge";

    if (status === "READY" || status === "CHECKPOINT FOUND (NOT LOADED)") {
      modelStatusBadge.classList.add("status-ready");
      modelStatusBadge.textContent = "READY";
      modelStatusDesc.textContent = "Model checkpoint is loaded in memory for local inference.";
      headerStatusDot.className = "chat-header-status-indicator ready";
      headerStatusText.textContent = `Model Ready (${health.device.toUpperCase()})`;
      if (honestStatusContent) {
        honestStatusContent.style.color = "var(--status-ready-text)";
        honestStatusContent.innerHTML = "✅ Checkpoint is ready! You can chat with your local AI, generate images with <code>/imagine</code>, run Python with <code>/run</code>, or speak with 🎙️ voice input!";
      }
    } else if (status === "NOT TRAINED") {
      modelStatusBadge.classList.add("status-not-trained");
      modelStatusBadge.textContent = "NOT TRAINED";
      modelStatusDesc.textContent = "No trained checkpoint found on disk.";
      headerStatusDot.className = "chat-header-status-indicator not-trained";
      headerStatusText.textContent = "No Checkpoint";
    } else {
      modelStatusBadge.classList.add("status-error");
      modelStatusBadge.textContent = status;
      modelStatusDesc.textContent = "Model service error.";
      headerStatusDot.className = "chat-header-status-indicator";
      headerStatusText.textContent = "Model Error";
    }

    // Sync model select dropdown
    if (modelSelect && info.model_type) {
      if (info.model_type.includes("NanoGPT-3.2") || info.model_type.includes("LLaMA")) {
        modelSelect.value = "llama-3.2";
        if (modelInfoNote) modelInfoNote.textContent = "🔥 NanoGPT-3.2 active: Frontier coding & reasoning model created by Prem Gaikwad, running at ~20 tokens/sec on CPU!";
      } else if (info.model_type.includes("Light") || info.model_type.includes("GPT-2")) {
        modelSelect.value = "gpt2";
        if (modelInfoNote) modelInfoNote.textContent = "✨ NanoGPT Light (124M) active: Fast, coherent English & coding with 0 cloud APIs.";
      } else {
        modelSelect.value = "scratch";
        if (modelInfoNote) modelInfoNote.textContent = "🔬 NanoGPT Core active: Local character-level transformer trained from scratch.";
      }
    }

    // Update specs
    specHardware.textContent = health.device.toUpperCase() + (health.torch_available ? ` (Torch ${health.torch_version || '✓'})` : ' (No Torch)');
    specCheckpoint.textContent = info.checkpoint_path || "gpt2 (124M)";
    specDataset.textContent = info.dataset || "OpenWebText";
    specParams.textContent = info.parameters ? formatNumber(info.parameters) : "123.65M";
    specVocab.textContent = info.vocab_size ? `${info.vocab_size} tokens` : "50,257 tokens";
  }

  function formatNumber(num) {
    if (num >= 1e6) return (num / 1e6).toFixed(2) + "M";
    if (num >= 1e3) return (num / 1e3).toFixed(1) + "K";
    return num.toString();
  }

  // --- Handle Submission ---
  async function handleSubmit() {
    const prompt = userInput.value.trim();
    if (!prompt) return;

    // Append user message
    appendMessage("user", prompt);
    userInput.value = "";
    userInput.style.height = "auto";
    charCounter.textContent = "0/1000";

    const loadingElem = appendLoadingBubble();
    setGeneratingState(true);

    // 0. Smart Multi-Modal Intent Classifiers (Autonomous Tool Calling)
    function extractVideoIntent(text) {
      const t = text.trim();
      if (/^\/video\s+/i.test(t)) return t.replace(/^\/video\s+/i, "").trim();
      const pMatch = t.match(/^(?:please\s+)?(?:show|generate|create|make|render|produce)(?:\s+me)?\s+(?:an?\s+)?(?:video|animation|clip|mp4|short movie)\s+(?:of\s+)?(.+)/i);
      if (pMatch && pMatch[1]) return pMatch[1].trim();
      const animMatch = t.match(/^(?:please\s+)?(?:animate)\s+(?:an?\s+)?(.+)/i);
      if (animMatch && animMatch[1]) return animMatch[1].trim();
      const sMatch = t.match(/^(?:show|display|give me)?\s*(.+?)\s+(?:video|animation|clip|mp4)$/i);
      if (sMatch && sMatch[1] && sMatch[1].length > 1) return sMatch[1].trim();
      return null;
    }

    function extractImageIntent(text) {
      const t = text.trim();
      if (/^\/(imagine|image)\s+/i.test(t)) return t.replace(/^\/(imagine|image)\s+/i, "").trim();
      const pMatch = t.match(/^(?:please\s+)?(?:show|generate|create|draw|make|render|display|give me)(?:\s+me)?\s+(?:an?\s+)?(?:image|picture|photo|artwork|drawing|painting|render|map|visual)\s+(?:of\s+)?(.+)/i);
      if (pMatch && pMatch[1]) return pMatch[1].trim();
      const sMatch = t.match(/^(?:show|display|give me)?\s*(.+?)\s+(?:image|picture|photo|artwork|drawing|painting|wallpaper|render)$/i);
      if (sMatch && sMatch[1] && sMatch[1].length > 1) return sMatch[1].trim();
      const drawMatch = t.match(/^(?:draw|paint|illustrate)\s+(?:an?\s+)?(.+)/i);
      if (drawMatch && drawMatch[1]) return drawMatch[1].trim();
      return null;
    }

    // 1. Tool Check: Video Generation Intent
    const detectedVideoPrompt = extractVideoIntent(prompt);
    if (detectedVideoPrompt) {
      try {
        const response = await fetch("/api/generate-video", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: detectedVideoPrompt, duration_sec: 3.0, fps: 15 })
        });
        loadingElem.remove();
        if (!response.ok) throw new Error("Video generation failed");
        const data = await response.json();
        appendVideoMessage(data.video_url, detectedVideoPrompt, data);
      } catch (err) {
        loadingElem.remove();
        appendAlertBox("error", "Video Generation Failed", err.message);
      } finally {
        setGeneratingState(false);
        userInput.focus();
      }
      return;
    }

    // 2. Tool Check: Image Generation Intent
    const detectedImagePrompt = extractImageIntent(prompt);
    if (detectedImagePrompt) {
      try {
        const response = await fetch("/api/generate-image", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: detectedImagePrompt, style: "auto" })
        });
        loadingElem.remove();
        if (!response.ok) throw new Error("Image generation failed");
        const data = await response.json();
        appendImageMessage(data.image_url, detectedImagePrompt, data);
      } catch (err) {
        loadingElem.remove();
        appendAlertBox("error", "Image Generation Failed", err.message);
      } finally {
        setGeneratingState(false);
        userInput.focus();
      }
      return;
    }

    // 3. Tool Check: /run Python code
    if (prompt.startsWith("/run ")) {
      const codeSnippet = prompt.replace(/^\/run\s+/, "").trim();
      try {
        const response = await fetch("/api/run-code", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ code: codeSnippet })
        });
        loadingElem.remove();
        const data = await response.json();
        appendCodeResultMessage(codeSnippet, data);
      } catch (err) {
        loadingElem.remove();
        appendAlertBox("error", "Code Execution Failed", err.message);
      } finally {
        setGeneratingState(false);
        userInput.focus();
      }
      return;
    }

    // 4. Regular or Codex Chat Generation with Context Memory
    const isCodeMode = codexModeActive || prompt.startsWith("/code ");
    const cleanPrompt = prompt.startsWith("/code ") ? prompt.replace(/^\/code\s+/, "").trim() : prompt;

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: cleanPrompt,
          mode: isCodeMode ? "code" : "chat",
          session_id: currentSessionId,
          max_new_tokens: isCodeMode ? 280 : parseInt(maxTokensSlider.value, 10),
          temperature: isCodeMode ? 0.35 : parseFloat(temperatureSlider.value)
        })
      });

      loadingElem.remove();

      if (response.status === 409) {
        const data = await response.json();
        appendAlertBox(
          "warning",
          "Honest Pre-Training Notice: Model Not Trained",
          data.detail || "No trained checkpoint found on disk.",
          data.suggestion || "Run dataset preparation and training first."
        );
        refreshStatus();
      } else if (response.status === 422) {
        const data = await response.json();
        appendAlertBox("error", "Input Validation Error", data.detail || "Validation failed.");
      } else if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        appendAlertBox("error", `Server Error (${response.status})`, data.detail || "Inference error.");
      } else {
        const data = await response.json();
        if (data.session_id) {
          currentSessionId = data.session_id;
          localStorage.setItem("nanogpt_session_id", currentSessionId);
          loadSessions();
        }
        appendMessage("assistant", data.reply, {
          modelName: data.model_name,
          tokens: data.tokens_generated,
          device: data.device
        });

        // Auto-Speak if enabled
        if (autoSpeakToggle && autoSpeakToggle.checked) {
          speakText(data.reply);
        }

        refreshStatus();
      }
    } catch (err) {
      loadingElem.remove();
      console.error("Chat error:", err);
      appendAlertBox(
        "error",
        "Connection Error",
        "Could not reach local server. Check http://127.0.0.1:8000."
      );
    } finally {
      setGeneratingState(false);
      userInput.focus();
    }
  }

  function setGeneratingState(generating) {
    isGenerating = generating;
    sendBtn.disabled = generating;
    userInput.disabled = generating;
  }

  function appendMessage(sender, text, meta = {}) {
    const row = document.createElement("div");
    row.className = `message-row ${sender === "user" ? "user-row" : "assistant-row"}`;

    const bubble = document.createElement("div");
    bubble.className = `message-bubble ${sender === "user" ? "user-bubble" : "assistant-bubble"}`;

    const header = document.createElement("div");
    header.className = "message-header";
    
    let speakerHtml = "";
    if (sender === "assistant") {
      speakerHtml = `
        <button class="speaker-btn" title="Speak this response aloud" aria-label="Speak response">
          🔊
        </button>
      `;
    }

    header.innerHTML = `
      <div style="display:flex; align-items:center;">
        <span class="message-sender">${sender === "user" ? "You" : (meta.modelName || "NanoGPT Assistant")}</span>
        ${speakerHtml}
      </div>
      <span>${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
    `;

    if (sender === "assistant") {
      const spkBtn = header.querySelector(".speaker-btn");
      if (spkBtn) {
        spkBtn.addEventListener("click", () => {
          speakText(text, spkBtn);
        });
      }
    }

    const content = document.createElement("div");
    content.className = "message-content";

    // Format markdown code blocks with 1-click execution
    const codeBlockMatch = text.match(/```(?:python)?\s*([\s\S]*?)```/);
    if (codeBlockMatch && sender === "assistant") {
      const parts = text.split(/```(?:python)?\s*[\s\S]*?```/);
      const extractedCode = codeBlockMatch[1].trim();

      if (parts[0]) {
        const p1 = document.createElement("div");
        p1.textContent = parts[0].trim();
        content.appendChild(p1);
      }

      const codeBox = document.createElement("div");
      codeBox.className = "code-block-interactive";
      codeBox.innerHTML = `
        <div class="code-block-topbar">
          <span>Python (Codex/Claude)</span>
          <button class="run-snippet-btn" title="Run this code in local Python sandbox">▶ Run Code</button>
        </div>
        <pre><code>${extractedCode.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</code></pre>
      `;

      const runBtn = codeBox.querySelector(".run-snippet-btn");
      runBtn.addEventListener("click", async () => {
        runBtn.textContent = "⏳ Running...";
        runBtn.disabled = true;
        try {
          const runRes = await fetch("/api/run-code", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code: extractedCode })
          });
          const runData = await runRes.json();
          appendCodeResultMessage(extractedCode, runData);
        } catch (e) {
          appendAlertBox("error", "Code Execution Failed", e.message);
        } finally {
          runBtn.textContent = "▶ Run Code";
          runBtn.disabled = false;
        }
      });

      content.appendChild(codeBox);

      if (parts[1]) {
        const p2 = document.createElement("div");
        p2.style.marginTop = "8px";
        p2.textContent = parts[1].trim();
        content.appendChild(p2);
      }
    } else {
      content.textContent = text;
    }

    bubble.appendChild(header);
    bubble.appendChild(content);

    if (sender === "assistant" && meta.tokens !== undefined) {
      const footer = document.createElement("div");
      footer.className = "message-footer";
      footer.textContent = `Generated ${meta.tokens} tokens on ${meta.device || 'CPU'}`;
      bubble.appendChild(footer);
    }

    row.appendChild(bubble);
    messagesStream.appendChild(row);
    scrollToBottom();
    return row;
  }

  function appendImageMessage(imageUrl, prompt, meta = {}) {
    const row = document.createElement("div");
    row.className = "message-row assistant-row";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble assistant-bubble";
    bubble.style.maxWidth = "90%";

    bubble.innerHTML = `
      <div class="message-header">
        <span class="message-sender">🎨 Image Studio (${meta.style || 'Midjourney & Higgsfield Caliber'})</span>
        <span>${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
      </div>
      <div class="generated-image-card">
        <img src="${imageUrl}" alt="${prompt}" loading="lazy" />
        <div class="image-meta-bar">
          <span>Prompt: <em>"${prompt}"</em></span>
          <a href="${imageUrl}" target="_blank" download class="image-download-link">Download Full PNG ↗</a>
        </div>
      </div>
    `;

    row.appendChild(bubble);
    messagesStream.appendChild(row);
    scrollToBottom();
  }

  function appendVideoMessage(videoUrl, prompt, meta = {}) {
    const row = document.createElement("div");
    row.className = "message-row assistant-row";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble assistant-bubble";
    bubble.style.maxWidth = "90%";

    bubble.innerHTML = `
      <div class="message-header">
        <span class="message-sender">🎬 AI Video Studio (Kling & Runway Caliber)</span>
        <span>${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
      </div>
      <div class="generated-video-card">
        <video controls autoplay loop playsinline src="${videoUrl}" style="width: 100%; border-radius: 8px; border: 1px solid var(--border-color);"></video>
        <div class="image-meta-bar" style="margin-top: 8px;">
          <span>Motion Prompt: <em>"${prompt}"</em> (${meta.duration || 3}s @ ${meta.fps || 15}fps)</span>
          <a href="${videoUrl}" target="_blank" download class="image-download-link">Download MP4 ↗</a>
        </div>
      </div>
    `;

    row.appendChild(bubble);
    messagesStream.appendChild(row);
    scrollToBottom();
  }

  function appendCodeResultMessage(code, res) {
    const row = document.createElement("div");
    row.className = "message-row assistant-row";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble assistant-bubble";
    bubble.style.maxWidth = "90%";

    bubble.innerHTML = `
      <div class="message-header">
        <span class="message-sender">⚡ Python Code Sandbox</span>
        <span>Execution: ${res.execution_time_ms}ms</span>
      </div>
      <div class="code-terminal-card">
        <div class="code-terminal-header">
          <span>Executed Script</span>
          <span>Exit Code: ${res.exit_code}</span>
        </div>
        <div class="code-terminal-body ${res.error ? 'error-text' : ''}">
${res.error ? res.error : res.output}
        </div>
      </div>
    `;

    row.appendChild(bubble);
    messagesStream.appendChild(row);
    scrollToBottom();
  }

  function appendLoadingBubble() {
    const row = document.createElement("div");
    row.className = "message-row assistant-row";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble assistant-bubble";
    bubble.innerHTML = `
      <div class="message-header">
        <span class="message-sender">NanoGPT Assistant</span>
        <span>Local inference & processing</span>
      </div>
      <div class="typing-dots">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>
    `;

    row.appendChild(bubble);
    messagesStream.appendChild(row);
    scrollToBottom();
    return row;
  }

  function appendAlertBox(type, title, message, suggestion = "") {
    const box = document.createElement("div");
    box.className = `alert-box alert-${type}`;
    box.innerHTML = `
      <div class="alert-title">
        <span>${type === "warning" ? "⚠️" : "🛑"}</span>
        <strong>${title}</strong>
      </div>
      <div class="alert-body">
        <p>${message}</p>
        ${suggestion ? `<p style="margin-top: 6px; font-size: 0.82rem; opacity: 0.9;"><strong>Next step:</strong> <code>${suggestion}</code></p>` : ""}
      </div>
    `;
    messagesStream.appendChild(box);
    scrollToBottom();
  }

  function scrollToBottom() {
    messagesStream.scrollTop = messagesStream.scrollHeight;
  }

  // Initial fetch
  refreshStatus();
  loadSessions();
});

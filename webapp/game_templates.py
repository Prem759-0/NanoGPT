"""
Interactive HTML5 / Canvas / JavaScript Game Templates for NanoGPT Lab.
High-fidelity, zero-dependency, single-file games playable in browser sandbox.
"""

def get_snake_game_html() -> str:
    return '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Cyber Snake 2026 - NanoGPT Edition</title>
  <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      user-select: none;
    }
    body {
      background: radial-gradient(circle at center, #0f172a 0%, #020617 100%);
      color: #38bdf8;
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      padding: 12px;
    }
    .arcade-card {
      background: rgba(15, 23, 42, 0.95);
      border: 2px solid #38bdf8;
      border-radius: 16px;
      box-shadow: 0 0 30px rgba(56, 189, 248, 0.3);
      padding: 16px;
      display: flex;
      flex-direction: column;
      align-items: center;
      max-width: 440px;
      width: 100%;
    }
    .scoreboard {
      width: 100%;
      display: flex;
      justify-content: space-between;
      margin-bottom: 12px;
      font-weight: 700;
      font-size: 15px;
      letter-spacing: 0.5px;
    }
    .score-val { color: #4ade80; text-shadow: 0 0 10px rgba(74, 222, 128, 0.5); }
    .high-val { color: #f43f5e; text-shadow: 0 0 10px rgba(244, 63, 94, 0.5); }
    canvas {
      background: #020617;
      border: 2px solid #1e293b;
      border-radius: 10px;
      box-shadow: inset 0 0 20px rgba(0,0,0,0.8);
      display: block;
    }
    .controls {
      display: flex;
      gap: 10px;
      margin-top: 14px;
      width: 100%;
    }
    .game-btn {
      flex: 1;
      background: rgba(56, 189, 248, 0.15);
      border: 1px solid #38bdf8;
      color: #38bdf8;
      font-weight: 700;
      font-size: 14px;
      padding: 8px 14px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .game-btn:hover {
      background: #38bdf8;
      color: #020617;
      box-shadow: 0 0 15px #38bdf8;
    }
    .d-pad {
      display: grid;
      grid-template-columns: repeat(3, 44px);
      grid-gap: 6px;
      margin-top: 12px;
    }
    .d-pad button {
      background: #1e293b;
      border: 1px solid #334155;
      color: #38bdf8;
      font-size: 18px;
      height: 44px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: bold;
    }
    .d-pad button:active {
      background: #38bdf8;
      color: #020617;
    }
    .tip {
      margin-top: 10px;
      font-size: 12px;
      color: #64748b;
      text-align: center;
    }
  </style>
</head>
<body>
  <div class="arcade-card">
    <div class="scoreboard">
      <div>SCORE: <span id="score" class="score-val">0</span></div>
      <div>HIGH: <span id="high" class="high-val">0</span></div>
    </div>
    
    <canvas id="gameCanvas" width="400" height="400"></canvas>

    <div class="controls">
      <button id="startBtn" class="game-btn">▶ Restart Game</button>
      <button id="pauseBtn" class="game-btn">⏸ Pause</button>
    </div>

    <!-- On-screen controls for mobile or mouse -->
    <div class="d-pad">
      <div></div>
      <button id="btnUp">▲</button>
      <div></div>
      <button id="btnLeft">◀</button>
      <button id="btnDown">▼</button>
      <button id="btnRight">▶</button>
    </div>

    <div class="tip">Controls: <b>Arrow Keys</b> or <b>W, A, S, D</b> | Spacebar to Pause</div>
  </div>

  <script>
    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas.getContext('2d');
    const scoreEl = document.getElementById('score');
    const highEl = document.getElementById('high');
    const startBtn = document.getElementById('startBtn');
    const pauseBtn = document.getElementById('pauseBtn');

    const gridSize = 20;
    const tileCount = canvas.width / gridSize; // 20x20 tiles

    let snake = [{ x: 10, y: 10 }, { x: 9, y: 10 }, { x: 8, y: 10 }];
    let velocity = { x: 1, y: 0 };
    let nextVelocity = { x: 1, y: 0 };
    let food = { x: 15, y: 10 };
    let score = 0;
    let highScore = localStorage.getItem('nano_snake_highscore') || 0;
    highEl.textContent = highScore;

    let isGameOver = false;
    let isPaused = false;
    let gameLoopId = null;
    let gameSpeed = 110; // ms per tick

    // Web Audio Sound FX Synthesizer (0 external files needed)
    let audioCtx = null;
    function playSound(freq, duration, type = 'sine') {
      try {
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (audioCtx.state === 'suspended') audioCtx.resume();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
        gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
      } catch (e) {}
    }

    function spawnFood() {
      while (true) {
        food = {
          x: Math.floor(Math.random() * tileCount),
          y: Math.floor(Math.random() * tileCount)
        };
        const collides = snake.some(s => s.x === food.x && s.y === food.y);
        if (!collides) break;
      }
    }

    function resetGame() {
      snake = [{ x: 10, y: 10 }, { x: 9, y: 10 }, { x: 8, y: 10 }];
      velocity = { x: 1, y: 0 };
      nextVelocity = { x: 1, y: 0 };
      score = 0;
      scoreEl.textContent = score;
      isGameOver = false;
      isPaused = false;
      pauseBtn.textContent = '⏸ Pause';
      spawnFood();
      if (gameLoopId) clearInterval(gameLoopId);
      gameLoopId = setInterval(gameStep, gameSpeed);
    }

    function gameStep() {
      if (isPaused || isGameOver) return;

      velocity = { ...nextVelocity };
      const head = { x: snake[0].x + velocity.x, y: snake[0].y + velocity.y };

      // Wall collision
      if (head.x < 0 || head.x >= tileCount || head.y < 0 || head.y >= tileCount) {
        endGame();
        return;
      }

      // Self collision
      if (snake.some(segment => segment.x === head.x && segment.y === head.y)) {
        endGame();
        return;
      }

      snake.unshift(head);

      // Check food collision
      if (head.x === food.x && head.y === food.y) {
        score += 10;
        scoreEl.textContent = score;
        if (score > highScore) {
          highScore = score;
          highEl.textContent = highScore;
          localStorage.setItem('nano_snake_highscore', highScore);
        }
        playSound(580, 0.12, 'triangle');
        spawnFood();
      } else {
        snake.pop();
      }

      draw();
    }

    function draw() {
      // Clear canvas
      ctx.fillStyle = '#020617';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw subtle grid
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 1;
      for (let i = 0; i < canvas.width; i += gridSize) {
        ctx.beginPath();
        ctx.moveTo(i, 0); ctx.lineTo(i, canvas.height);
        ctx.moveTo(0, i); ctx.lineTo(canvas.width, i);
        ctx.stroke();
      }

      // Draw Food (glowing neon magenta circle)
      ctx.shadowBlur = 15;
      ctx.shadowColor = '#f43f5e';
      ctx.fillStyle = '#f43f5e';
      ctx.beginPath();
      ctx.arc(
        food.x * gridSize + gridSize / 2,
        food.y * gridSize + gridSize / 2,
        gridSize / 2 - 2,
        0,
        Math.PI * 2
      );
      ctx.fill();

      // Draw Snake (glowing cyan / green)
      snake.forEach((segment, index) => {
        if (index === 0) {
          ctx.shadowBlur = 15;
          ctx.shadowColor = '#38bdf8';
          ctx.fillStyle = '#38bdf8'; // Head
        } else {
          ctx.shadowBlur = 8;
          ctx.shadowColor = '#4ade80';
          ctx.fillStyle = '#22c55e'; // Body
        }
        ctx.fillRect(
          segment.x * gridSize + 1,
          segment.y * gridSize + 1,
          gridSize - 2,
          gridSize - 2
        );
      });
      ctx.shadowBlur = 0; // Reset shadow

      // Game Over Overlay
      if (isGameOver) {
        ctx.fillStyle = 'rgba(2, 6, 23, 0.85)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = '#f43f5e';
        ctx.font = 'bold 28px "Segoe UI", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('GAME OVER', canvas.width / 2, canvas.height / 2 - 20);

        ctx.fillStyle = '#e2e8f0';
        ctx.font = '16px "Segoe UI", sans-serif';
        ctx.fillText(`Final Score: ${score}`, canvas.width / 2, canvas.height / 2 + 15);
        ctx.fillText('Press Restart to Play Again', canvas.width / 2, canvas.height / 2 + 45);
      }
    }

    function endGame() {
      isGameOver = true;
      clearInterval(gameLoopId);
      playSound(180, 0.35, 'sawtooth');
      draw();
    }

    function togglePause() {
      if (isGameOver) return;
      isPaused = !isPaused;
      pauseBtn.textContent = isPaused ? '▶ Resume' : '⏸ Pause';
    }

    function changeDirection(dx, dy) {
      if (isGameOver || isPaused) return;
      // Prevent reversing onto self
      if (dx !== 0 && velocity.x !== 0) return;
      if (dy !== 0 && velocity.y !== 0) return;
      nextVelocity = { x: dx, y: dy };
    }

    // Keyboard controls
    window.addEventListener('keydown', (e) => {
      switch (e.key) {
        case 'ArrowUp':
        case 'w':
        case 'W':
          e.preventDefault(); changeDirection(0, -1); break;
        case 'ArrowDown':
        case 's':
        case 'S':
          e.preventDefault(); changeDirection(0, 1); break;
        case 'ArrowLeft':
        case 'a':
        case 'A':
          e.preventDefault(); changeDirection(-1, 0); break;
        case 'ArrowRight':
        case 'd':
        case 'D':
          e.preventDefault(); changeDirection(1, 0); break;
        case ' ':
          e.preventDefault(); togglePause(); break;
      }
    });

    // Button controls
    startBtn.addEventListener('click', resetGame);
    pauseBtn.addEventListener('click', togglePause);
    document.getElementById('btnUp').addEventListener('click', () => changeDirection(0, -1));
    document.getElementById('btnDown').addEventListener('click', () => changeDirection(0, 1));
    document.getElementById('btnLeft').addEventListener('click', () => changeDirection(-1, 0));
    document.getElementById('btnRight').addEventListener('click', () => changeDirection(1, 0));

    // Start immediately
    resetGame();
  </script>
</body>
</html>'''


def get_chess_game_html() -> str:
    return '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Cyber Chess - NanoGPT Edition</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; }
    body {
      background: radial-gradient(circle at center, #0f172a 0%, #020617 100%);
      color: #38bdf8;
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      padding: 12px;
    }
    .chess-container {
      background: rgba(15, 23, 42, 0.95);
      border: 2px solid #38bdf8;
      border-radius: 16px;
      box-shadow: 0 0 35px rgba(56, 189, 248, 0.3);
      padding: 16px;
      display: flex;
      flex-direction: column;
      align-items: center;
      max-width: 480px;
      width: 100%;
    }
    .top-bar {
      width: 100%;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      font-weight: 700;
      font-size: 15px;
    }
    .turn-badge {
      padding: 4px 14px;
      border-radius: 20px;
      font-size: 13px;
      font-weight: bold;
      transition: all 0.3s;
    }
    .turn-white {
      background: #f8fafc;
      color: #0f172a;
      box-shadow: 0 0 12px rgba(255, 255, 255, 0.6);
    }
    .turn-black {
      background: #1e293b;
      color: #38bdf8;
      border: 1px solid #38bdf8;
      box-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
    }
    .board-wrapper {
      position: relative;
      border: 2px solid #334155;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    }
    .chess-board {
      display: grid;
      grid-template-columns: repeat(8, 46px);
      grid-template-rows: repeat(8, 46px);
    }
    @media (max-width: 440px) {
      .chess-board {
        grid-template-columns: repeat(8, 38px);
        grid-template-rows: repeat(8, 38px);
      }
    }
    .square {
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 32px;
      cursor: pointer;
      position: relative;
      transition: background 0.15s;
    }
    @media (max-width: 440px) {
      .square { font-size: 26px; }
    }
    .square.light { background: #cbd5e1; color: #0f172a; }
    .square.dark { background: #334155; color: #f8fafc; }
    .square.selected {
      background: #0284c7 !important;
      color: #fff;
    }
    .square.valid-move::after {
      content: '';
      width: 14px;
      height: 14px;
      background: rgba(34, 197, 94, 0.85);
      border-radius: 50%;
      position: absolute;
      box-shadow: 0 0 8px #22c55e;
    }
    .square.valid-capture {
      background: rgba(239, 68, 68, 0.65) !important;
    }
    .status-msg {
      margin-top: 10px;
      font-size: 13px;
      color: #94a3b8;
      min-height: 18px;
      text-align: center;
    }
    .captured-box {
      width: 100%;
      display: flex;
      justify-content: space-between;
      margin-top: 10px;
      font-size: 18px;
      min-height: 28px;
      padding: 4px 8px;
      background: rgba(0,0,0,0.3);
      border-radius: 6px;
    }
    .action-row {
      display: flex;
      gap: 10px;
      margin-top: 12px;
      width: 100%;
    }
    .game-btn {
      flex: 1;
      background: rgba(56, 189, 248, 0.15);
      border: 1px solid #38bdf8;
      color: #38bdf8;
      font-weight: 700;
      font-size: 13px;
      padding: 8px 14px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .game-btn:hover {
      background: #38bdf8;
      color: #020617;
      box-shadow: 0 0 15px #38bdf8;
    }
  </style>
</head>
<body>
  <div class="chess-container">
    <div class="top-bar">
      <span>♟️ CYBER CHESS</span>
      <div id="turnBadge" class="turn-badge turn-white">White's Turn</div>
    </div>

    <div class="board-wrapper">
      <div id="board" class="chess-board"></div>
    </div>

    <div id="statusMsg" class="status-msg">Click a white piece to begin.</div>

    <div class="captured-box">
      <div id="whiteCaptured"></div>
      <div id="blackCaptured"></div>
    </div>

    <div class="action-row">
      <button id="resetBtn" class="game-btn">🔄 New Game</button>
      <button id="undoBtn" class="game-btn">↩ Undo Move</button>
    </div>
  </div>

  <script>
    // Unicode Pieces
    const PIECES = {
      'P': '♙', 'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔',
      'p': '♟', 'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚'
    };

    let boardState = [
      ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
      ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
      ['', '', '', '', '', '', '', ''],
      ['', '', '', '', '', '', '', ''],
      ['', '', '', '', '', '', '', ''],
      ['', '', '', '', '', '', '', ''],
      ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
      ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    ];

    let currentTurn = 'white';
    let selectedSquare = null;
    let validMoves = [];
    let moveHistory = [];
    let whiteCapturedList = [];
    let blackCapturedList = [];

    let audioCtx = null;
    function playMoveSound(isCapture = false) {
      try {
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (audioCtx.state === 'suspended') audioCtx.resume();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.frequency.setValueAtTime(isCapture ? 420 : 320, audioCtx.currentTime);
        osc.type = isCapture ? 'triangle' : 'sine';
        gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + (isCapture ? 0.2 : 0.08));
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + (isCapture ? 0.2 : 0.08));
      } catch (e) {}
    }

    function isWhite(piece) { return piece && piece === piece.toUpperCase(); }
    function isBlack(piece) { return piece && piece === piece.toLowerCase(); }

    function getMoves(row, col) {
      const piece = boardState[row][col];
      if (!piece) return [];
      const moves = [];
      const isW = isWhite(piece);
      const type = piece.toLowerCase();

      function addIfValid(r, c) {
        if (r < 0 || r > 7 || c < 0 || c > 7) return false;
        const target = boardState[r][c];
        if (!target) {
          moves.push({ r, c, capture: false });
          return true;
        }
        if ((isW && isBlack(target)) || (!isW && isWhite(target))) {
          moves.push({ r, c, capture: true });
        }
        return false;
      }

      // Pawn
      if (type === 'p') {
        const dir = isW ? -1 : 1;
        const startRow = isW ? 6 : 1;
        // Forward 1
        if (row + dir >= 0 && row + dir <= 7 && !boardState[row + dir][col]) {
          moves.push({ r: row + dir, c: col, capture: false });
          // Forward 2
          if (row === startRow && !boardState[row + 2 * dir][col]) {
            moves.push({ r: row + 2 * dir, c: col, capture: false });
          }
        }
        // Captures
        for (let dc of [-1, 1]) {
          const nr = row + dir;
          const nc = col + dc;
          if (nr >= 0 && nr <= 7 && nc >= 0 && nc <= 7) {
            const target = boardState[nr][nc];
            if (target && ((isW && isBlack(target)) || (!isW && isWhite(target)))) {
              moves.push({ r: nr, c: nc, capture: true });
            }
          }
        }
      }

      // Knight
      if (type === 'n') {
        const deltas = [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]];
        for (let [dr, dc] of deltas) addIfValid(row + dr, col + dc);
      }

      // Bishop / Queen
      if (type === 'b' || type === 'q') {
        const dirs = [[-1,-1], [-1,1], [1,-1], [1,1]];
        for (let [dr, dc] of dirs) {
          let r = row + dr, c = col + dc;
          while (addIfValid(r, c)) { r += dr; c += dc; }
        }
      }

      // Rook / Queen
      if (type === 'r' || type === 'q') {
        const dirs = [[-1,0], [1,0], [0,-1], [0,1]];
        for (let [dr, dc] of dirs) {
          let r = row + dr, c = col + dc;
          while (addIfValid(r, c)) { r += dr; c += dc; }
        }
      }

      // King
      if (type === 'k') {
        const deltas = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];
        for (let [dr, dc] of deltas) addIfValid(row + dr, col + dc);
      }

      return moves;
    }

    function renderBoard() {
      const boardEl = document.getElementById('board');
      boardEl.innerHTML = '';

      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const sq = document.createElement('div');
          const isLight = (r + c) % 2 === 0;
          sq.className = `square ${isLight ? 'light' : 'dark'}`;
          sq.dataset.row = r;
          sq.dataset.col = c;

          const piece = boardState[r][c];
          if (piece) {
            sq.textContent = PIECES[piece] || '';
          }

          if (selectedSquare && selectedSquare.r === r && selectedSquare.c === c) {
            sq.classList.add('selected');
          }

          const move = validMoves.find(m => m.r === r && m.c === c);
          if (move) {
            if (move.capture) sq.classList.add('valid-capture');
            else sq.classList.add('valid-move');
          }

          sq.addEventListener('click', () => handleSquareClick(r, c));
          boardEl.appendChild(sq);
        }
      }

      const badge = document.getElementById('turnBadge');
      badge.textContent = currentTurn === 'white' ? "White's Turn" : "Black's Turn";
      badge.className = `turn-badge ${currentTurn === 'white' ? 'turn-white' : 'turn-black'}`;

      document.getElementById('whiteCaptured').textContent = whiteCapturedList.map(p => PIECES[p]).join(' ');
      document.getElementById('blackCaptured').textContent = blackCapturedList.map(p => PIECES[p]).join(' ');
    }

    function handleSquareClick(r, c) {
      const piece = boardState[r][c];
      const move = validMoves.find(m => m.r === r && m.c === c);

      if (move && selectedSquare) {
        const captured = boardState[r][c];
        moveHistory.push({
          board: boardState.map(row => [...row]),
          turn: currentTurn,
          whiteCaptured: [...whiteCapturedList],
          blackCaptured: [...blackCapturedList]
        });

        if (captured) {
          if (isWhite(captured)) blackCapturedList.push(captured);
          else whiteCapturedList.push(captured);
        }

        const movingPiece = boardState[selectedSquare.r][selectedSquare.c];
        if (movingPiece === 'P' && r === 0) boardState[r][c] = 'Q';
        else if (movingPiece === 'p' && r === 7) boardState[r][c] = 'q';
        else boardState[r][c] = movingPiece;

        boardState[selectedSquare.r][selectedSquare.c] = '';
        playMoveSound(!!captured);

        currentTurn = currentTurn === 'white' ? 'black' : 'white';
        selectedSquare = null;
        validMoves = [];

        document.getElementById('statusMsg').textContent = `Moved to ${String.fromCharCode(97 + c)}${8 - r}. ${currentTurn === 'white' ? "White's turn" : "Black's turn"}.`;
        renderBoard();
        return;
      }

      if (piece && ((currentTurn === 'white' && isWhite(piece)) || (currentTurn === 'black' && isBlack(piece)))) {
        selectedSquare = { r, c };
        validMoves = getMoves(r, c);
        document.getElementById('statusMsg').textContent = `Selected ${PIECES[piece]} at ${String.fromCharCode(97 + c)}${8 - r}. Choose destination.`;
      } else {
        selectedSquare = null;
        validMoves = [];
      }
      renderBoard();
    }

    document.getElementById('resetBtn').addEventListener('click', () => {
      boardState = [
        ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
        ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
        ['', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', ''],
        ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
        ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
      ];
      currentTurn = 'white';
      selectedSquare = null;
      validMoves = [];
      moveHistory = [];
      whiteCapturedList = [];
      blackCapturedList = [];
      document.getElementById('statusMsg').textContent = "New game started. White's turn.";
      renderBoard();
    });

    document.getElementById('undoBtn').addEventListener('click', () => {
      if (moveHistory.length === 0) return;
      const last = moveHistory.pop();
      boardState = last.board;
      currentTurn = last.turn;
      whiteCapturedList = last.whiteCaptured;
      blackCapturedList = last.blackCaptured;
      selectedSquare = null;
      validMoves = [];
      document.getElementById('statusMsg').textContent = "Move undone.";
      renderBoard();
    });

    renderBoard();
  </script>
</body>
</html>'''

import { refreshTable, setQuery, setLoading } from './data-view.js';
import { tour } from './tour.js';

const $ = (id) => document.getElementById(id);
const DIRECTIONS = [[0, -1], [1, 0], [0, 1], [-1, 0]];
const MARGIN = 26;
const LETTERS = 'ABCDEFGHIJKLMNOP';
const CRASH = '#ff6eb4';
let current, pending = false, connected = false;
let answerSeconds = null; // TabPFN's answer time for the prediction on the board

// The server records a frame for every board change. The browser fetches them in
// batches and shows them one by one, holding on predictions so they can be read.
let lastSeq = null;
const queue = [];
// What is on the board right now.
const view = {state: null, prev: null, slideStart: 0, pending: false, q: null, action: null, revealAt: 0};

async function api(path, body) {
  const response = await fetch(`/api/${path}`, body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  });
  const value = await response.json();
  if (!response.ok) throw new Error(typeof value.detail === 'string' ? value.detail : JSON.stringify(value.detail));
  return value;
}

function showError(message) {
  $('error').hidden = !message;
  $('error').textContent = message || '';
}

// Speed is the point of the demo: how long learning took, and how long each answer takes.
function renderSpeed() {
  const learned = current?.fitted && current.learn_seconds != null;
  const answer = answerSeconds ?? current?.predict_seconds;
  $('speed').hidden = !learned;
  if (!learned) return;
  const text = `Learned from <strong>${current.learn_rows.toLocaleString()}</strong> moves in `
    + `<strong>${current.learn_seconds.toFixed(1)} s</strong>`
    + (answer != null ? ` · answered in <strong>${answer.toFixed(2)} s</strong>` : '');
  if ($('speed').innerHTML !== text) $('speed').innerHTML = text;
}

function render() {
  if (!current) return;
  renderSpeed();
  const busy = pending || !!current.job || !connected;
  $('clear').disabled = busy;
  $('play').disabled = pending || !connected || (!current.job && !current.has_token);
  $('play').textContent = current.job ? 'Stop' : current.state.done ? 'Play again' : current.state.steps ? 'Continue' : 'Watch TabPFN play';
  $('memory').textContent = `${current.rows.toLocaleString()} saved moves`;
  $('setup').hidden = current.has_token;
  $('mode-badge').hidden = current.model_mode !== 'stub';
  $('game-status').textContent = current.message;
  $('game-status').title = current.message;
  if (!current.rows && !current.query) setQuery(null);
  if (current.error) showError(current.error);
  renderOverlay();
  tour.update(current);
  setLoading(current.phase === 'training');
}

// Slim strip over the final board while TabPFN learns from completed games.
function renderOverlay() {
  const refit = current.job === 'playing' && current.phase === 'training';
  $('overlay').hidden = !refit;
  $('overlay-title').textContent = refit ? 'Learning from that game…' : '';
  $('overlay-progress').textContent = refit ? current.message : '';
}

// Frames for one TabPFN move arrive as: query (rows sent, answer pending) → prediction
// (values known, snake still) → move (snake moved). Random steps only send move/reset.
function show(frame, now) {
  switch (frame.kind) {
    case 'reset':
      Object.assign(view, {state: frame.state, prev: null, pending: false, q: null});
      setQuery(null);
      break;
    case 'query':
      Object.assign(view, {state: frame.state, pending: true, q: null});
      setQuery(frame.query);
      break;
    case 'prediction':
      answerSeconds = frame.seconds;
      renderSpeed();
      Object.assign(view, {state: frame.state, pending: false, q: frame.q_values, action: frame.action, revealAt: now});
      setQuery(frame.query, frame.q_values, frame.action);
      break;
    default: // move
      Object.assign(view, {prev: view.state, state: frame.state, slideStart: now, pending: false, q: null});
      if (!frame.q_values) setQuery(null);
  }
  $('score').textContent = view.state.score;
  $('moves').textContent = view.state.steps;
}

// The square each turn leads to, from the current head and heading.
function candidates(state) {
  const [hx, hy] = state.snake[0];
  return [-1, 0, 1].map(turn => {
    const [dx, dy] = DIRECTIONS[(state.direction + turn + 4) % 4];
    return [hx + dx, hy + dy];
  });
}

const ease = t => t < 0 ? 0 : t > 1 ? 1 : 1 - Math.pow(1 - t, 3);

function paint(now) {
  const state = view.state;
  if (!state) return;
  const ctx = $('board').getContext('2d'), size = 640 - MARGIN, cell = size / state.size;
  ctx.clearRect(0, 0, 640, 640);
  ctx.fillStyle = '#6f7d72'; ctx.font = '500 16px system-ui, sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  for (let i = 0; i < state.size; i++) {
    ctx.fillText(LETTERS[i], MARGIN + (i + .5) * cell, MARGIN / 2);
    ctx.fillText(String(i + 1), MARGIN / 2, MARGIN + (i + .5) * cell);
  }
  ctx.save(); ctx.translate(MARGIN, MARGIN);
  ctx.fillStyle = '#173f31'; ctx.beginPath(); ctx.roundRect(0, 0, size, size, 10); ctx.fill();
  ctx.strokeStyle = '#24503e'; ctx.lineWidth = 1;
  for (let i = 1; i < state.size; i++) {
    ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, size); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * cell); ctx.lineTo(size, i * cell); ctx.stroke();
  }

  // TabPFN's view of the next move, drawn over the snake: an arrow from the head into each
  // square a turn leads to. Opacity follows the relative preference, so three near-equal
  // values read as indifference rather than a confident pick. Crash turns get a red tick.
  function drawArrow(hx, hy, dx, dy, style, width) {
    const sx = (hx + .5 + dx * .30) * cell, sy = (hy + .5 + dy * .30) * cell;
    const ex = (hx + .5 + dx * .84) * cell, ey = (hy + .5 + dy * .84) * cell;
    const head = cell * .15;
    ctx.strokeStyle = style; ctx.lineWidth = width; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(ex - (dx + dy) * head, ey - (dy - dx) * head);
    ctx.lineTo(ex, ey);
    ctx.lineTo(ex - (dx - dy) * head, ey - (dy + dx) * head);
    ctx.stroke();
  }
  // Pink with a dark outline, so the mark stands apart from the green snake and board
  // for red–green colour blindness as well; the shape (bar or cross) says the rest.
  function drawCrashTick(hx, hy, x, y, wall, chosen) {
    const dx = x - hx, dy = y - hy;
    ctx.beginPath();
    if (wall) { // A bar on the head square's edge that faces the wall.
      const cx = (hx + .5 + dx * .46) * cell, cy = (hy + .5 + dy * .46) * cell, half = cell * .2;
      ctx.moveTo(cx - dy * half, cy - dx * half); ctx.lineTo(cx + dy * half, cy + dx * half);
    } else { // A cross on the body segment.
      const cx = (x + .5) * cell, cy = (y + .5) * cell, r = cell * .11;
      ctx.moveTo(cx - r, cy - r); ctx.lineTo(cx + r, cy + r);
      ctx.moveTo(cx + r, cy - r); ctx.lineTo(cx - r, cy + r);
    }
    const width = cell * (chosen ? .06 : .035);
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#102e24'; ctx.lineWidth = width + 5; ctx.stroke();
    ctx.strokeStyle = chosen ? CRASH : 'rgba(255, 110, 180, .75)'; ctx.lineWidth = width; ctx.stroke();
  }
  // Returns true when the chosen turn is a crash, so the board can flag it.
  function drawPredictions() {
    if (!view.pending && !view.q) return false;
    const [hx, hy] = state.snake[0], q = view.q;
    const lo = q && Math.min(...q), hi = q && Math.max(...q);
    const decided = q && hi - lo >= 0.05;
    let crashChosen = false;
    candidates(state).forEach(([x, y], i) => {
      const offBoard = x < 0 || y < 0 || x >= state.size || y >= state.size;
      const eating = state.food && x === state.food[0] && y === state.food[1];
      const occupied = eating ? state.snake : state.snake.slice(0, -1);
      const onBody = occupied.some(([sx, sy]) => sx === x && sy === y);
      const chosen = !!q && i === view.action;
      if (offBoard || onBody) {
        drawCrashTick(hx, hy, x, y, offBoard, chosen);
        crashChosen = crashChosen || chosen;
        return;
      }
      let style, width;
      if (!q) {
        style = `rgba(255, 255, 255, ${.28 + .12 * Math.sin(now / 220)})`; width = cell * .06;
      } else if (chosen) {
        style = 'rgba(244, 255, 230, .95)'; width = cell * .09;
      } else {
        const preference = decided ? (q[i] - lo) / (hi - lo) : .5;
        style = `rgba(166, 227, 161, ${.15 + .45 * preference})`; width = cell * .06;
      }
      drawArrow(hx, hy, x - hx, y - hy, style, width);
    });
    return crashChosen;
  }

  // The head glows while TabPFN evaluates the board.
  if (state.food) {
    const [x, y] = state.food, cx = (x + .5) * cell, cy = (y + .5) * cell;
    ctx.fillStyle = '#edb16d'; ctx.beginPath();
    ctx.arc(cx, cy, cell * .20, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#ffdbab'; ctx.lineWidth = 3; ctx.beginPath();
    ctx.moveTo((x + .52) * cell, (y + .29) * cell); ctx.lineTo((x + .6) * cell, (y + .2) * cell); ctx.stroke();
  }
  if (view.pending && !state.done) {
    const [hx, hy] = state.snake[0];
    ctx.fillStyle = `rgba(208, 238, 160, ${.16 + .1 * Math.sin(now / 160)})`;
    ctx.beginPath(); ctx.roundRect(hx * cell - 3, hy * cell - 3, cell + 6, cell + 6, 16); ctx.fill();
  }
  // Slide the body forward over 260 ms after a move.
  let segments = state.snake;
  const t = ease((now - view.slideStart) / 260);
  if (view.prev && t < 1) {
    const prev = view.prev.snake;
    segments = state.snake.map(([x, y], i) => {
      const [px, py] = prev[Math.min(i, prev.length - 1)];
      return [px + (x - px) * t, py + (y - py) * t];
    });
  }
  [...segments].reverse().forEach(([x, y], index) => {
    const head = index === segments.length - 1;
    ctx.fillStyle = head ? '#d0eea0' : '#7fb775'; ctx.beginPath();
    ctx.roundRect(x * cell + 5, y * cell + 5, cell - 10, cell - 10, head ? 13 : 8); ctx.fill();
    if (head) {
      const [dx, dy] = DIRECTIONS[state.direction];
      for (const side of [-1, 1]) {
        ctx.fillStyle = '#173f31'; ctx.beginPath();
        ctx.arc((x + .5 + dx * .19 + dy * side * .14) * cell,
          (y + .5 + dy * .19 - dx * side * .14) * cell, 4, 0, Math.PI * 2); ctx.fill();
      }
    }
  });

  if (drawPredictions()) { // TabPFN picked a crash: flash the board edge.
    ctx.globalAlpha = .55 + .35 * Math.sin((now - view.revealAt) / 90);
    ctx.strokeStyle = CRASH; ctx.lineWidth = 8;
    ctx.beginPath(); ctx.roundRect(4, 4, size - 8, size - 8, 10); ctx.stroke();
    ctx.globalAlpha = 1;
  }

  if (state.done && t >= 1) {
    const mid = size / 2;
    ctx.fillStyle = '#102e24cc'; ctx.fillRect(0, mid - 70, size, 140);
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = '#f1f5e4'; ctx.font = 'bold 30px sans-serif';
    const [title, detail] = state.reason === 'board filled' ? ['Board complete!', `${state.score} apples`]
      : state.reason === 'starvation' ? ['The snake starved', `${state.score} apples · ${state.steps} moves`]
      : ['The snake crashed', `${state.score} apples · ${state.steps} moves`];
    ctx.fillText(title, mid, mid - 13);
    ctx.fillStyle = '#f1f5e4'; ctx.font = '19px sans-serif'; ctx.fillText(detail, mid, mid + 22);
  }
  ctx.restore();
}

async function refreshFrames() {
  if (lastSeq === null || current.seq < lastSeq) { // First load or server restart.
    lastSeq = current.seq;
    queue.length = 0;
    show({kind: 'reset', state: current.state}, performance.now());
    if (current.decision_query && current.q_values) {
      if (current.query) {
        show({kind: 'prediction', state: current.decision_state, query: current.decision_query,
          q_values: current.q_values, action: current.last_action}, performance.now());
      } else {
        setQuery(current.decision_query, current.q_values, current.last_action);
      }
    } else if (current.query && current.phase === 'predicting') { // Joined mid-prediction.
      show({kind: 'query', state: current.state, query: current.query}, performance.now());
    }
    return;
  }
  if (current.seq === lastSeq) return;
  const data = await api(`frames?after=${lastSeq}`);
  queue.push(...data.frames);
  lastSeq = data.latest;
  // Fast random play outruns the screen; once the job is over, jump to the final board
  // instead of replaying the backlog, and never let the backlog grow past a second or so.
  if (!current.job && queue.length > 1 && !queue.some(f => f.kind === 'prediction')) queue.splice(0, queue.length - 1);
  else if (queue.length > 90) queue.splice(0, queue.length - 60);
}

// Show queued frames, skipping ahead when hundreds are waiting (random steps), and
// holding on TabPFN's query and prediction so they can be read before the snake moves.
let lastShownAt = 0, holdUntil = 0;
function tick(now) {
  if (queue.length && now >= holdUntil && (queue.length > 5 || now - lastShownAt >= 40)) {
    let frame;
    for (let i = Math.max(1, Math.ceil(queue.length / 30)); i > 0 && queue.length; i--) frame = queue.shift();
    show(frame, now);
    lastShownAt = now;
    if (frame.kind === 'query') holdUntil = now + 200;
    if (frame.kind === 'prediction') holdUntil = now + 450;
  }
  if (view.pending && !queue.length && current && !current.job && !current.query) {
    view.pending = false;
    setQuery(null); // A stopped or failed request has no prediction to show.
  }
  paint(now);
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

async function refresh() {
  current = await api('state');
  connected = true;
  render();
  await refreshFrames();
  await refreshTable(['random-steps', 'preparing', 'playing'].includes(current.job));
}

async function command(path, body = {}) {
  pending = true;
  render();
  showError(null);
  try { await api(path, body); await refresh(); }
  catch (error) { showError(error.message); }
  finally { pending = false; render(); }
}

$('clear').onclick = async () => {
  if (current.rows) {
    if (!confirm(`Clear ${current.rows.toLocaleString()} saved moves? A backup will be saved.`)) return;
    await command('clear');
    if (current.error) return;
  }
  tour.begin('practice'); // Refill the table with random moves, then fit and play.
};
$('play').onclick = () => {
  if (current.job) return command('pause');
  if (current.fitted) return tour.play();
  tour.begin('practice'); // Prepare first; the tour card shows progress and the rules.
};
tour.init({api, refresh, command, onChange: render});

let resumed = false;
async function poll() {
  try {
    await refresh();
    if (!resumed) { resumed = true; tour.resume(current); }
  } catch (error) {
    connected = false;
    render();
    showError(`The server isn’t responding. ${error.message}`);
  }
  setTimeout(poll, 300);
}
poll();

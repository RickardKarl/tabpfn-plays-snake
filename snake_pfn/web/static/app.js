import { refreshTable, setQuery, setLoading } from './data-view.js';
import { refreshLog } from './log-view.js';
import { tour } from './tour.js';

const $ = (id) => document.getElementById(id);
const DIRECTIONS = [[0, -1], [1, 0], [0, 1], [-1, 0]];
const MARGIN = 26;
const LETTERS = 'ABCDEFGHIJKLMNOP';
let current, presets, pending = false, connected = false;

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

// The player can step the apple while any game is running; the hint and the evaded
// counter belong to a game against TabPFN only.
const inGame = () => !!current && ['waiting', 'predicting', 'random'].includes(current.phase);
const versusGame = () => inGame() && current.job === 'playing';

// When the current phase began, for the turn indicator's countdown.
let phase = null, phaseSince = 0;

function render() {
  if (!current) return;
  if (current.phase !== phase) { phase = current.phase; phaseSince = performance.now(); }
  const busy = pending || !!current.job || !connected;
  $('inputs').disabled = busy;
  $('clear').disabled = busy || !current.rows;
  $('play').disabled = pending || !connected || (!current.job && !current.has_token);
  $('play').textContent = current.job ? 'Stop' : current.outcome ? 'Play again' : 'Play';
  $('memory').textContent = `${current.rows.toLocaleString()} saved moves`;
  $('setup').hidden = current.has_token;
  $('mode-badge').hidden = current.model_mode !== 'stub';
  $('apple-hint').hidden = !versusGame();
  $('evaded').hidden = !versusGame();
  $('board').classList.toggle('playable', inGame());
  $('limit').textContent = current.hunger_limit;
  document.querySelectorAll('.limit').forEach(el => el.textContent = current.hunger_limit);
  const preset = Object.keys(presets).find(name => {
    const groups = presets[name];
    return !current.features.exclude.length && groups.length === current.features.groups.length
      && groups.every(group => current.features.groups.includes(group));
  });
  $('custom-inputs').hidden = !!preset;
  $('inputs').value = preset || 'custom';
  if (current.error) showError(current.error);
  renderOverlay();
  tour.update(current);
  setLoading(current.phase === 'training');
}

// Slim strip over the game-over board while TabPFN refits between games.
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
    case 'apple': // The player moved the apple; same board otherwise.
      Object.assign(view, {state: frame.state, prev: null});
      break;
    case 'query':
      Object.assign(view, {state: frame.state, pending: true, q: null});
      setQuery(frame.query);
      break;
    case 'prediction':
      Object.assign(view, {pending: false, q: frame.q_values, action: frame.action, revealAt: now});
      setQuery(undefined, frame.q_values, frame.action);
      break;
    default: // move
      Object.assign(view, {prev: view.state, state: frame.state, slideStart: now, pending: false, q: null});
      setQuery(null);
  }
  $('hungry').textContent = view.state.hungry;
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
  function drawCrashTick(hx, hy, x, y, wall, chosen) {
    const dx = x - hx, dy = y - hy;
    ctx.strokeStyle = chosen ? '#ff6b5e' : 'rgba(224, 101, 91, .7)';
    ctx.lineWidth = cell * (chosen ? .06 : .035); ctx.lineCap = 'round';
    ctx.beginPath();
    if (wall) { // A bar on the head square's edge that faces the wall.
      const cx = (hx + .5 + dx * .46) * cell, cy = (hy + .5 + dy * .46) * cell, half = cell * .2;
      ctx.moveTo(cx - dy * half, cy - dx * half); ctx.lineTo(cx + dy * half, cy + dx * half);
    } else { // A cross on the body segment.
      const cx = (x + .5) * cell, cy = (y + .5) * cell, r = cell * .11;
      ctx.moveTo(cx - r, cy - r); ctx.lineTo(cx + r, cy + r);
      ctx.moveTo(cx + r, cy - r); ctx.lineTo(cx - r, cy + r);
    }
    ctx.stroke();
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

  // Whose turn: during the player's turn the apple wears a ring that shrinks as the turn
  // window runs out; while TabPFN thinks, the snake's head glows instead.
  const versus = current && current.job === 'playing';
  const applesTurn = versus && current.phase === 'waiting' && state.food;
  if (state.food) {
    const [x, y] = state.food, cx = (x + .5) * cell, cy = (y + .5) * cell;
    if (applesTurn) {
      const left = Math.max(0, 1 - (now - phaseSince) / (current.turn_window * 1000));
      ctx.fillStyle = `rgba(237, 177, 109, ${.12 + .08 * Math.sin(now / 160)})`;
      ctx.beginPath(); ctx.arc(cx, cy, cell * .42, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#ffdbab'; ctx.lineWidth = 4; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.arc(cx, cy, cell * .36, -Math.PI / 2, -Math.PI / 2 + left * Math.PI * 2); ctx.stroke();
    }
    ctx.fillStyle = '#edb16d'; ctx.beginPath();
    ctx.arc(cx, cy, cell * .20, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#ffdbab'; ctx.lineWidth = 3; ctx.beginPath();
    ctx.moveTo((x + .52) * cell, (y + .29) * cell); ctx.lineTo((x + .6) * cell, (y + .2) * cell); ctx.stroke();
  }
  if (versus && current.phase === 'predicting' && !state.done) {
    const [hx, hy] = state.snake[0];
    ctx.fillStyle = `rgba(208, 238, 160, ${.16 + .1 * Math.sin(now / 160)})`;
    ctx.beginPath(); ctx.roundRect(hx * cell - 3, hy * cell - 3, cell + 6, cell + 6, 16); ctx.fill();
  }
  // The player's requested apple step, shown as a ghost until the snake's next move.
  const pendingFood = inGame() && current.pending_food;
  if (pendingFood && (!state.food || pendingFood[0] !== state.food[0] || pendingFood[1] !== state.food[1])) {
    const [x, y] = pendingFood;
    ctx.setLineDash([4, 4]); ctx.strokeStyle = '#edb16d'; ctx.lineWidth = 2; ctx.beginPath();
    ctx.arc((x + .5) * cell, (y + .5) * cell, cell * .20, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
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
    ctx.strokeStyle = '#ff6b5e'; ctx.lineWidth = 8;
    ctx.beginPath(); ctx.roundRect(4, 4, size - 8, size - 8, 10); ctx.stroke();
    ctx.globalAlpha = 1;
  }

  const outcome = current && !current.job ? current.outcome : null;
  if ((state.done || outcome) && t >= 1) {
    const mid = size / 2;
    ctx.fillStyle = '#102e24cc'; ctx.fillRect(0, mid - 70, size, 140);
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = outcome === 'eaten' ? '#ffb3a8' : '#f1f5e4'; ctx.font = 'bold 30px sans-serif';
    const moves = `${state.steps} move${state.steps === 1 ? '' : 's'}`;
    const [title, detail] = outcome === 'eaten' ? ['Game over', `TabPFN ate you after ${moves}`]
      : outcome === 'starved' ? ['You win!', `The snake starved after ${moves}`]
      : outcome === 'crashed' ? ['You win!', `The snake crashed after ${moves}`]
      : state.reason === 'board filled' ? ['Board complete!', `${state.score} apples`]
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
    if (current.query && current.phase === 'predicting') { // Joined mid-prediction.
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
  if (!current.job && queue.length > 1) queue.splice(0, queue.length - 1);
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
  await refreshLog();
}

async function command(path, body = {}) {
  pending = true;
  render();
  showError(null);
  try { await api(path, body); await refresh(); }
  catch (error) { showError(error.message); }
  finally { pending = false; render(); }
}

$('clear').onclick = () => {
  if (!confirm(`Clear all ${current.rows.toLocaleString()} saved moves? The file is archived, not deleted.`)) return;
  command('clear').then(() => { if (!current.error) tour.begin('hook'); }); // A fresh table restarts the intro.
};
$('play').onclick = () => {
  if (current.job) return command('pause');
  if (current.fitted) return tour.play();
  tour.begin('practice'); // Prepare first; the tour card shows progress and the rules.
};
$('replay-intro').onclick = () => { if (!current.job) tour.begin('hook'); };
tour.init({api, refresh, command, onChange: render});
$('inputs').onchange = () => command('features', {groups: presets[$('inputs').value], exclude: []});

// Play as the apple: one step per snake move, by arrow key or by clicking a neighbour.
// Steps that are impossible (off the board, onto the snake, not a neighbour) are ignored
// quietly; the game already shows where the apple can go.
async function moveApple(x, y) {
  if (!inGame() || !current.state.food) return;
  const {size, snake, food} = current.state;
  const [fx, fy] = food;
  const neighbour = Math.abs(x - fx) + Math.abs(y - fy) === 1, stay = x === fx && y === fy;
  const onBoard = x >= 0 && y >= 0 && x < size && y < size;
  if (!stay && (!neighbour || !onBoard || snake.some(([sx, sy]) => sx === x && sy === y))) return;
  try { current = await api('apple', {x, y}); render(); }
  catch (error) { console.warn(error.message); }
}
document.addEventListener('keydown', event => {
  const delta = {ArrowUp: [0, -1], ArrowRight: [1, 0], ArrowDown: [0, 1], ArrowLeft: [-1, 0]}[event.key];
  if (!delta || !inGame() || !current.state.food) return;
  event.preventDefault();
  const from = current.state.food; // Each press re-picks the step from where the apple is.
  moveApple(from[0] + delta[0], from[1] + delta[1]);
});
$('board').addEventListener('click', event => {
  if (!inGame()) return;
  const rect = $('board').getBoundingClientRect(), scale = 640 / rect.width;
  const size = current.state.size, cell = (640 - MARGIN) / size;
  const x = Math.floor(((event.clientX - rect.left) * scale - MARGIN) / cell);
  const y = Math.floor(((event.clientY - rect.top) * scale - MARGIN) / cell);
  if (x >= 0 && y >= 0 && x < size && y < size) moveApple(x, y);
});

let resumed = false;
async function poll() {
  try {
    if (!presets) presets = (await api('features')).presets;
    await refresh();
    if (!resumed) { resumed = true; tour.resume(current); }
  } catch (error) {
    connected = false;
    render();
    showError(`Cannot connect to the server. ${error.message}`);
  }
  setTimeout(poll, 300);
}
poll();

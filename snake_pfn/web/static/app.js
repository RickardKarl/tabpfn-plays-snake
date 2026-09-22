import { refreshTable, setQuery, setLoading } from './data-view.js';
import { refreshLog } from './log-view.js';

const $ = (id) => document.getElementById(id);
const ACTIONS = ['left', 'straight', 'right'];
const DIRECTIONS = [[0, -1], [1, 0], [0, 1], [-1, 0]];
const MARGIN = 26;
const LETTERS = 'ABCDEFGHIJKLMNOP';
const RANK_COLOURS = ['#a6e3a1', '#e6c76e', '#e0906d']; // best, middle, worst
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

function statusText() {
  // While TabPFN works, the status card already shows the message.
  if (current.job) return ['training', 'predicting', 'waiting'].includes(current.phase) ? '' : current.message;
  if (current.error) return 'Stopped.';
  if (!current.random_runs) return 'Play random steps first. TabPFN learns from those moves.';
  if (current.fitted) return 'Model ready. TabPFN plays again with the same model.';
  if (current.rows >= 10) return current.has_token
    ? 'Moves saved. Playing with TabPFN first loads them as its table, then plays one move per prediction.'
    : 'Moves saved.';
  return 'Play random steps to get started.';
}

function render() {
  if (!current) return;
  const busy = pending || !!current.job || !connected;
  $('inputs').disabled = $('random-steps').disabled = busy;
  $('clear').disabled = busy || !current.rows;
  const canPlay = current.random_runs > 0 && (current.fitted || (current.has_token && current.rows >= 10));
  $('play').disabled = pending || !connected || (!current.job && !canPlay);
  $('play').textContent = current.job ? 'Stop' : 'Play game with TabPFN';
  $('memory').textContent = `${current.rows.toLocaleString()} saved moves`;
  $('setup').hidden = current.has_token;
  $('mode-badge').hidden = current.model_mode !== 'stub';
  const playing = current.job === 'playing';
  $('apple-hint').hidden = !playing;
  $('evaded').hidden = !playing;
  $('board').classList.toggle('playable', playing);
  const limit = 2 * current.state.size * current.state.size;
  $('limit').textContent = limit;
  document.querySelectorAll('.apple-hint .limit').forEach(el => el.textContent = limit);
  const preset = Object.keys(presets).find(name => {
    const groups = presets[name];
    return !current.features.exclude.length && groups.length === current.features.groups.length
      && groups.every(group => current.features.groups.includes(group));
  });
  $('custom-inputs').hidden = !!preset;
  $('inputs').value = preset || 'custom';
  $('status').textContent = statusText();
  if (current.error) showError(current.error);
  renderStatus();
  setLoading(current.phase === 'training', current.message);
}

// One status card for TabPFN: sleeping, loading the table as context, or predicting.
const STATES = {
  idle: ['Sleeping', c => c.fitted
    ? 'Model ready. The next game with TabPFN reuses its loaded table.'
    : c.rows >= 10 ? 'No model yet. The first game with TabPFN loads the saved moves first.'
    : 'Waiting for saved moves.'],
  random: ['Sleeping', () => 'Random steps do not use TabPFN.'],
  waiting: ['Your move', c => c.pending_food
    ? `Apple will step to ${'ABCDEFGHIJKLMNOP'[c.pending_food[0]]}${c.pending_food[1] + 1}. Press another arrow to change, or click the apple to stay.`
    : 'Arrow keys step the apple one square. TabPFN is asked once your turn ends.'],
  training: ['Loading table', c => c.message],
  predicting: ['Predicting moves', c => c.message],
};
const STUB_STATES = {
  ...STATES,
  training: ['Loading table (stub)', c => c.message],
  predicting: ['Predicting moves (stub)', c => c.message],
};
let phaseKey = null, phaseSince = 0;
function renderStatus() {
  const states = current.model_mode === 'stub' ? STUB_STATES : STATES;
  const [state, detail] = states[current.phase] || states.idle;
  $('tabpfn-status').dataset.state = current.phase in STATES ? current.phase : 'idle';
  $('tabpfn-state').textContent = state;
  $('tabpfn-detail').textContent = detail(current);
  const key = `${current.phase}:${current.message}`;
  if (key !== phaseKey) { phaseKey = key; phaseSince = Date.now(); }
  if (!['training', 'predicting', 'waiting'].includes(current.phase)) $('tabpfn-elapsed').textContent = '';
}
setInterval(() => {
  if (!current || !['training', 'predicting', 'waiting'].includes(current.phase)) return;
  $('tabpfn-elapsed').textContent = `${((Date.now() - phaseSince) / 1000).toFixed(1)} s`;
}, 100);

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
  $('score').textContent = view.state.score;
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

  // TabPFN's view of the next move. Free squares get a tile drawn under the snake so the
  // head slides onto the winner. Crash moves are marked after the snake is drawn: a red
  // band on the head's edge for a wall, a red dashed outline over the body segment.
  const showQ = view.pending || view.q;
  const order = view.q ? [0, 1, 2].sort((a, b) => view.q[b] - view.q[a]) : null;
  const crashes = [];
  if (showQ) {
    candidates(state).forEach(([x, y], i) => {
      const offBoard = x < 0 || y < 0 || x >= state.size || y >= state.size;
      const eating = state.food && x === state.food[0] && y === state.food[1];
      const occupied = eating ? state.snake : state.snake.slice(0, -1);
      const onBody = occupied.some(([sx, sy]) => sx === x && sy === y);
      if (offBoard || onBody) { crashes.push({x, y, i, kind: offBoard ? 'wall' : 'body'}); return; }
      ctx.beginPath(); ctx.roundRect(x * cell + 4, y * cell + 4, cell - 8, cell - 8, 11);
      if (view.pending) {
        ctx.fillStyle = `rgba(255, 255, 255, ${0.10 + 0.07 * Math.sin(now / 220)})`; ctx.fill();
        ctx.setLineDash([6, 6]); ctx.strokeStyle = 'rgba(255, 255, 255, .4)'; ctx.lineWidth = 2; ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(255, 255, 255, .5)'; ctx.font = `600 ${cell * .32}px system-ui, sans-serif`;
        ctx.fillText('?', (x + .5) * cell, (y + .5) * cell);
      } else {
        const rank = order.indexOf(i), q = view.q[i];
        ctx.globalAlpha = rank === 0 ? .95 : .72;
        ctx.fillStyle = RANK_COLOURS[rank]; ctx.fill();
        ctx.globalAlpha = 1;
        if (i === view.action) {
          ctx.lineWidth = 3 + 1.5 * Math.sin((now - view.revealAt) / 140);
          ctx.strokeStyle = '#f4ffe6'; ctx.stroke();
        }
        ctx.fillStyle = '#173f31';
        ctx.font = `700 ${cell * .27}px system-ui, sans-serif`;
        ctx.fillText((q >= 0 ? '+' : '') + q.toFixed(2), (x + .5) * cell, (y + .46) * cell);
        ctx.font = `500 ${cell * .13}px system-ui, sans-serif`;
        ctx.fillText(ACTIONS[i], (x + .5) * cell, (y + .74) * cell);
      }
    });
  }

  // Draw one crash candidate (wall or body) with its value, on top of everything else.
  function drawCrash({x, y, i, kind}) {
    const CRASH = '#e0655b';
    const [hx, hy] = state.snake[0];
    const label = view.pending ? '?' : (view.q[i] >= 0 ? '+' : '') + view.q[i].toFixed(2);
    const chosen = view.q && i === view.action;
    ctx.setLineDash([5, 4]); ctx.lineWidth = chosen ? 4 : 2.5; ctx.strokeStyle = CRASH;
    let px, py; // where the value pill goes
    if (kind === 'wall') {
      // A band along the head square's edge that faces the wall.
      const dx = x - hx, dy = y - hy, t = cell * .16;
      const bx = dx > 0 ? (hx + 1) * cell - t : hx * cell, by = dy > 0 ? (hy + 1) * cell - t : hy * cell;
      const w = dx === 0 ? cell : t, h = dy === 0 ? cell : t;
      ctx.globalAlpha = .85; ctx.fillStyle = CRASH; ctx.fillRect(bx, by, w, h); ctx.globalAlpha = 1;
      ctx.strokeRect(bx + 1, by + 1, w - 2, h - 2);
      px = (hx + .5 + dx * .28) * cell; py = (hy + .5 + dy * .28) * cell;
    } else {
      ctx.beginPath(); ctx.roundRect(x * cell + 4, y * cell + 4, cell - 8, cell - 8, 11); ctx.stroke();
      px = (x + .5) * cell; py = (y + .5) * cell;
    }
    ctx.setLineDash([]);
    // Value pill with the crash kind under it.
    ctx.font = `700 ${cell * .18}px system-ui, sans-serif`;
    const w = ctx.measureText(label).width + cell * .16, h = cell * .30;
    ctx.fillStyle = CRASH; ctx.beginPath(); ctx.roundRect(px - w / 2, py - h / 2, w, h, h / 2); ctx.fill();
    ctx.fillStyle = '#fff5f2'; ctx.fillText(label, px, py - cell * .02);
    ctx.font = `600 ${cell * .11}px system-ui, sans-serif`;
    ctx.fillText(`${ACTIONS[i]} · ${kind}`, px, py + cell * .22);
  }

  if (state.food) {
    const [x, y] = state.food; ctx.fillStyle = '#edb16d'; ctx.beginPath();
    ctx.arc((x + .5) * cell, (y + .5) * cell, cell * .20, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#ffdbab'; ctx.lineWidth = 3; ctx.beginPath();
    ctx.moveTo((x + .52) * cell, (y + .29) * cell); ctx.lineTo((x + .6) * cell, (y + .2) * cell); ctx.stroke();
  }
  // The player's requested apple step, shown as a ghost until the snake's next move.
  const pendingFood = current && current.job === 'playing' && current.pending_food;
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

  crashes.forEach(drawCrash);

  if (state.done && t >= 1) {
    const mid = size / 2;
    ctx.fillStyle = '#102e24bb'; ctx.fillRect(0, mid - 70, size, 140);
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = '#f1f5e4'; ctx.font = 'bold 30px sans-serif';
    const title = state.reason === 'board filled' ? 'Board complete!'
      : state.reason === 'starvation' ? (current?.apple_moves ? 'You win! The snake starved' : 'The snake starved')
      : state.reason === 'collision' ? 'The snake crashed' : 'Episode complete';
    ctx.fillText(title, mid, mid - 13);
    ctx.font = '19px sans-serif'; ctx.fillText(`${state.score} apples · ${state.reason}`, mid, mid + 22);
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
    if (frame.kind === 'query') holdUntil = now + 400;
    if (frame.kind === 'prediction') holdUntil = now + 1300;
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
  await refreshTable(current.job === 'random-steps' || current.job === 'playing');
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

$('random-steps').onclick = () => command('random-steps', {moves: 1000});
$('clear').onclick = () => {
  if (confirm(`Clear all ${current.rows.toLocaleString()} saved moves? The file is archived, not deleted.`)) command('clear');
};
$('play').onclick = () => current.job ? command('pause') : command('play', {
  moves: 100, policy: 'tabpfn', auto_fit: true, rounds: 3
});
$('inputs').onchange = () => command('features', {groups: presets[$('inputs').value], exclude: []});

// Play as the apple: one step per snake move, by arrow key or by clicking a neighbour.
async function moveApple(x, y) {
  if (!current || current.job !== 'playing' || !current.state.food) return;
  try { current = await api('apple', {x, y}); render(); }
  catch (error) { $('status').textContent = error.message; }
}
document.addEventListener('keydown', event => {
  const delta = {ArrowUp: [0, -1], ArrowRight: [1, 0], ArrowDown: [0, 1], ArrowLeft: [-1, 0]}[event.key];
  if (!delta || !current || current.job !== 'playing' || !current.state.food) return;
  event.preventDefault();
  const from = current.state.food; // Each press re-picks the step from where the apple is.
  moveApple(from[0] + delta[0], from[1] + delta[1]);
});
$('board').addEventListener('click', event => {
  if (!current || current.job !== 'playing') return;
  const rect = $('board').getBoundingClientRect(), scale = 640 / rect.width;
  const size = current.state.size, cell = (640 - MARGIN) / size;
  const x = Math.floor(((event.clientX - rect.left) * scale - MARGIN) / cell);
  const y = Math.floor(((event.clientY - rect.top) * scale - MARGIN) / cell);
  if (x >= 0 && y >= 0 && x < size && y < size) moveApple(x, y);
});

async function poll() {
  try {
    if (!presets) presets = (await api('features')).presets;
    await refresh();
  } catch (error) {
    connected = false;
    render();
    $('status').textContent = 'Cannot connect to the server.';
    showError(error.message);
  }
  setTimeout(poll, 300);
}
poll();

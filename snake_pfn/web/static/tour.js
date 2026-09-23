// Guided first run. The browser owns the tour; each step calls an existing endpoint and
// the "watch" steps advance themselves when the server finishes.
const SEEN = 'snake-pfn.seen-intro';
const $ = id => document.getElementById(id);
const n = value => Number(value).toLocaleString();

const PLAY = {moves: 100, policy: 'tabpfn', auto_fit: true, rounds: 3, versus: true};
const A1 = [0, 0]; // The intro's first apple, top-left.

// Arrow keys around the apple.
const KEYS = `<svg class="keys" viewBox="0 0 160 124" aria-hidden="true">
  <g fill="#24382e" stroke="#4a6b58" stroke-width="1.5">
    <rect x="60" y="2" width="40" height="30" rx="6"/><rect x="14" y="45" width="40" height="30" rx="6"/>
    <rect x="106" y="45" width="40" height="30" rx="6"/><rect x="60" y="90" width="40" height="30" rx="6"/>
  </g>
  <g fill="#e6ecdf" font-size="18" text-anchor="middle" font-family="system-ui, sans-serif">
    <text x="80" y="24">↑</text><text x="34" y="67">←</text><text x="126" y="67">→</text><text x="80" y="112">↓</text>
  </g>
  <circle cx="80" cy="61" r="11" fill="#edb16d"/>
  <path d="M81 51 l4 -5" stroke="#ffdbab" stroke-width="2.5" stroke-linecap="round"/>
</svg>`;

const STEPS = [
  {
    id: 'hook', place: 'center', lit: [],
    title: 'Can you outsmart TabPFN?',
    body: () => `<p>TabPFN plays the snake. <strong>You</strong> play the apple.</p>
      <p class="live"><span class="live-dot"></span>Live demo: every snake move is a real call to TabPFN.</p>`,
    next: 'Show me how',
  },
  {
    // The intro always starts from an empty table (earlier moves are archived, not deleted)
    // and a fresh board with the apple in A1.
    id: 'opponent', place: 'bottom', lit: ['game-panel'],
    enter: c => [...(c.rows ? [['clear', {}]] : []), ['reset', {food: A1}]],
    title: 'This is the board.',
    body: () => '<p>TabPFN will steer the snake. But TabPFN is a tabular model: it does not see pixels or rules. It sees a table.</p>',
  },
  {
    id: 'table', place: 'bottom', lit: ['data-view'],
    title: 'This is everything TabPFN knows about Snake.',
    body: () => '<p>Nothing. The table is empty.</p>',
    next: 'Watch a few moves',
  },
  {
    // Auto-completes when the six moves are in, then waits for a click so the jump to
    // fast practice is a decision the player makes, not something that happens to them.
    id: 'demo', place: 'bottom', lit: ['game-panel', 'data-view'], auto: true,
    title: 'Every move becomes one row.',
    body: () => '<p>The board before the move, the turn it took, and what it earned. Watch the table fill.</p>',
    enter: () => [['random-steps', {moves: 6, delay: 1.2, food: A1}]],
    done: c => c.job === null,
    then: {
      title: 'Six moves. Six rows.',
      body: c => `<p>Not much to learn from yet. TabPFN needs to see far more than a handful of moves before it can play, so let it watch on its own, much faster.</p>
        <p class="progress">${n(c.rows)} rows so far.</p>`,
      next: c => `Let it watch ${n(c.practice_moves)} moves`,
    },
  },
  {
    // Random moves until the table is long enough, then wait for a click before training.
    id: 'practice', place: 'bottom', lit: ['game-panel', 'data-view'], auto: true,
    title: 'Watching random moves.',
    body: c => `<p class="counter"><strong>${n(c.rows)}</strong> moves seen</p>`,
    enter: c => c.rows < c.practice_moves
      ? [['random-steps', {moves: c.practice_moves - c.rows, delay: 0.01}]] : null,
    done: c => c.job === null,
    then: {
      title: c => `${n(c.rows)} moves seen.`,
      body: () => '<p>Enough to learn from.</p>',
      next: 'Next',
    },
  },
  {
    // Training starts quietly on entry; the card is only about the rules.
    id: 'rules', place: 'center', lit: [],
    enter: c => (c.fitted ? null : [['fit', {rounds: 3}]]),
    title: 'Game rules',
    body: c => `<ol>
        <li>TabPFN moves the snake, one live prediction per move. You move the apple.</li>
        <li>After each snake move, step the apple with the arrow keys or tap a neighbouring square.</li>
        <li>Survive ${c.hunger_limit} moves without getting eaten.</li>
      </ol>
      ${KEYS}`,
    next: 'Got it',
  },
  {
    id: 'ready', place: 'center', lit: [],
    title: 'Are you ready?',
    body: () => '',
    next: 'Play',
  },
  {
    // Only seen when Play is pressed while training is still running; the game starts by itself.
    id: 'waiting', place: 'center', lit: [], auto: true, finish: true,
    title: 'Awaiting TabPFN initialization…',
    body: () => '',
    enter: c => (c.fitted || c.job ? null : [['fit', {rounds: 3}]]),
    done: c => c.job === null && c.fitted,
  },
];
const index = id => STEPS.findIndex(s => s.id === id);

let deps = null, current = null, step = null, armed = false, error = null, completed = false;

function panels() { return ['game-panel', 'data-view', 'log-view'].map(id => $(id)); }

function paint() {
  const card = $('tour');
  if (step === null) {
    card.hidden = true;
    delete document.body.dataset.tour;
    panels().forEach(p => p.classList.remove('lit'));
    return;
  }
  const s = STEPS[step], view = completed && s.then ? s.then : s;
  const text = value => (typeof value === 'function' ? (current ? value(current) : '') : value || '');
  document.body.dataset.tour = s.id;
  panels().forEach(p => p.classList.toggle('lit', s.lit.includes(p.id)));
  card.hidden = false;
  card.dataset.place = s.place;
  // paint() runs on every poll; only touch the DOM when something changed, so the card
  // never re-lays out under a click in progress.
  const set = (id, prop, value) => { if ($(id)[prop] !== value) $(id)[prop] = value; };
  set('tour-title', 'textContent', text(view.title));
  set('tour-body', 'innerHTML', text(view.body) + (error ? `<p class="error">${error}</p>` : ''));
  set('tour-dots', 'innerHTML', STEPS.map((_, i) => `<i class="${i === step ? 'on' : ''}"></i>`).join(''));
  set('tour-next', 'hidden', s.auto && !completed && !error);
  set('tour-next', 'textContent', error ? 'Retry' : (text(view.next) || 'Next'));
}

async function enter(i, resumed = false) {
  const s = STEPS[i];
  // enter(current) returns a list of [path, body] calls to make in order, or null.
  const actions = s.enter && !resumed && current ? s.enter(current) : null;
  step = i; error = null; completed = false; armed = resumed || !actions;
  paint();
  if (!actions) return update(current); // Nothing to start: finish at once if already done.
  try {
    for (const [path, body] of actions) await deps.api(path, body);
    await deps.refresh();
    armed = true;
  } catch (exc) {
    error = exc.message;
    paint();
  }
}

function advance() {
  if (STEPS[step].finish) return finish(true);
  enter(step + 1);
}

function finish(play) {
  localStorage.setItem(SEEN, '1');
  step = null; error = null; completed = false;
  paint();
  deps.onChange();
  if (play) deps.command('play', PLAY);
}

function update(c) {
  current = c;
  if (step === null || !c) return;
  const s = STEPS[step];
  if (s.auto && armed && !error && !completed) {
    if (c.error) error = c.error;
    else if (s.done(c)) {
      if (!s.then) return advance();
      completed = true;
    }
  }
  paint();
}

export const tour = {
  init(dependencies) {
    deps = dependencies;
    $('tour-next').onclick = () => {
      console.debug('tour: next pressed on', STEPS[step]?.id, {armed, completed, error});
      return error ? enter(step) : advance();
    };
  },
  seen: () => !!localStorage.getItem(SEEN),
  get active() { return step !== null; },
  begin(id = 'hook') {
    if (id === 'hook') localStorage.removeItem(SEEN);
    enter(index(id));
  },
  // On page load: rejoin a running job, or show the hook to a first-time visitor. An empty
  // table is a first visit too: TabPFN has never seen Snake.
  resume(c) {
    current = c;
    if (c.job === 'random-steps') enter(index(c.rows < 20 ? 'demo' : 'practice'), true);
    else if (c.job === 'fitting') enter(index('waiting'), true);
    else if (!c.job && (c.rows === 0 || !this.seen())) enter(index('hook'));
  },
  update,
  play: () => deps.command('play', PLAY),
};

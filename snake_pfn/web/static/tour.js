// Guided first run. The browser owns the tour; each step calls an existing endpoint and
// the "watch" steps advance themselves when the server finishes.
const SEEN = 'snake-pfn.seen-intro';
const $ = id => document.getElementById(id);
const n = value => Number(value).toLocaleString();

const PLAY = {moves: 100, policy: 'tabpfn', auto_fit: true, rounds: 3, stop_after_episode: true};
const A1 = [0, 0]; // The intro's first apple, top-left.

const STEPS = [
  {
    id: 'hook', place: 'center', lit: [],
    title: 'Snake, played by TabPFN',
    body: c => `<p>First we record some random moves. Then TabPFN uses that table to choose where to go.</p>
      <p class="live"><span class="live-dot"></span>${c.model_mode === 'stub' ? 'Debug: simulated predictions' : 'Live: every move calls TabPFN'}</p>`,
    next: 'Show me',
  },
  {
    // The intro always starts from an empty table (earlier moves are archived, not deleted)
    // and a fresh board with the apple in A1.
    id: 'meet', place: 'bottom', lit: ['game-panel', 'data-view'],
    enter: c => [...(c.rows ? [['clear', {}]] : []), ['reset', {food: A1}]],
    title: 'The board becomes numbers',
    body: () => '<p>TabPFN reads a table of moves and rewards. Let’s put a few moves in it.</p>',
    next: 'Watch a few moves',
  },
  {
    // Auto-completes when the moves are in, then waits for a click so the jump to
    // fast practice is a decision the player makes, not something that happens to them.
    id: 'demo', place: 'bottom', lit: ['game-panel', 'data-view'], auto: true,
    title: 'Every move becomes a row',
    body: () => '<p>We save what the board looked like, which turn the snake took, and the reward it earned.</p>',
    enter: () => [['random-steps', {moves: 5, delay: 0.7, food: A1}]],
    done: c => c.job === null,
    then: {
      title: c => `${n(c.rows)} moves, ${n(c.rows)} rows`,
      body: () => '<p>Let’s collect more examples before TabPFN takes over.</p>',
      next: c => `Watch ${n(c.practice_moves)} moves`,
    },
  },
  {
    // Random moves until the table is long enough, then wait for a click before training.
    id: 'practice', place: 'bottom', lit: ['game-panel', 'data-view'], auto: true,
    title: 'Collecting moves',
    body: c => `<p class="counter"><strong>${n(c.rows)}</strong> moves</p>`,
    enter: c => c.rows < c.practice_moves
      ? [['random-steps', {moves: c.practice_moves - c.rows, delay: 0.003}]] : null,
    done: c => c.job === null,
    then: {
      title: c => `${n(c.rows)} moves`,
      body: () => '<p>Now we can use these moves to fit TabPFN.</p>',
      next: 'Next',
    },
  },
  {
    // Training starts quietly on entry; the card is only about the rules.
    id: 'rules', place: 'center', lit: [],
    enter: c => [['reset', {}], ...(c.fitted ? [] : [['fit', {rounds: 3}]])],
    title: 'The rules',
    body: c => `<ul>
        <li>TabPFN scores all three turns. The snake takes the highest score.</li>
        <li>Each apple adds a point and makes the snake longer.</li>
        <li>Walls, its own body, or ${c.hunger_limit} moves without food end the game.</li>
      </ul>`,
    next: 'Watch TabPFN play',
  },
  {
    // Only seen when Play is pressed while training is still running; the game starts by itself.
    id: 'waiting', place: 'center', lit: [], auto: true, finish: true,
    title: 'Getting TabPFN ready…',
    body: () => '',
    enter: c => (c.fitted || c.job ? null : [['fit', {rounds: 3}]]),
    done: c => c.job === null && c.fitted,
  },
];
const index = id => STEPS.findIndex(s => s.id === id);

let deps = null, current = null, step = null, armed = false, error = null, completed = false;

function panels() { return ['game-panel', 'data-view'].map(id => $(id)); }

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

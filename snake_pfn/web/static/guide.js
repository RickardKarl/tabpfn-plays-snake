import { example } from './guide-example.js';

const tabs = [...document.querySelectorAll('.guide-tabs [role="tab"]')];

function selectTab(selected) {
  tabs.forEach(tab => {
    const active = tab === selected;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    document.getElementById(tab.getAttribute('aria-controls')).hidden = !active;
  });
}

tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => selectTab(tab));
  tab.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = tabs.length - 1;
    else return;
    event.preventDefault();
    selectTab(tabs[next]);
    tabs[next].focus();
  });
});

const byId = id => document.getElementById(id);
const moveButtons = [...document.querySelectorAll('[data-example-action]')];
const names = ['Left', 'Straight', 'Right'];
const directions = [[0, -1], [1, 0], [0, 1], [-1, 0]];
const yesNo = value => value ? 'yes' : 'no';

function showExample(action) {
  const {state, moves} = example;
  const move = moves[action], f = move.features;
  const crash = !!f.danger.will_collide;
  const square = crash ? 'the wall' : `${'ABCD'[move.head[0]]}${move.head[1] + 1}`;
  moveButtons.forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.exampleAction) === action)));
  byId('feature-move-summary').textContent = crash
    ? 'Straight ahead is a wall. The snake would crash.'
    : `${names[action]} takes the head to ${square}, ${f.food.next_food_distance < f.food.food_distance ? 'closer to' : 'farther from'} the apple.`;
  byId('example-distance').textContent = `${f.food.food_distance} → ${f.food.next_food_distance}${crash ? ' (past the wall)' : ''}`;
  byId('example-crash').textContent = crash ? 'Yes' : 'No';
  byId('example-space').textContent = `${f.space.reachable_cells} of ${state.size ** 2}`;
  byId('feature-action').textContent = `${names[action]} → action = ${action}`;
  byId('feature-outcome').textContent = `Crash ${f.outcome.will_crash} · eat ${f.outcome.will_eat} · closer ${f.outcome.closer_to_apple > 0 ? '+' : ''}${f.outcome.closer_to_apple} · room ${f.outcome.room_left}`;

  // Draw the same board for every turn; only the proposed destination changes.
  const cells = [], labels = [];
  for (let y = 0; y < state.size; y++) {
    for (let x = 0; x < state.size; x++) {
      const value = f.board[`cell_${y}_${x}`];
      const type = value === -1 ? 'food' : value === state.snake.length ? 'head' : value > 0 ? 'body' : 'empty';
      cells.push(`<rect class="example-cell ${type}" x="${28 + x * 50}" y="${28 + y * 50}" width="48" height="48" rx="5"/>`);
    }
    labels.push(`<text class="example-axis" x="${52 + y * 50}" y="17">${'ABCD'[y]}</text><text class="example-axis" x="14" y="${57 + y * 50}">${y + 1}</text>`);
  }
  const [dx, dy] = directions[move.direction];
  const hx = 52 + state.snake[0][0] * 50, hy = 52 + state.snake[0][1] * 50;
  const ex = hx + dx * 42, ey = hy + dy * 42;
  const target = crash
    ? '<path class="example-collision" d="M226 133v38"/>'
    : `<rect class="example-target" x="${29 + move.head[0] * 50}" y="${29 + move.head[1] * 50}" width="46" height="46" rx="5"/>`;
  const arrow = `<path class="example-direction${crash ? ' blocked' : ''}" d="M${hx + dx * 14} ${hy + dy * 14}L${ex} ${ey}m${-dx * 7 - dy * 6} ${-dy * 7 + dx * 6}L${ex} ${ey}l${-dx * 7 + dy * 6} ${-dy * 7 - dx * 6}"/>`;
  byId('feature-board').innerHTML = `<title id="feature-board-title">Example: turn ${names[action].toLowerCase()}</title><desc id="feature-board-desc">A four by four board. The snake faces right from D3; the apple is at D1. ${byId('feature-move-summary').textContent}</desc>${labels.join('')}${cells.join('')}${target}${arrow}`;
}

moveButtons.forEach(button => button.addEventListener('click', () => showExample(Number(button.dataset.exampleAction))));
showExample(0);

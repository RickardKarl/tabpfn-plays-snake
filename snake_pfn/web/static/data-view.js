// Read-only view of the rows TabPFN receives. Numbers are shown with readable labels;
// the raw value stays available on hover. No model or API calls happen here.
const pageSize = 20;
let offset = 0;
let loading = false;
let previousTable;
const element = (id) => document.getElementById(id);
const ACTIONS = ['Turn left', 'Go straight', 'Turn right'];
const TURNS = ['↰', '↑', '↱'];
// Kept apart from the history page so a table refresh never drops the current question.
const candidates = {query: null, values: null, chosen: null};

const DIRECTIONS = ['up', 'right', 'down', 'left'];
const COLUMNS = 'ABCDEFGHIJKLMNOP';
const LABELS = {
  action: 'Move',
  heading_x: 'Heading x',
  heading_y: 'Heading y',
  length: 'Snake length',
  hunger_fraction: 'Hunger',
  food_forward: 'Apple ahead',
  food_right: 'Apple to the right',
  food_distance: 'Apple distance',
  next_food_distance: 'Apple distance after move',
  will_eat: 'Eats apple',
  danger_left: 'Crash if left',
  danger_straight: 'Crash if straight',
  danger_right: 'Crash if right',
  will_collide: 'Crash on this move',
  reachable_cells: 'Reachable cells after move',
  space_per_segment: 'Reachable cells per segment',
  will_crash: 'Crashes',
  closer_to_apple: 'Toward apple',
  room_left: 'Room left',
};
const YES_NO = new Set(['will_eat', 'danger_left', 'danger_straight', 'danger_right', 'will_collide',
  'will_crash']);

function label(column) {
  const cell = column.match(/^cell_(\d+)_(\d+)$/);
  if (cell) return `${COLUMNS[Number(cell[2])] ?? cell[2]}${Number(cell[1]) + 1}`;
  return LABELS[column] || column;
}

// Turn one numeric input into words. `head` is the largest body rank in this row.
function describe(column, value, direction, head) {
  if (column === 'action') return {text: DIRECTIONS[direction] ?? String(value)};
  if (column.startsWith('cell_')) {
    if (value === 0) return {text: 'empty', className: 'cell-empty'};
    if (value === -1) return {text: 'apple', className: 'cell-apple'};
    if (value === head) return {text: 'head', className: 'cell-snake cell-head'};
    if (value === 1) return {text: 'tail', className: 'cell-snake'};
    return {text: 'body', className: 'cell-snake'};
  }
  if (column === 'heading_x') return {text: value < 0 ? 'left' : value > 0 ? 'right' : '0'};
  if (column === 'heading_y') return {text: value < 0 ? 'up' : value > 0 ? 'down' : '0'};
  if (column === 'closer_to_apple') return {text: value > 0 ? 'closer' : value < 0 ? 'further' : '—'};
  if (column === 'hunger_fraction') return {text: `${Math.round(value * 100)}%`};
  if (YES_NO.has(column)) return {text: value ? 'yes' : 'no'};
  return {text: Number.isInteger(value) ? String(value) : value.toFixed(2)};
}

// Signed, so good and bad never rely on colour alone: +1 apple, −1 crash, −0.01 a plain step.
function formatOutput(value) {
  if (value === null || value === undefined) return '—';
  // Rewards are ±1 or −0.01; predicted returns need one more decimal.
  const twoDecimals = Math.abs(value * 100 - Math.round(value * 100)) < 1e-9;
  const text = Math.abs(value).toFixed(twoDecimals ? 2 : 3);
  return value > 0 ? `+${text}` : value < 0 ? `−${text}` : text;
}

function tone(value) {
  return value >= 0.05 ? 'good' : value <= -0.05 ? 'bad' : 'neutral';
}

// One table row: a label in the # column, the inputs in words, and the reward.
function drawRow(columns, values, direction, first, reward) {
  const row = document.createElement('tr');
  const head = Math.max(0, ...columns.map((c, i) => c.startsWith('cell_') ? values[i] : 0));
  const number = document.createElement('td');
  number.textContent = first;
  row.append(number);
  values.forEach((value, i) => {
    const cell = document.createElement('td');
    const shown = describe(columns[i], value, direction, head);
    cell.textContent = shown.text;
    if (shown.className) cell.className = shown.className;
    cell.title = `${columns[i]} = ${value}`;
    row.append(cell);
  });
  const output = document.createElement('td');
  output.textContent = formatOutput(reward);
  if (reward !== null && reward !== undefined) {
    output.className = tone(reward);
    output.title = String(reward);
  }
  row.append(output);
  return row;
}

let shown = null; // The saved-move page currently in the table.

function drawTable(data, options = {}) {
  const {start = 0, highlight = -1} = options;
  const table = element('training-table');
  const header = document.createElement('thead');
  const heading = document.createElement('tr');
  ['#', ...data.columns.map(label), 'Reward'].forEach((name, index) => {
    const cell = document.createElement('th');
    cell.scope = 'col';
    cell.textContent = name;
    if (index > 0 && index <= data.columns.length) cell.title = data.columns[index - 1];
    heading.append(cell);
  });
  header.append(heading);
  const body = document.createElement('tbody');
  let highlighted = null;
  data.rows.forEach((values, index) => {
    const row = drawRow(data.columns, values, data.directions?.[index], String(start + index + 1), data.rewards[index]);
    if (index === highlight) { row.classList.add('latest'); highlighted = row; }
    body.append(row);
  });
  table.replaceChildren(header, body);
  shown = {columns: data.columns, last: start + data.rows.length >= data.total};
  drawCandidates();
  return highlighted;
}

// TabPFN's three options, appended under the latest saved move with the reward unknown
// until it answers. They only show on the last page, directly after the move history.
function drawCandidates() {
  const table = element('training-table');
  table.querySelector('tbody.candidates')?.remove();
  const {query, values, chosen} = candidates;
  if (!query || !shown?.last || query.columns.join() !== shown.columns.join()) return;
  const body = document.createElement('tbody');
  body.className = 'candidates';
  query.rows.forEach((inputs, i) => {
    const row = drawRow(query.columns, inputs, query.directions[i], TURNS[i], values ? values[i] : null);
    row.title = `Possible move: ${ACTIONS[i].toLowerCase()}`;
    const reward = row.lastChild;
    if (!values) {
      reward.textContent = '?';
      reward.className = 'pending';
      reward.title = 'Waiting for TabPFN';
    } else {
      reward.title = `Predicted future reward: ${values[i]}`;
      if (i === chosen) {
        row.classList.add('chosen');
        row.title += ' (chosen: highest predicted return)';
      }
    }
    body.append(row);
  });
  table.append(body);
  const box = table.closest('.table-scroll');
  box.scrollTop = box.scrollHeight;
}

// query: rows to ask about (undefined keeps the current ones, null clears); values: the answers.
export function setQuery(query, values = null, chosen = null) {
  if (query !== undefined) candidates.query = query;
  candidates.values = values;
  candidates.chosen = chosen;
  drawCandidates();
}

// Dim the table and sweep a light over it while it is uploaded as TabPFN's context.
export function setLoading(active) {
  element('training-table').closest('.table-frame').classList.toggle('loading', active);
}

// `follow` keeps the view on the latest saved move while a game is running.
export async function refreshTable(follow = false) {
  if (loading) return;
  loading = true;
  try {
    const query = follow ? `latest=true&limit=${pageSize}` : `offset=${offset}&limit=${pageSize}`;
    // Always the observed rewards, never the fitted targets, so the last column means one thing.
    const response = await fetch(`/api/table?${query}`);
    if (!response.ok) throw new Error('Could not load the data table.');
    const data = await response.json();
    if (follow) {
      offset = data.offset;
    } else if (offset && offset >= data.total) {
      offset = 0;
      previousTable = null;
      return; // The next poll loads the first page after a reset.
    }
    element('table-error').hidden = true;
    const fingerprint = JSON.stringify(data);
    if (fingerprint === previousTable) return;
    previousTable = fingerprint;

    const highlight = data.latest_index === null || data.latest_index === undefined
      ? -1 : data.latest_index - offset;
    const highlighted = drawTable(data, {start: offset, highlight});
    if (highlighted && follow && !candidates.query) { // Scroll the table box only, never the page.
      const box = element('training-table').closest('.table-scroll');
      box.scrollTop = Math.max(0, highlighted.offsetTop + highlighted.offsetHeight - box.clientHeight + 8);
    }

    element('table-page').textContent = data.total
      ? `${offset + 1}–${offset + data.rows.length} of ${data.total.toLocaleString()}` : '';
    element('table-prev').disabled = offset === 0;
    element('table-next').disabled = offset + data.rows.length >= data.total;
  } catch (error) {
    element('table-error').hidden = false;
    element('table-error').textContent = error.message;
  } finally {
    loading = false;
  }
}

element('table-prev').onclick = () => {
  if (loading) return;
  offset = Math.max(0, offset - pageSize);
  refreshTable();
};
element('table-next').onclick = () => {
  if (loading) return;
  offset += pageSize;
  refreshTable();
};

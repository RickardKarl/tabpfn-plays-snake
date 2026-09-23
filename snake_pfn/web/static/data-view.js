// Read-only view of the rows TabPFN receives. Numbers are shown with readable labels;
// the raw value stays available on hover. No model or API calls happen here.
const pageSize = 20;
let offset = 0;
let loading = false;
let previousTable;
const element = (id) => document.getElementById(id);
const ACTIONS = ['left', 'straight', 'right'];
// The three candidate rows TabPFN is being asked about, shown as ghost rows at the bottom.
const ghost = {query: null, values: null, chosen: null};
let loadingNote = false;

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
};
const YES_NO = new Set(['will_eat', 'danger_left', 'danger_straight', 'danger_right', 'will_collide']);

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
    return {text: 'snake', className: 'cell-snake'};
  }
  if (column === 'heading_x') return {text: value < 0 ? 'left' : value > 0 ? 'right' : '0'};
  if (column === 'heading_y') return {text: value < 0 ? 'up' : value > 0 ? 'down' : '0'};
  if (column === 'hunger_fraction') return {text: `${Math.round(value * 100)}%`};
  if (YES_NO.has(column)) return {text: value ? 'yes' : 'no'};
  return {text: Number.isInteger(value) ? String(value) : value.toFixed(2)};
}

function formatOutput(value) {
  if (value === null || value === undefined) return '—';
  // Rewards are ±1 or −0.01; fitted targets and Q values need one more decimal.
  const twoDecimals = Math.abs(value * 100 - Math.round(value * 100)) < 1e-9;
  return twoDecimals ? value.toFixed(2) : value.toFixed(3);
}

function drawTable(id, data, outputs, outputLabel, options = {}) {
  const {start = 0, highlight = -1} = options;
  const table = element(id);
  const header = document.createElement('thead');
  const heading = document.createElement('tr');
  ['#', ...data.columns.map(label), outputLabel].forEach((name, index) => {
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
    const row = document.createElement('tr');
    if (index === highlight) { row.className = 'latest'; highlighted = row; }
    const head = Math.max(0, ...data.columns.map((c, i) => c.startsWith('cell_') ? values[i] : 0));
    const number = document.createElement('td');
    number.textContent = String(start + index + 1);
    row.append(number);
    values.forEach((value, i) => {
      const cell = document.createElement('td');
      const shown = describe(data.columns[i], value, data.directions?.[index], head);
      cell.textContent = shown.text;
      if (shown.className) cell.className = shown.className;
      cell.title = `${data.columns[i]} = ${value}`;
      row.append(cell);
    });
    const output = document.createElement('td');
    output.textContent = formatOutput(outputs[index]);
    output.title = outputs[index] === null ? 'Available after training' : String(outputs[index]);
    row.append(output);
    body.append(row);
  });
  table.replaceChildren(header, body);
  renderGhost();
  return highlighted;
}

function renderGhost() {
  const body = element('training-table').tBodies[0];
  if (!body) return;
  body.querySelectorAll('tr.query').forEach(row => row.remove());
  if (!ghost.query) return;
  const {columns, rows, directions} = ghost.query;
  const order = ghost.values ? [0, 1, 2].sort((a, b) => ghost.values[b] - ghost.values[a]) : null;
  rows.forEach((values, i) => {
    const row = document.createElement('tr');
    row.className = 'query' + (ghost.values && i === ghost.chosen ? ' chosen' : '');
    const head = Math.max(0, ...columns.map((c, k) => c.startsWith('cell_') ? values[k] : 0));
    const number = document.createElement('td');
    number.textContent = ghost.values ? (i === ghost.chosen ? '→' : '·') : '?';
    number.title = `Candidate: turn ${ACTIONS[i]}`;
    row.append(number);
    values.forEach((value, k) => {
      const cell = document.createElement('td');
      const shown = describe(columns[k], value, directions[i], head);
      cell.textContent = shown.text;
      if (shown.className) cell.className = shown.className;
      cell.title = `${columns[k]} = ${value}`;
      row.append(cell);
    });
    const output = document.createElement('td');
    if (ghost.values) {
      output.textContent = formatOutput(ghost.values[i]);
      output.className = `rank-${order.indexOf(i)}`;
      output.title = 'Predicted expected future reward. Higher is better.';
    } else {
      output.textContent = '?';
      output.className = 'pending';
      output.title = 'Waiting for TabPFN';
    }
    row.append(output);
    body.append(row);
  });
  const box = element('training-table').closest('.table-scroll');
  box.scrollTop = box.scrollHeight;
}

// query: rows to ask about (undefined keeps the current ones, null clears); values: the answers.
export function setQuery(query, values = null, chosen = null) {
  if (query !== undefined) ghost.query = query;
  ghost.values = values;
  ghost.chosen = chosen;
  renderGhost();
}

// Dim the table and sweep a light over it while it is uploaded as TabPFN's context.
export function setLoading(active) {
  element('training-table').closest('.table-frame').classList.toggle('loading', active);
  if (active) {
    loadingNote = true;
  } else if (loadingNote) {
    loadingNote = false;
    previousTable = null; // Rewrite the note from table data on the next refresh.
  }
}

// `follow` keeps the view on the latest saved move while a game is running.
export async function refreshTable(follow = false) {
  if (loading) return;
  loading = true;
  try {
    const query = follow
      ? `source=experience&latest=true&limit=${pageSize}`
      : `offset=${offset}&limit=${pageSize}`;
    const response = await fetch(`/api/table?${query}`);
    if (!response.ok) throw new Error('Could not load the data table.');
    const data = await response.json();
    if (follow) {
      offset = data.offset;
    } else if (offset && offset >= data.total) {
      offset = 0;
      previousTable = null;
      return; // The next poll loads the first page after a smaller fit or a reset.
    }
    element('table-error').hidden = true;
    const fingerprint = JSON.stringify(data);
    if (fingerprint === previousTable) return;
    previousTable = fingerprint;
    if (loadingNote) { // Keep the upload note while the table is being loaded.
      drawTable('training-table', data, data.source === 'fit' ? data.targets : data.rewards,
        data.source === 'fit' ? 'Target (y)' : 'Reward', {start: offset});
      return;
    }

    const isFit = data.source === 'fit';
    const highlight = data.latest_index === null || data.latest_index === undefined
      ? -1 : data.latest_index - offset;
    const highlighted = drawTable('training-table', data, isFit ? data.targets : data.rewards,
      isFit ? 'Target (y)' : 'Reward', {start: offset, highlight});
    if (highlighted && follow) { // Scroll the table box only, never the page.
      const box = element('training-table').closest('.table-scroll');
      box.scrollTop = Math.max(0, highlighted.offsetTop + highlighted.offsetHeight - box.clientHeight + 8);
    }

    element('table-page').textContent = data.total
      ? `${offset + 1}–${offset + data.rows.length} of ${data.total}` : '0 rows';
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

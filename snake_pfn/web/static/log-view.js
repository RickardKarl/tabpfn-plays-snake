// Debug log: everything the runner, learner, and TabPFN adapter report, newest at the bottom.
let after = 0;
let loading = false;
const maxRows = 1000;
const element = (id) => document.getElementById(id);

function addEntry(entry) {
  const list = element('log-entries');
  const row = document.createElement('div');
  row.className = `entry ${entry.level}`;
  const time = document.createElement('span');
  time.className = 'time';
  const stamp = new Date(entry.time); // Server sends UTC; show local time.
  time.textContent = stamp.toLocaleTimeString([], {hour12: false}) + '.' + String(stamp.getMilliseconds()).padStart(3, '0');
  const source = document.createElement('span');
  source.className = 'source';
  source.textContent = entry.source;
  const message = document.createElement('span');
  message.className = 'message';
  message.textContent = entry.message;
  row.append(time, source, message);
  list.append(row);
  while (list.childElementCount > maxRows) list.firstElementChild.remove();
}

export async function refreshLog() {
  if (loading) return;
  loading = true;
  try {
    const response = await fetch(`/api/log?after=${after}`);
    if (!response.ok) return;
    const data = await response.json();
    if (data.latest < after) { // The server restarted; start over.
      after = 0;
      element('log-entries').replaceChildren();
      return;
    }
    if (!data.entries.length) return;
    const box = element('log-scroll');
    const pinned = box.scrollHeight - box.scrollTop - box.clientHeight < 8;
    data.entries.forEach(addEntry);
    after = data.latest;
    element('log-count').textContent = `${element('log-entries').childElementCount} entries`;
    if (pinned || element('log-follow').checked) box.scrollTop = box.scrollHeight;
  } finally {
    loading = false;
  }
}

element('log-clear').onclick = () => {
  element('log-entries').replaceChildren();
  element('log-count').textContent = '0 entries';
};

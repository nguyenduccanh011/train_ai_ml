const DATA_URL = '../results/leaderboard/leaderboard.json';
const SUMMARY_URL = '../results/leaderboard/summary.json';

let allRows = [];
let summary = {};
let filteredRows = [];
let sortCol = 'composite_score';
let sortDir = -1;
let scoreMode = 'global';
let showSuperseded = false;
let searchQuery = '';
let filters = {
  bundle: '',
  strategy: '',
  feature_set: '',
  entry_model: '',
  year: '',
};

const els = {
  body: document.getElementById('leaderboardBody'),
  visibleCount: document.getElementById('visibleCount'),
  activeCount: document.getElementById('activeCount'),
  groupCount: document.getElementById('groupCount'),
  modeLabel: document.getElementById('modeLabel'),
  globalMode: document.getElementById('globalMode'),
  fairMode: document.getElementById('fairMode'),
  searchInput: document.getElementById('searchInput'),
  showSuperseded: document.getElementById('showSuperseded'),
  bundleFilter: document.getElementById('bundleFilter'),
  strategyFilter: document.getElementById('strategyFilter'),
  featureFilter: document.getElementById('featureFilter'),
  modelFilter: document.getElementById('modelFilter'),
  yearFilter: document.getElementById('yearFilter'),
};

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function formatNum(value, digits = 2) {
  if (value === null || value === undefined || value === '') return '—';
  const num = Number(value);
  if (!Number.isFinite(num)) return escapeHtml(value);
  return num.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

function formatInt(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num.toLocaleString() : '—';
}

function compareValues(a, b, col) {
  const av = col === 'rank' ? a.rank : a[col];
  const bv = col === 'rank' ? b.rank : b[col];
  const an = Number(av);
  const bn = Number(bv);

  if (Number.isFinite(an) && Number.isFinite(bn)) return an - bn;
  return String(av ?? '').localeCompare(String(bv ?? ''), undefined, { numeric: true, sensitivity: 'base' });
}

function rowMatchesSearch(row) {
  if (!searchQuery) return true;
  const haystack = [
    row.run_id,
    row.run_name,
    row.bundle,
    row.strategy,
    row.feature_set,
    row.entry_model,
    row.exit_model_type,
    row.fairness_group_key,
  ].join(' ').toLowerCase();
  return haystack.includes(searchQuery);
}

function windowKey(row) {
  return `${row.first_test_year}–${row.last_test_year}`;
}

function getFairBaselineGroup(rows) {
  if (summary.baseline_fairness_group_key) return summary.baseline_fairness_group_key;
  const activeRows = rows.filter((row) => !row.superseded);
  const candidates = activeRows.length ? activeRows : rows;
  return [...candidates].sort((a, b) => Number(b.composite_score) - Number(a.composite_score))[0]?.fairness_group_key || '';
}

function applyFilters() {
  const fairGroup = scoreMode === 'fair' ? getFairBaselineGroup(allRows) : '';

  filteredRows = allRows.filter((row) => {
    if (!showSuperseded && row.superseded) return false;
    if (fairGroup && row.fairness_group_key !== fairGroup) return false;
    if (filters.bundle && row.bundle !== filters.bundle) return false;
    if (filters.strategy && row.strategy !== filters.strategy) return false;
    if (filters.feature_set && row.feature_set !== filters.feature_set) return false;
    if (filters.entry_model && row.entry_model !== filters.entry_model) return false;
    if (filters.year && windowKey(row) !== filters.year) return false;
    return rowMatchesSearch(row);
  });

  const rankedRows = [...filteredRows].sort((a, b) => Number(b.composite_score) - Number(a.composite_score));
  rankedRows.forEach((row, index) => { row.rank = index + 1; });

  filteredRows.sort((a, b) => compareValues(a, b, sortCol) * sortDir);
  renderStats(fairGroup);
  renderTable();
}

function renderStats(fairGroup) {
  const activeRows = allRows.filter((row) => !row.superseded);
  const groups = new Set(allRows.map((row) => row.fairness_group_key));
  els.visibleCount.textContent = filteredRows.length.toLocaleString();
  els.activeCount.textContent = activeRows.length.toLocaleString();
  els.groupCount.textContent = groups.size.toLocaleString();
  els.modeLabel.textContent = scoreMode === 'fair' ? `Fair ${fairGroup.slice(0, 6)}` : 'Global';
}

function rankBadge(rank) {
  if (rank === 1) return '<span class="badge gold">#1</span>';
  if (rank <= 3) return `<span class="badge silver">#${rank}</span>`;
  if (rank <= 10) return `<span class="badge bronze">#${rank}</span>`;
  return `<span class="muted">#${rank}</span>`;
}

function rowClass(row) {
  const classes = [];
  if (row.superseded) classes.push('superseded');
  if (row.rank === 1) classes.push('top-1');
  else if (row.rank <= 3) classes.push('top-3');
  else if (row.rank <= 10) classes.push('top-10');
  return classes.join(' ');
}

function renderWarnings(row) {
  const warnings = row.warnings || [];
  const fairnessWarnings = [];
  if (row.same_symbols_as_baseline === false) fairnessWarnings.push('different symbol count');
  if (row.same_window_as_baseline === false) fairnessWarnings.push('different test window');
  if (row.same_cost_as_baseline === false) fairnessWarnings.push('different cost profile');
  if (row.same_target_as_baseline === false) fairnessWarnings.push('different target');
  const allWarnings = warnings.concat(fairnessWarnings);
  if (!allWarnings.length) return '<span class="muted">—</span>';
  const title = escapeHtml(allWarnings.join(' | '));
  return `<span class="badge warn" title="${title}">${allWarnings.length}</span>`;
}

function renderFairness(row) {
  const fairKey = escapeHtml(row.fairness_group_key || '');
  const badges = [`<span class="badge" title="${fairKey}">${fairKey.slice(0, 6)}</span>`];
  if (row.is_baseline) badges.push('<span class="badge baseline">Baseline</span>');
  if (row.same_window_as_baseline) badges.push('<span class="badge fair-ok">Same window</span>');
  if ([row.same_symbols_as_baseline, row.same_window_as_baseline, row.same_cost_as_baseline, row.same_target_as_baseline].includes(false)) {
    badges.push('<span class="badge warn">Cross-group</span>');
  }
  return badges.join(' ');
}

function renderExit(row) {
  const label = `${row.exit_model_type || 'none'} ${row.exit_model_enabled ? 'on' : 'off'}`;
  const cls = row.exit_model_enabled ? 'exit-on' : 'exit-off';
  return `<span class="badge ${cls}">${escapeHtml(label)}</span>`;
}

function renderTable() {
  if (!filteredRows.length) {
    els.body.innerHTML = '<tr><td colspan="19" class="empty">No rows match current filters.</td></tr>';
    return;
  }

  els.body.innerHTML = filteredRows.map((row) => {
    const pnlClass = Number(row.avg_pnl) >= 0 ? 'positive' : 'negative';
    const mddClass = Number(row.max_drawdown) > 0 ? 'negative' : 'muted';
    return `
      <tr class="${rowClass(row)}">
        <td>${rankBadge(row.rank)}</td>
        <td class="num positive">${formatNum(row.composite_score, 2)}</td>
        <td class="run-name" title="${escapeHtml(row.run_name)}">${escapeHtml(row.run_name)}</td>
        <td>${escapeHtml(row.bundle)}</td>
        <td>${escapeHtml(row.strategy)}</td>
        <td>${escapeHtml(row.feature_set)}</td>
        <td>${escapeHtml(row.entry_model)}</td>
        <td>${renderExit(row)}</td>
        <td class="num">${formatInt(row.trades)}</td>
        <td class="num">${formatNum(row.wr, 2)}%</td>
        <td class="num ${pnlClass}">${formatNum(row.avg_pnl, 3)}</td>
        <td class="num">${formatNum(row.pf, 2)}</td>
        <td class="num">${formatNum(row.sharpe, 3)}</td>
        <td class="num ${mddClass}">${formatNum(row.max_drawdown, 2)}</td>
        <td class="num">${formatNum(row.yearly_consistency, 3)}</td>
        <td class="num">${formatInt(row.n_symbols)}</td>
        <td>${escapeHtml(windowKey(row))}</td>
        <td>${renderWarnings(row)}</td>
        <td>${renderFairness(row)}</td>
      </tr>`;
  }).join('');
}

function fillSelect(select, values, placeholder) {
  const current = select.value;
  select.innerHTML = `<option value="">${placeholder}</option>` + values.map((value) => (
    `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`
  )).join('');
  select.value = values.includes(current) ? current : '';
}

function renderFilters() {
  const unique = (field) => [...new Set(allRows.map((row) => row[field]).filter(Boolean))].sort();
  fillSelect(els.bundleFilter, unique('bundle'), 'All bundles');
  fillSelect(els.strategyFilter, unique('strategy'), 'All strategies');
  fillSelect(els.featureFilter, unique('feature_set'), 'All feature sets');
  fillSelect(els.modelFilter, unique('entry_model'), 'All entry models');
  fillSelect(els.yearFilter, [...new Set(allRows.map(windowKey))].sort(), 'All windows');
}

function setScoreMode(nextMode) {
  scoreMode = nextMode;
  els.globalMode.classList.toggle('active', scoreMode === 'global');
  els.fairMode.classList.toggle('active', scoreMode === 'fair');
  applyFilters();
}

function bindEvents() {
  els.globalMode.addEventListener('click', () => setScoreMode('global'));
  els.fairMode.addEventListener('click', () => setScoreMode('fair'));
  els.searchInput.addEventListener('input', (event) => {
    searchQuery = event.target.value.trim().toLowerCase();
    applyFilters();
  });
  els.showSuperseded.addEventListener('change', (event) => {
    showSuperseded = event.target.checked;
    applyFilters();
  });

  [
    [els.bundleFilter, 'bundle'],
    [els.strategyFilter, 'strategy'],
    [els.featureFilter, 'feature_set'],
    [els.modelFilter, 'entry_model'],
    [els.yearFilter, 'year'],
  ].forEach(([select, key]) => {
    select.addEventListener('change', (event) => {
      filters[key] = event.target.value;
      applyFilters();
    });
  });

  document.querySelectorAll('th[data-sort]').forEach((th) => {
    th.addEventListener('click', () => {
      const col = th.dataset.sort;
      if (sortCol === col) sortDir *= -1;
      else {
        sortCol = col;
        sortDir = col === 'composite_score' ? -1 : 1;
      }
      applyFilters();
    });
  });
}

async function loadData() {
  try {
    const [dataResponse, summaryResponse] = await Promise.all([
      fetch(DATA_URL, { cache: 'no-store' }),
      fetch(SUMMARY_URL, { cache: 'no-store' }),
    ]);
    if (!dataResponse.ok) throw new Error(`${dataResponse.status} ${dataResponse.statusText}`);
    allRows = await dataResponse.json();
    summary = summaryResponse.ok ? await summaryResponse.json() : {};
    renderFilters();
    applyFilters();
  } catch (error) {
    els.body.innerHTML = `<tr><td colspan="19" class="error">Failed to load ${DATA_URL}: ${escapeHtml(error.message)}</td></tr>`;
  }
}

bindEvents();
loadData();

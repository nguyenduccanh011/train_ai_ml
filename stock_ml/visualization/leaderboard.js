// Market-aware data URLs
const MARKET_CONFIGS = {
  all: {
    dataUrl: '../results/leaderboard/leaderboard.json',
    summaryUrl: '../results/leaderboard/summary.json',
    label: 'All Markets'
  },
  vn_stock: {
    dataUrl: '../results/leaderboard/by_market/vn_stock/leaderboard.json',
    summaryUrl: '../results/leaderboard/by_market/vn_stock/summary.json',
    label: 'VN Stock',
    market: 'vn_stock'
  },
  vn_derivatives_family: {
    dataUrl: '../results/leaderboard/by_market_family/vn_derivatives/leaderboard.json',
    summaryUrl: '../results/leaderboard/by_market_family/vn_derivatives/summary.json',
    label: 'VN Derivatives (All Timeframes)',
    marketFamily: 'vn_derivatives'
  },
  vn_derivatives: {
    dataUrl: '../results/leaderboard/by_market/vn_derivatives/leaderboard.json',
    summaryUrl: '../results/leaderboard/by_market/vn_derivatives/summary.json',
    label: 'VN Derivatives 1H',
    market: 'vn_derivatives',
    timeframe: '1H'
  },
  vn_derivatives_30m: {
    dataUrl: '../results/leaderboard/by_market/vn_derivatives_30m/leaderboard.json',
    summaryUrl: '../results/leaderboard/by_market/vn_derivatives_30m/summary.json',
    label: 'VN Derivatives 30M',
    market: 'vn_derivatives_30m',
    timeframe: '30m'
  },
  vn_derivatives_1d: {
    dataUrl: '../results/leaderboard/by_market/vn_derivatives_1d/leaderboard.json',
    summaryUrl: '../results/leaderboard/by_market/vn_derivatives_1d/summary.json',
    label: 'VN Derivatives 1D',
    market: 'vn_derivatives_1d',
    timeframe: '1D'
  },
  vn_derivatives_15m: {
    dataUrl: '../results/leaderboard/by_market/vn_derivatives_15m/leaderboard.json',
    summaryUrl: '../results/leaderboard/by_market/vn_derivatives_15m/summary.json',
    label: 'VN Derivatives 15M',
    market: 'vn_derivatives_15m',
    timeframe: '15m'
  }
};

const MARKET_STORAGE_KEY = 'leaderboard.market';

function isValidMarket(value) {
  return Object.prototype.hasOwnProperty.call(MARKET_CONFIGS, value);
}

function getInitialMarket() {
  try {
    const params = new URLSearchParams(window.location.search);
    const marketFromQuery = params.get('market');
    if (isValidMarket(marketFromQuery)) return marketFromQuery;
  } catch (_) {}

  try {
    const marketFromStorage = window.localStorage.getItem(MARKET_STORAGE_KEY);
    if (isValidMarket(marketFromStorage)) return marketFromStorage;
  } catch (_) {}

  return 'all';
}

function persistMarketSelection(market) {
  if (!isValidMarket(market)) return;

  try {
    window.localStorage.setItem(MARKET_STORAGE_KEY, market);
  } catch (_) {}

  try {
    const url = new URL(window.location.href);
    if (market === 'all') url.searchParams.delete('market');
    else url.searchParams.set('market', market);
    window.history.replaceState({}, '', url.toString());
  } catch (_) {}
}

let currentMarket = getInitialMarket();
let allRows = [];
let summary = {};
let filteredRows = [];
let sortCol = 'composite_score';
let sortDir = -1;
let scoreMode = 'global';
let showSuperseded = false;
let searchQuery = '';
let apiAvailable = false;
let filters = {
  bundle: '',
  strategy: '',
  feature_set: '',
  entry_model: '',
  year: '',
  state: '',
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
  stateFilter: document.getElementById('stateFilter'),
  marketFilter: document.getElementById('marketFilter'),
  dataPath: document.getElementById('dataPath'),
  apiBanner: document.getElementById('apiBanner'),
  toasts: document.getElementById('toasts'),
};

if (els.marketFilter) {
  els.marketFilter.value = isValidMarket(currentMarket) ? currentMarket : 'all';
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function toast(message, kind = 'info', timeout = 4000) {
  if (!els.toasts) return;
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = message;
  els.toasts.appendChild(el);
  if (timeout) setTimeout(() => el.remove(), timeout);
  return el;
}

async function apiFetch(path, options) {
  const resp = await fetch(`/api${path}`, options);
  if (!resp.ok) {
    let detail = `${resp.status}`;
    try { detail = (await resp.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return resp.json();
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

function pnlClass(value) {
  return Number(value) >= 0 ? 'positive' : 'negative';
}

function tableMessageRow(message, className) {
  const colCount = document.querySelectorAll('thead th').length;
  return `<tr><td colspan="${colCount}" class="${className}">${message}</td></tr>`;
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

function rowMarketFamily(row) {
  if (row.market_family) return row.market_family;
  if (String(row.market || '').startsWith('vn_derivatives')) return 'vn_derivatives';
  return row.market || 'unknown';
}

function rowMatchesMarketConfig(row, cfg) {
  if (cfg.market && row.market !== cfg.market) return false;
  if (cfg.timeframe && row.timeframe !== cfg.timeframe) return false;
  if (cfg.marketFamily && rowMarketFamily(row) !== cfg.marketFamily) return false;
  return true;
}

function applyMarketConfigFilter(rows, cfg) {
  if (!cfg.market && !cfg.timeframe && !cfg.marketFamily) return rows;
  return rows.filter((row) => rowMatchesMarketConfig(row, cfg));
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
    if (filters.state && (row.state || 'trained') !== filters.state) return false;
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

function renderState(row) {
  const state = row.state || 'trained';
  return `<span class="badge state-${escapeHtml(state)}">${escapeHtml(state)}</span>`;
}

function renderActions(row) {
  if (!apiAvailable) return '<span class="muted">—</span>';
  const id = escapeHtml(row.run_id);
  const state = row.state || 'trained';
  const pinLabel = state === 'pinned' ? 'Unpin' : 'Pin';
  const pinTarget = state === 'pinned' ? 'trained' : 'pinned';
  const retireLabel = state === 'retired' ? 'Unretire' : 'Retire';
  const retireTarget = state === 'retired' ? 'trained' : 'retired';
  return `<div class="actions">
    <button class="act-btn pin" data-act="state" data-id="${id}" data-state="${pinTarget}">${pinLabel}</button>
    <button class="act-btn" data-act="state" data-id="${id}" data-state="${retireTarget}">${retireLabel}</button>
    <button class="act-btn" data-act="retrain" data-id="${id}">Retrain</button>
    <button class="act-btn danger" data-act="delcache" data-id="${id}">Del cache</button>
    <button class="act-btn danger" data-act="delete" data-id="${id}">Delete</button>
  </div>`;
}

function renderTable() {
  if (!filteredRows.length) {
    els.body.innerHTML = tableMessageRow('No rows match current filters.', 'empty');
    return;
  }

  els.body.innerHTML = filteredRows.map((row) => {
    const avgPnlClass = pnlClass(row.avg_pnl);
    const totalPnlClass = pnlClass(row.total_pnl);
    const mddClass = Number(row.max_drawdown) > 0 ? 'negative' : 'muted';
    return `
      <tr class="${rowClass(row)}">
        <td>${rankBadge(row.rank)}</td>
        <td>${renderState(row)}</td>
        <td class="num positive">${formatNum(row.composite_score, 2)}</td>
        <td class="run-name" title="${escapeHtml(row.run_name)}">${escapeHtml(row.run_name)}</td>
        <td>${escapeHtml(row.bundle)}</td>
        <td>${escapeHtml(row.strategy)}</td>
        <td>${escapeHtml(row.feature_set)}</td>
        <td>${escapeHtml(row.entry_model)}</td>
        <td>${renderExit(row)}</td>
        <td class="num">${formatInt(row.trades)}</td>
        <td class="num">${formatNum(row.wr, 2)}%</td>
        <td class="num ${avgPnlClass}">${formatNum(row.avg_pnl, 3)}</td>
        <td class="num ${totalPnlClass}">${formatNum(row.total_pnl, 2)}</td>
        <td class="num">${formatNum(row.pf, 2)}</td>
        <td class="num">${formatNum(row.sharpe, 3)}</td>
        <td class="num ${mddClass}">${formatNum(row.max_drawdown, 2)}</td>
        <td class="num">${formatNum(row.yearly_consistency, 3)}</td>
        <td class="num">${formatInt(row.n_symbols)}</td>
        <td>${escapeHtml(windowKey(row))}</td>
        <td>${renderWarnings(row)}</td>
        <td>${renderFairness(row)}</td>
        <td>${renderActions(row)}</td>
      </tr>`;
  }).join('');

  bindRowActions();
}

function findRow(runId) {
  return allRows.find((r) => r.run_id === runId);
}

async function doSetState(runId, state) {
  try {
    await apiFetch(`/runs?run_id=${encodeURIComponent(runId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state }),
    });
    const row = findRow(runId);
    if (row) row.state = state;
    toast(`State → ${state}`, 'success');
    applyFilters();
  } catch (err) {
    toast(`Set state failed: ${err.message}`, 'error');
  }
}

async function doRetrain(runId) {
  try {
    const res = await apiFetch(`/runs/retrain?run_id=${encodeURIComponent(runId)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    toast(`Retrain started (job ${res.job_id})`, 'info', 6000);
    pollJob(res.job_id, runId);
  } catch (err) {
    toast(`Retrain failed: ${err.message}`, 'error');
  }
}

async function pollJob(jobId, runId) {
  try {
    const status = await apiFetch(`/jobs/${jobId}`);
    if (status.status === 'running') {
      setTimeout(() => pollJob(jobId, runId), 4000);
      return;
    }
    if (status.status === 'done') {
      toast(`Retrain done: ${runId.split('/').pop()}`, 'success', 6000);
      loadData(currentMarket);
    } else {
      toast(`Retrain error (exit ${status.exit_code}) — see ${status.log}`, 'error', 8000);
    }
  } catch (err) {
    toast(`Job poll failed: ${err.message}`, 'error');
  }
}

async function doDeleteCache(runId) {
  if (!window.confirm(`Quarantine cache for this run? Backtest metrics stay on the leaderboard.\n\n${runId}`)) return;
  try {
    const res = await apiFetch(`/runs/cache?run_id=${encodeURIComponent(runId)}`, { method: 'DELETE' });
    toast(`Quarantined ${res.quarantined_cache.length} cache file(s)`, 'success');
  } catch (err) {
    toast(`Delete cache failed: ${err.message}`, 'error');
  }
}

async function doDelete(runId) {
  if (!window.confirm(`Delete this run entirely (artifacts + cache)? This removes it from the leaderboard.\n\n${runId}`)) return;
  try {
    await apiFetch(`/runs?run_id=${encodeURIComponent(runId)}`, { method: 'DELETE' });
    toast('Run deleted', 'success');
    loadData(currentMarket);
  } catch (err) {
    toast(`Delete failed: ${err.message}`, 'error');
  }
}

function bindRowActions() {
  els.body.querySelectorAll('button[data-act]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const { act, id, state } = btn.dataset;
      if (act === 'state') doSetState(id, state);
      else if (act === 'retrain') doRetrain(id);
      else if (act === 'delcache') doDeleteCache(id);
      else if (act === 'delete') doDelete(id);
    });
  });
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

function resetFilters() {
  filters = { bundle: '', strategy: '', feature_set: '', entry_model: '', year: '', state: '' };
  searchQuery = '';
  els.searchInput.value = '';
  els.bundleFilter.value = '';
  els.strategyFilter.value = '';
  els.featureFilter.value = '';
  els.modelFilter.value = '';
  els.yearFilter.value = '';
  if (els.stateFilter) els.stateFilter.value = '';
}

function setScoreMode(nextMode) {
  scoreMode = nextMode;
  els.globalMode.classList.toggle('active', scoreMode === 'global');
  els.fairMode.classList.toggle('active', scoreMode === 'fair');
  applyFilters();
}

async function detectApi() {
  try {
    const resp = await fetch('/api/runs?market=__ping__', { cache: 'no-store' });
    apiAvailable = resp.ok;
  } catch (_) {
    apiAvailable = false;
  }
  if (els.apiBanner) {
    if (apiAvailable) {
      els.apiBanner.classList.add('ok');
      els.apiBanner.innerHTML = 'Live mode — actions enabled (Pin / Retrain / Delete).';
    } else {
      els.apiBanner.classList.remove('ok');
    }
  }
}

async function loadFromApi(cfg) {
  const params = new URLSearchParams();
  if (cfg.market) params.set('market', cfg.market);
  const rows = await apiFetch(`/runs${params.toString() ? `?${params}` : ''}`);
  return applyMarketConfigFilter(rows, cfg);
}

async function loadData(market) {
  const cfg = MARKET_CONFIGS[market];
  if (!cfg) {
    allRows = [];
    summary = {};
    els.body.innerHTML = tableMessageRow(`Unknown leaderboard selection: ${escapeHtml(market)}`, 'error');
    return;
  }
  if (els.dataPath) els.dataPath.textContent = apiAvailable ? '/api/runs' : cfg.dataUrl;
  els.body.innerHTML = tableMessageRow(`Loading ${cfg.label}...`, 'empty');

  try {
    if (apiAvailable) {
      allRows = await loadFromApi(cfg);
      // summary still from static file for fairness baseline (best-effort)
      try {
        const sr = await fetch(cfg.summaryUrl, { cache: 'no-store' });
        summary = sr.ok ? await sr.json() : {};
      } catch (_) { summary = {}; }
    } else {
      const [dataResponse, summaryResponse] = await Promise.all([
        fetch(cfg.dataUrl, { cache: 'no-store' }),
        fetch(cfg.summaryUrl, { cache: 'no-store' }),
      ]);
      if (!dataResponse.ok) throw new Error(`${dataResponse.status} ${dataResponse.statusText}`);
      allRows = applyMarketConfigFilter(await dataResponse.json(), cfg);
      summary = summaryResponse.ok ? await summaryResponse.json() : {};
    }
    resetFilters();
    renderFilters();
    applyFilters();
  } catch (error) {
    const msg = market !== 'all'
      ? `${cfg.label} leaderboard not found. Run experiments then rebuild: python -m stock_ml.scripts.build_leaderboard rebuild`
      : `Failed to load leaderboard: ${escapeHtml(error.message)}`;
    els.body.innerHTML = tableMessageRow(msg, 'error');
  }
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
  els.marketFilter.addEventListener('change', (event) => {
    currentMarket = event.target.value;
    persistMarketSelection(currentMarket);
    scoreMode = 'global';
    els.globalMode.classList.add('active');
    els.fairMode.classList.remove('active');
    loadData(currentMarket);
  });

  [
    [els.bundleFilter, 'bundle'],
    [els.strategyFilter, 'strategy'],
    [els.featureFilter, 'feature_set'],
    [els.modelFilter, 'entry_model'],
    [els.yearFilter, 'year'],
    [els.stateFilter, 'state'],
  ].forEach(([select, key]) => {
    if (!select) return;
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
        sortDir = ['composite_score', 'trades', 'wr', 'avg_pnl', 'total_pnl', 'pf', 'sharpe'].includes(col) ? -1 : 1;
      }
      applyFilters();
    });
  });
}

bindEvents();
persistMarketSelection(currentMarket);
detectApi().then(() => loadData(currentMarket));

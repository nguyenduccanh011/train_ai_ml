// Leaderboard data source — always the FastAPI DB endpoint (/api/v1).
// Base URL comes from api-config.js; falls back to the Nginx-relative path.
const getApiBase = () => {
  if (window.API_CONFIG && window.API_CONFIG.baseUrl) {
    return window.API_CONFIG.baseUrl;
  }
  // Fallback to relative path (works in production behind Nginx)
  return '/api/v1';
};

// Per-market client-side filter of the full board (market/timeframe/family).
const MARKET_CONFIGS = {
  all: {
    label: 'All Markets'
  },
  vn_stock: {
    label: 'VN Stock',
    market: 'vn_stock'
  },
  vn_derivatives_family: {
    label: 'VN Derivatives (All Timeframes)',
    marketFamily: 'vn_derivatives'
  },
  vn_derivatives: {
    label: 'VN Derivatives 1H',
    market: 'vn_derivatives',
    timeframe: '1H'
  },
  vn_derivatives_30m: {
    label: 'VN Derivatives 30M',
    market: 'vn_derivatives_30m',
    timeframe: '30m'
  },
  vn_derivatives_1d: {
    label: 'VN Derivatives 1D',
    market: 'vn_derivatives_1d',
    timeframe: '1D'
  },
  vn_derivatives_15m: {
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
let filteredRows = [];
let sortCol = 'composite_score';
let sortDir = -1;
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
  experiment_group: '',
};

const els = {
  body: document.getElementById('leaderboardBody'),
  visibleCount: document.getElementById('visibleCount'),
  activeCount: document.getElementById('activeCount'),
  searchInput: document.getElementById('searchInput'),
  showSuperseded: document.getElementById('showSuperseded'),
  experimentGroupFilter: document.getElementById('experimentGroupFilter'),
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
  const resp = await fetch(`${getApiBase()}${path}`, options);
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
  ].join(' ').toLowerCase();
  return haystack.includes(searchQuery);
}

function windowKey(row) {
  return `${row.first_test_year}–${row.last_test_year}`;
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
  filteredRows = allRows.filter((row) => {
    if (!showSuperseded && row.superseded) return false;
    if (filters.bundle && row.bundle !== filters.bundle) return false;
    if (filters.strategy && row.strategy !== filters.strategy) return false;
    if (filters.feature_set && row.feature_set !== filters.feature_set) return false;
    if (filters.entry_model && row.entry_model !== filters.entry_model) return false;
    if (filters.year && windowKey(row) !== filters.year) return false;
    if (filters.state && (row.state || 'trained') !== filters.state) return false;
    if (filters.experiment_group && (row.experiment_group || 'ungrouped') !== filters.experiment_group) return false;
    return rowMatchesSearch(row);
  });

  const rankedRows = [...filteredRows].sort((a, b) => Number(b.composite_score) - Number(a.composite_score));
  rankedRows.forEach((row, index) => { row.rank = index + 1; });

  filteredRows.sort((a, b) => compareValues(a, b, sortCol) * sortDir);
  renderStats();
  renderTable();
}

function renderStats() {
  const activeRows = allRows.filter((row) => !row.superseded);
  els.visibleCount.textContent = filteredRows.length.toLocaleString();
  els.activeCount.textContent = activeRows.length.toLocaleString();
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
  if (!warnings.length) return '<span class="muted">—</span>';
  const title = escapeHtml(warnings.join(' | '));
  return `<span class="badge warn" title="${title}">${warnings.length}</span>`;
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
    const pnlPctClass = pnlClass(row.pnl_pct);
    const mddClass = Number(row.max_drawdown) > 0 ? 'negative' : 'muted';
    const groupBadge = row.experiment_group ? `<span class="badge">${escapeHtml(row.experiment_group)}</span>` : '<span class="muted">—</span>';
    const typeBadge = row.variant_type ? `<span class="badge">${escapeHtml(row.variant_type)}</span>` : '<span class="muted">—</span>';
    const windowKey = row.backtest_window_key && row.backtest_window_key !== 'unknown' ? escapeHtml(row.backtest_window_key) : (row.first_test_year && row.last_test_year ? `${row.first_test_year}–${row.last_test_year}` : '—');
    const commission = row.cost_commission || (row.cost_profile ? row.cost_profile.commission : 'unknown');
    const costDisplay = commission && commission !== 'unknown' ? escapeHtml(String(commission)) : '—';
    const detailsUrl = `model-details.html?run_id=${encodeURIComponent(row.run_id)}`;
    return `
      <tr class="${rowClass(row)}">
        <td>${rankBadge(row.rank)}</td>
        <td>${row.state || 'trained'}</td>
        <td class="num positive">${formatNum(row.composite_score, 2)}</td>
        <td class="run-name" title="${escapeHtml(row.name)}">${escapeHtml(row.name)}</td>
        <td><a href="${detailsUrl}" style="color: #2962ff; text-decoration: none; font-size: 12px;">Chi tiết →</a></td>
        <td title="${escapeHtml(row.metadata_notes || '')}">${groupBadge}</td>
        <td>${typeBadge}</td>
        <td>${escapeHtml(row.market || '—')}</td>
        <td>${escapeHtml(row.timeframe || '—')}</td>
        <td>${windowKey}</td>
        <td class="num">${formatInt(row.n_symbols)}</td>
        <td class="num">${costDisplay}</td>
        <td class="num">${formatInt(row.trades)}</td>
        <td class="num">${formatNum(row.wr * 100, 1)}%</td>
        <td class="num">${formatNum(row.pf, 2)}</td>
        <td class="num ${avgPnlClass}">${formatNum(row.avg_pnl * 100, 2)}%</td>
        <td class="num ${pnlPctClass}">${formatNum(row.pnl_pct, 2)}%</td>
        <td class="num ${row.cagr_overlay != null ? pnlClass(row.cagr_overlay) : 'muted'}" title="${row.cagr_overlay != null ? 'Stage-2 overlay chính thức (stock_ml.portfolio)' + (row.maxdd_overlay != null ? ' · DD ' + formatNum(row.maxdd_overlay * 100, 1) + '%' : '') + (row.overlay_note ? ' · ' + String(row.overlay_note).replace(/"/g, '') : '') : 'chưa chạy overlay chính thức'}">${row.cagr_overlay != null ? formatNum(row.cagr_overlay * 100, 1) + '%' : '—'}</td>
        <td class="num ${row.cagr_nav != null ? pnlClass(row.cagr_nav) : 'muted'}" title="${row.nav_adv != null ? 'NAV ×' + formatNum(row.nav_adv, 2) + ' (T+0 lý thuyết)' : 'chưa chấm NAV sim'}">${row.cagr_nav != null ? formatNum(row.cagr_nav * 100, 1) + '%' : '—'}</td>
        <td class="num ${row.cagr_t2 != null ? pnlClass(row.cagr_t2) : 'muted'}" title="${row.cagr_t2 != null ? 'CAGR dưới T+2 thực tế' + (row.maxdd_t2 != null ? ' · DD ' + formatNum(row.maxdd_t2 * 100, 1) + '%' : '') : 'chưa có T+2 (không phải model combo offline)'}">${row.cagr_t2 != null ? formatNum(row.cagr_t2 * 100, 1) + '%' : '—'}</td>
        <td class="num ${row.maxdd_nav != null ? 'negative' : 'muted'}">${row.maxdd_nav != null ? formatNum(row.maxdd_nav * 100, 1) + '%' : '—'}</td>
        <td class="num">${formatNum(row.max_win * 100, 2)}%</td>
        <td class="num">${formatNum(row.max_loss * 100, 2)}%</td>
        <td class="num">${formatNum(row.avg_hold, 1)}</td>
        <td class="num ${mddClass}">${formatNum(row.max_drawdown * 100, 2)}%</td>
        <td class="num">${formatNum(row.sharpe, 2)}</td>
        <td>${row.audit_status || '—'}</td>
        <td>${row.date_run || '—'}</td>
        <td title="${escapeHtml(row.notes)}">${escapeHtml((row.notes || '').substring(0, 30))}</td>
      </tr>`;
  }).join('');

  bindRowActions();
}

function findRow(runId) {
  return allRows.find((r) => r.run_id === runId);
}

async function doSetState(runId, state) {
  try {
    await apiFetch(`/runs/${encodeURIComponent(runId)}/state`, {
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

async function doDeleteCache(runId) {
  if (!window.confirm(`Quarantine cache for this run? Backtest metrics stay on the leaderboard.\n\n${runId}`)) return;
  try {
    const res = await apiFetch(`/runs/${encodeURIComponent(runId)}/cache`, { method: 'DELETE' });
    toast(`Quarantined ${res.quarantined_cache.length} cache file(s)`, 'success');
  } catch (err) {
    toast(`Delete cache failed: ${err.message}`, 'error');
  }
}

async function doDelete(runId) {
  if (!window.confirm(`Delete this run entirely (artifacts + cache)? This removes it from the leaderboard.\n\n${runId}`)) return;
  try {
    await apiFetch(`/runs/${encodeURIComponent(runId)}`, { method: 'DELETE' });
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
  fillSelect(els.experimentGroupFilter, unique('experiment_group'), 'All groups');
  fillSelect(els.bundleFilter, unique('bundle'), 'All bundles');
  fillSelect(els.strategyFilter, unique('strategy'), 'All strategies');
  fillSelect(els.featureFilter, unique('feature_set'), 'All feature sets');
  fillSelect(els.modelFilter, unique('entry_model'), 'All entry models');
  fillSelect(els.yearFilter, [...new Set(allRows.map(windowKey))].sort(), 'All windows');
}

function resetFilters() {
  filters = { bundle: '', strategy: '', feature_set: '', entry_model: '', year: '', state: '', experiment_group: '' };
  searchQuery = '';
  els.searchInput.value = '';
  els.experimentGroupFilter.value = '';
  els.bundleFilter.value = '';
  els.strategyFilter.value = '';
  els.featureFilter.value = '';
  els.modelFilter.value = '';
  els.yearFilter.value = '';
  if (els.stateFilter) els.stateFilter.value = '';
}

async function detectApi() {
  // Always use API - load from DB only, no JSON fallback
  apiAvailable = true;
  if (els.apiBanner) {
    els.apiBanner.classList.add('ok');
    els.apiBanner.innerHTML = 'Database mode — data from /api/v1/leaderboard';
  }
}

async function loadFromApi(cfg) {
  // Load from DB via API - no JSON fallback.
  // Fetch the full board (market/strategy filtering happens client-side below),
  // so runs beyond the old 200-row default (e.g. rule_only models) are visible.
  const response = await fetch(`${getApiBase()}/leaderboard?limit=5000`);
  if (!response.ok) throw new Error(`API error: ${response.status}`);
  const data = await response.json();
  const rows = data.models || [];
  return applyMarketConfigFilter(rows, cfg);
}

async function loadData(market) {
  const cfg = MARKET_CONFIGS[market];
  if (!cfg) {
    allRows = [];
    els.body.innerHTML = tableMessageRow(`Unknown leaderboard selection: ${escapeHtml(market)}`, 'error');
    return;
  }
  if (els.dataPath) els.dataPath.textContent = '/api/runs';
  els.body.innerHTML = tableMessageRow(`Loading ${cfg.label}...`, 'empty');

  try {
    allRows = await loadFromApi(cfg);
    resetFilters();
    renderFilters();
    applyFilters();
  } catch (error) {
    els.body.innerHTML = tableMessageRow(
      `Failed to load ${cfg.label}: ${escapeHtml(error.message)}`, 'error');
  }
}

function bindEvents() {
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
    loadData(currentMarket);
  });

  [
    [els.experimentGroupFilter, 'experiment_group'],
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
        sortDir = ['composite_score', 'trades', 'wr', 'avg_pnl', 'total_pnl', 'pf', 'sharpe', 'cagr_overlay', 'cagr_nav', 'cagr_t2'].includes(col) ? -1 : 1;
      }
      applyFilters();
    });
  });
}

// =============================================================================
// Cache management functions
// =============================================================================

async function loadCacheStats() {
  try {
    const data = await apiFetch('/cache/stats');
    els.cacheFeatureMb = document.getElementById('cacheFeatureMb');
    els.cacheOrphanCount = document.getElementById('cacheOrphanCount');
    els.cacheOrphanMb = document.getElementById('cacheOrphanMb');
    els.cacheTrashMb = document.getElementById('cacheTrashMb');

    if (els.cacheFeatureMb) els.cacheFeatureMb.textContent = formatNum(data.feature_cache_mb, 1);
    if (els.cacheOrphanCount) els.cacheOrphanCount.textContent = data.orphan_count;
    if (els.cacheOrphanMb) els.cacheOrphanMb.textContent = formatNum(data.orphan_mb, 1);
    if (els.cacheTrashMb) els.cacheTrashMb.textContent = formatNum(data.trash_mb, 1);
  } catch (err) {
    toast(`Load cache stats failed: ${err.message}`, 'error');
  }
}

async function doGcSweep(apply) {
  const action = apply ? 'Sweep & Quarantine' : 'Dry-run GC';
  try {
    const data = await apiFetch('/gc/sweep', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ apply }),
    });
    const msg = `${action}: ${data.orphan_count} orphan(s) (${data.orphan_mb} MB)${apply ? ' → quarantined' : ' (dry-run)'}`;
    toast(msg, apply ? 'success' : 'info', 6000);
    if (apply) {
      loadCacheStats();
    }
  } catch (err) {
    toast(`GC failed: ${err.message}`, 'error');
  }
}

async function doPurgeTrash() {
  if (!window.confirm('Delete trash batches older than 7 days? This cannot be undone.')) return;
  try {
    const data = await apiFetch('/cache/purge-trash', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ older_than_days: 7.0 }),
    });
    toast(`Purged ${data.purged_dirs} trash batch(es), freed ${data.freed_mb} MB`, 'success', 6000);
    loadCacheStats();
  } catch (err) {
    toast(`Purge trash failed: ${err.message}`, 'error');
  }
}

async function doBulkRetire() {
  if (!window.confirm('Retire ALL trained models? (change state to retired)')) return;
  try {
    const data = await apiFetch('/runs/bulk-state', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        state: 'retired',
        filter: { current_state: 'trained' },
      }),
    });
    toast(`Retired ${data.updated} model(s)`, 'success', 6000);
    loadData(currentMarket);
  } catch (err) {
    toast(`Bulk retire failed: ${err.message}`, 'error');
  }
}

async function doBulkDelete() {
  if (!window.confirm('Delete ALL retired models? This removes them from leaderboard and deletes artifacts. Cannot be undone.')) return;
  try {
    const data = await apiFetch('/runs/bulk', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state: 'retired', confirm: true }),
    });
    toast(`Deleted ${data.deleted} model(s), freed ${data.freed_mb} MB`, 'success', 8000);
    loadData(currentMarket);
    loadCacheStats();
  } catch (err) {
    toast(`Bulk delete failed: ${err.message}`, 'error');
  }
}

function toggleCachePanel() {
  const wrap = document.getElementById('cachePanelWrap');
  if (!wrap) return;
  const show = wrap.style.display === 'none';
  wrap.style.display = show ? 'flex' : 'none';
  if (show) {
    loadCacheStats();
  }
}

function bindCachePanel() {
  const btnToggle = document.getElementById('btnToggleCache');
  const btnStats = document.getElementById('btnCacheStats');
  const btnDry = document.getElementById('btnGcDryRun');
  const btnApply = document.getElementById('btnGcApply');
  const btnPurge = document.getElementById('btnPurgeTrash');
  const btnRetire = document.getElementById('btnRetireAll');
  const btnDelete = document.getElementById('btnDeleteRetired');

  if (btnToggle) btnToggle.addEventListener('click', toggleCachePanel);
  if (btnStats) btnStats.addEventListener('click', loadCacheStats);
  if (btnDry) btnDry.addEventListener('click', () => doGcSweep(false));
  if (btnApply) btnApply.addEventListener('click', () => doGcSweep(true));
  if (btnPurge) btnPurge.addEventListener('click', doPurgeTrash);
  if (btnRetire) btnRetire.addEventListener('click', doBulkRetire);
  if (btnDelete) btnDelete.addEventListener('click', doBulkDelete);
}

bindEvents();
bindCachePanel();
persistMarketSelection(currentMarket);
detectApi().then(() => {
  loadData(currentMarket);
  if (apiAvailable) {
    loadCacheStats();
  }
});

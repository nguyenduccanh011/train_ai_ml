// Main app logic — data loading, toggle handlers, init
//
// Data source: the platform API (Postgres leaderboard + DuckDB OHLCV), NOT the
// legacy manifest.json/data_*/ static files. The in-memory shapes (manifest,
// modelIndices, currentRawData with {vk}_markers/_trades/_stats) are kept
// identical to the file-based version so chart.js/ui.js/state.js are untouched.

const DASHBOARD_SELECTION_KEY = 'dashboard.selection';

// Max comparison layers when falling back to top-ranked models (no pinned models).
const MAX_COMPARISON_LAYERS = 6;
// Distinct overlay colors assigned per model by rank.
const MODEL_PALETTE = [
  '#2962ff', '#26a69a', '#ef5350', '#ab47bc',
  '#ffa726', '#66bb6a', '#42a5f5', '#ec407a',
];

// Per-model trades grouped by symbol: { version_key: { SYMBOL: [trade, ...] } }.
// Trades endpoint returns the whole run, so fetch once and reuse across symbols.
var modelTradesCache = {};

function getApiBase() {
  return (window.API_CONFIG && window.API_CONFIG.baseUrl) || '/api/v1';
}

async function apiFetch(path) {
  const resp = await fetch(getApiBase() + path);
  if (!resp.ok) throw new Error(`API ${path} -> ${resp.status}`);
  return resp.json();
}

function marketFamilyOf(market) {
  const m = String(market || '');
  if (m.startsWith('vn_derivatives')) return 'vn_derivatives';
  return m || 'vn_stock';
}

function round2(v) { return Math.round((Number(v) || 0) * 100) / 100; }

// Pinned models first, then fill up to MAX_COMPARISON_LAYERS with top-ranked
// non-retired models (rows arrive ranked). Pinning highlights without hiding the
// rest of the top models.
function selectComparisonModels(rows) {
  const pinned = rows.filter(r => r.state === 'pinned');
  const pinnedIds = new Set(pinned.map(r => r.run_id));
  const fill = rows.filter(r => r.state !== 'retired' && !pinnedIds.has(r.run_id));
  return pinned.concat(fill).slice(0, Math.max(pinned.length, MAX_COMPARISON_LAYERS));
}

function buildModelsFromRows(rows) {
  return rows.map((r, i) => ({
    version_key: r.run_id,
    run_id: r.run_id,
    name: r.run_name || r.name || r.run_id,
    color: MODEL_PALETTE[i % MODEL_PALETTE.length],
    marker_shape: 'arrowUp',
    active: true,
    market: r.market || 'vn_stock',
    market_family: marketFamilyOf(r.market),
    timeframe: r.timeframe || 'unknown',
    composite_score: r.composite_score,
    state: r.state,
    order: i,
  }));
}

// Build modelIndices[vk] from /symbol-stats: per-symbol PnL/WR/trade-count used
// by the symbol dropdown and the "model covers this symbol" filter. DB stores
// pnl/win_rate as fractions; the dashboard expects percent → scale by 100.
async function loadModelIndex(model) {
  const vk = model.version_key;
  try {
    const data = await apiFetch(`/runs/${encodeURIComponent(model.run_id)}/symbol-stats`);
    const stats = (data && data.symbol_stats) || [];
    modelIndices[vk] = stats.map(s => ({
      symbol: s.symbol,
      [vk + '_pnl']: round2((s.total_pnl || 0) * 100),
      [vk + '_wr']: round2((s.win_rate || 0) * 100),
      [vk + '_trades']: s.trades || 0,
    }));
  } catch (e) {
    console.error('symbol-stats load failed for', vk, e);
    modelIndices[vk] = [];
  }
}

// Fetch a run's trades once, group by symbol, convert pnl_pct fraction → percent.
async function loadModelTrades(model) {
  const vk = model.version_key;
  if (modelTradesCache[vk]) return modelTradesCache[vk];
  const bySymbol = {};
  try {
    const data = await apiFetch(`/runs/${encodeURIComponent(model.run_id)}/trades`);
    for (const t of (data.trades || [])) {
      const sym = t.symbol;
      if (!sym) continue;
      (bySymbol[sym] = bySymbol[sym] || []).push({
        symbol: sym,
        entry_date: t.entry_date ? String(t.entry_date).slice(0, 10) : t.entry_date,
        exit_date: t.exit_date ? String(t.exit_date).slice(0, 10) : t.exit_date,
        entry_price: t.entry_price,
        exit_price: t.exit_price,
        holding_days: t.holding_days,
        pnl_pct: (Number(t.pnl_pct) || 0) * 100,
        exit_reason: t.exit_reason,
      });
    }
  } catch (e) {
    console.error('trades load failed for', vk, e);
  }
  modelTradesCache[vk] = bySymbol;
  return bySymbol;
}

// Port of unified_export.make_markers: entry arrow below bar, exit arrow above
// bar colored by win/loss with a PnL label and exit-reason abbreviation.
function buildMarkersForModel(trades, model) {
  const markers = [];
  const abbrevs = (manifest && manifest.exit_abbreviations) || {};
  const color = model.color;
  const shape = model.marker_shape || 'arrowUp';
  const name = model.name;
  const winColor = '#4caf50', lossColor = '#f44336';
  for (const t of trades) {
    const ed = t.entry_date, xd = t.exit_date;
    const pnl = Number(t.pnl_pct) || 0; // already percent
    const reason = t.exit_reason || '';
    const tag = abbrevs[reason] || (reason ? reason.slice(0, 2).toUpperCase() : '');
    if (ed) {
      markers.push({
        time: ed, position: 'belowBar', color, shape,
        text: `${name} Buy`, size: 1, method: model.version_key,
      });
    }
    if (xd) {
      markers.push({
        time: xd, position: 'aboveBar', color: pnl >= 0 ? winColor : lossColor,
        shape: 'arrowDown', size: 1, method: model.version_key,
        text: `${name} ${pnl >= 0 ? '+' : ''}${pnl.toFixed(1)}%${tag ? ' ' + tag : ''}`,
      });
    }
  }
  return markers;
}

function getDashboardMarkets() {
  if (!manifest || !manifest.models) return ['all', 'vn_stock', 'vn_derivatives'];
  return ['all', ...new Set(manifest.models.map(getModelMarketFamily).filter(Boolean))].sort();
}

function normalizeMarket(market) {
  const value = typeof market === 'string' && market.trim() ? market.trim() : 'all';
  return getDashboardMarkets().includes(value) ? value : 'all';
}

function normalizeTimeframe(timeframe) {
  return typeof timeframe === 'string' && timeframe.trim() ? timeframe.trim() : 'all';
}

function normalizeYear(year) {
  return normalizeYearSelection(year);
}

function normalizeSymbol(symbol) {
  return typeof symbol === 'string' ? symbol.trim() : '';
}

function readStoredSelection() {
  try {
    const raw = window.localStorage.getItem(DASHBOARD_SELECTION_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch (_) {
    return {};
  }
}

function readSelectionFromQuery() {
  try {
    const params = new URLSearchParams(window.location.search);
    return {
      market: params.get('market'),
      timeframe: params.get('timeframe'),
      year: params.get('year'),
      symbol: params.get('symbol'),
    };
  } catch (_) {
    return {};
  }
}

function getInitialSelection() {
  const stored = readStoredSelection();
  const query = readSelectionFromQuery();
  return {
    market: normalizeMarket(query.market || stored.market || 'all'),
    timeframe: normalizeTimeframe(query.timeframe || stored.timeframe || 'all'),
    year: normalizeYear(query.year || stored.year || 'all'),
    symbol: normalizeSymbol(query.symbol || stored.symbol || ''),
  };
}

function persistSelection() {
  const selection = {
    market: normalizeMarket(currentMarketFamily),
    timeframe: normalizeTimeframe(currentTimeframe),
    year: normalizeYear(currentYear),
    symbol: normalizeSymbol(currentSymbol),
  };

  try {
    window.localStorage.setItem(DASHBOARD_SELECTION_KEY, JSON.stringify(selection));
  } catch (_) {}

  try {
    const url = new URL(window.location.href);
    if (selection.market === 'all') url.searchParams.delete('market');
    else url.searchParams.set('market', selection.market);
    if (selection.timeframe === 'all') url.searchParams.delete('timeframe');
    else url.searchParams.set('timeframe', selection.timeframe);
    if (selection.year === 'all') url.searchParams.delete('year');
    else url.searchParams.set('year', selection.year);
    if (selection.symbol) url.searchParams.set('symbol', selection.symbol);
    else url.searchParams.delete('symbol');
    window.history.replaceState({}, '', url.toString());
  } catch (_) {}
}

function renderMarketOptions() {
  const marketSelect = document.getElementById('marketSelect');
  if (!marketSelect) return;
  const labels = { all: 'All Markets', vn_stock: 'VN Stock', vn_derivatives: 'VN Derivatives' };
  const desired = normalizeMarket(currentMarketFamily);
  marketSelect.innerHTML = '';
  for (const market of getDashboardMarkets()) {
    const opt = document.createElement('option');
    opt.value = market;
    opt.textContent = labels[market] || market;
    marketSelect.appendChild(opt);
  }
  marketSelect.value = desired;
}

function syncMarketSelect() {
  renderMarketOptions();
}

function syncTimeframeSelect() {
  const timeframeSelect = document.getElementById('timeframeSelect');
  if (!timeframeSelect) return;
  renderTimeframeOptions();
  if (timeframeSelect.value !== currentTimeframe) timeframeSelect.value = currentTimeframe;
}

function syncYearSelect() {
  const yearSelect = document.getElementById('yearSelect');
  if (!yearSelect) return;
  renderYearOptions();
  if (yearSelect.value !== currentYear) yearSelect.value = currentYear;
}

function findSymbolItem(symbol) {
  const target = normalizeSymbol(symbol).toUpperCase();
  if (!target) return null;
  return allSymbolItems.find((item) => String(item.symbol || '').toUpperCase() === target) || null;
}

let initialSelection = getInitialSelection();
currentMarketFamily = initialSelection.market;
currentMarket = initialSelection.market;
currentTimeframe = initialSelection.timeframe;
currentYear = initialSelection.year;
syncMarketSelect();

window.toggleLayer = function(vk) {
  modelVisibility[vk] = !modelVisibility[vk];
  const btn = document.getElementById('btn_' + vk);
  if (btn) btn.classList.toggle('active');
  if (candleSeries) candleSeries.setMarkers(getMarkers());
  updateStatsVisibility();
};

window.toggleTable = function() {
  showTable = !showTable;
  const btn = document.getElementById('btnTable');
  if (btn) btn.classList.toggle('active');
  document.getElementById('tablesContainer').style.display = showTable ? 'flex' : 'none';
  const ch = document.getElementById('chart');
  ch.style.height = showTable ? 'calc(100vh - 500px)' : 'calc(100vh - 300px)';
  if (chart) chart.resize(ch.clientWidth, ch.clientHeight);
};

function normalizeOhlcvForChart(ohlcv) {
  return (ohlcv || []).map(d => ({
    ...d,
    time: normalizeTimeForChart(d.time),
  }));
}

function normalizeMarkersForChart(markers) {
  return (markers || []).map(marker => ({
    ...marker,
    time: normalizeTimeForChart(marker.time),
  }));
}

function hasIntradayTimes(ohlcv) {
  return (ohlcv || []).some(d => typeof d.time === 'number' || isIntradayString(d.time));
}

function applyTimeScaleOptions(ohlcv) {
  if (!chart) return;
  chart.applyOptions({
    timeScale: { timeVisible: hasIntradayTimes(ohlcv), secondsVisible: false }
  });
}

function isSelectedYear(year) {
  if (currentYear === 'all') return true;
  return String(year) === currentYear;
}

function isTradeInSelectedYear(trade) {
  if (currentYear === 'all') return true;
  const entryYear = getYearFromDateLike(trade && trade.entry_date);
  const exitYear = getYearFromDateLike(trade && trade.exit_date);
  return isSelectedYear(entryYear) || isSelectedYear(exitYear);
}

function filterOhlcvBySelectedYear(ohlcv) {
  if (currentYear === 'all') return ohlcv || [];
  return (ohlcv || []).filter(row => {
    const year = getYearFromDateLike(row && row.time);
    return isSelectedYear(year);
  });
}

function filterMarkersBySelectedYear(markers) {
  if (currentYear === 'all') return markers || [];
  return (markers || []).filter(marker => {
    const year = getYearFromDateLike(marker && marker.time);
    return isSelectedYear(year);
  });
}

function filterTradesBySelectedYear(trades) {
  if (currentYear === 'all') return trades || [];
  return (trades || []).filter(isTradeInSelectedYear);
}

function buildCurrentDataFromRaw() {
  if (!currentRawData) {
    currentData = null;
    return null;
  }

  const filtered = {
    symbol: currentRawData.symbol,
    ohlcv: normalizeOhlcvForChart(filterOhlcvBySelectedYear(currentRawData.ohlcv || [])),
  };

  for (const [key, value] of Object.entries(currentRawData)) {
    if (key === 'symbol' || key === 'ohlcv') continue;

    if (key.endsWith('_markers') && Array.isArray(value)) {
      filtered[key] = normalizeMarkersForChart(filterMarkersBySelectedYear(value));
      continue;
    }

    if (key.endsWith('_trades') && Array.isArray(value)) {
      const filteredTrades = filterTradesBySelectedYear(value);
      filtered[key] = filteredTrades;
      const versionKey = key.slice(0, -('_trades'.length));
      if (currentYear === 'all') {
        filtered[versionKey + '_stats'] = currentRawData[versionKey + '_stats'] || buildStatsFromTrades(filteredTrades, versionKey);
      } else {
        filtered[versionKey + '_stats'] = buildStatsFromTrades(filteredTrades, versionKey);
      }
      continue;
    }

    if (key.endsWith('_stats')) {
      if (!(key in filtered)) filtered[key] = value;
      continue;
    }

    filtered[key] = value;
  }

  currentData = filtered;
  return filtered;
}

function applyCurrentView() {
  const data = buildCurrentDataFromRaw();
  if (!data) return;
  if (!chart) createChart();

  const ohlcv = data.ohlcv || [];
  applyTimeScaleOptions(ohlcv);
  candleSeries.setData(ohlcv);
  volumeSeries.setData(ohlcv.map(d => ({
    time: d.time, value: d.volume,
    color: d.close >= d.open ? 'rgba(38,166,154,0.3)' : 'rgba(239,83,80,0.3)'
  })));
  candleSeries.setMarkers(getMarkers());
  chart.timeScale().fitContent();
  renderStats();
  updateStatsVisibility();
  renderTradePanels();
  updateSearchInputDisplay();
  persistSelection();
}

// `arg` is a symbol (the file-based dashboard passed a JSON path here; the API
// flow passes the symbol directly — symbol items now carry file === symbol).
window.loadSymbol = async function(arg) {
  if (!arg) return;
  const symbol = String(arg).toUpperCase();

  let ohlcv = [];
  try {
    const data = await apiFetch(`/ohlcv/${encodeURIComponent(symbol)}`);
    ohlcv = data.ohlcv || [];
  } catch (e) {
    console.error('ohlcv load failed for', symbol, e);
  }

  const raw = { symbol, ohlcv };
  currentSymbol = symbol;
  persistSelection();

  const symbolModels = getFilteredModels().filter(model => {
    const idx = modelIndices[model.version_key] || [];
    return idx.some(entry => entry.symbol === symbol);
  });
  for (const model of symbolModels) {
    const vk = model.version_key;
    const bySymbol = await loadModelTrades(model);
    const trades = bySymbol[symbol] || [];
    raw[vk + '_trades'] = trades;
    raw[vk + '_markers'] = buildMarkersForModel(trades, model);
    raw[vk + '_stats'] = buildStatsFromTrades(trades, vk);
  }

  currentRawData = raw;
  syncYearSelect();
  applyCurrentView();
};

function getBaseDirForMarket(market) {
  const dirs = manifest.base_data_dirs || {};
  return dirs[market] || (market === 'vn_stock' ? manifest.base_data_dir : null) || manifest.base_data_dir || 'data';
}

async function loadBaseIndex(market) {
  if (baseIndices[market]) return baseIndices[market];
  // The symbol universe comes from modelIndices (per-model symbol-stats), not a
  // base OHLCV index file. An empty base index makes renderSymbolSelector fall
  // back to file === symbol, which loadSymbol resolves via the OHLCV endpoint.
  baseIndices[market] = { symbols: [] };
  return baseIndices[market];
}

function mergeBaseIndices(markets) {
  const bySymbol = new Map();
  for (const market of markets) {
    const index = baseIndices[market] || { symbols: [] };
    for (const entry of index.symbols || []) {
      if (!bySymbol.has(entry.symbol)) bySymbol.set(entry.symbol, entry);
    }
  }
  return { symbols: Array.from(bySymbol.values()) };
}

function getActiveBaseIndex() {
  return mergeBaseIndices(getActiveMarketsForBaseData());
}

async function refreshDashboardSelection(preferredSymbol) {
  const activeMarkets = getActiveMarketsForBaseData();
  await Promise.all(activeMarkets.map(loadBaseIndex));
  modelVisibility = {};
  syncMarketSelect();
  syncTimeframeSelect();
  syncYearSelect();
  persistSelection();
  renderToggleButtons();
  renderLegend();
  renderSymbolSelector(getActiveBaseIndex(), modelIndices);

  const targetItem = findSymbolItem(preferredSymbol) || findSymbolItem(currentSymbol);
  if (targetItem) {
    await loadSymbol(targetItem.file);
    return;
  }
  if (allSymbolItems.length > 0) {
    await loadSymbol(allSymbolItems[0].file);
    return;
  }

  currentRawData = null;
  currentData = null;
  currentSymbol = null;
  const searchInput = document.getElementById('symbolSearchInput');
  if (searchInput && !searchInput.matches(':focus')) searchInput.value = '';
  syncYearSelect();
  persistSelection();
}

window.switchMarket = async function(market, preferredSymbol) {
  currentMarketFamily = normalizeMarket(market);
  currentMarket = currentMarketFamily;
  currentTimeframe = 'all';
  await refreshDashboardSelection(preferredSymbol);
};

window.switchTimeframe = async function(timeframe, preferredSymbol) {
  currentTimeframe = normalizeTimeframe(timeframe);
  await refreshDashboardSelection(preferredSymbol);
};

window.switchYear = function(year) {
  currentYear = normalizeYear(year);
  syncYearSelect();
  persistSelection();
  applyCurrentView();
};

async function init() {
  try {
    // Build the manifest from the leaderboard (Postgres) instead of manifest.json.
    const data = await apiFetch('/leaderboard?limit=500');
    const rows = (data && data.models) || [];
    if (!rows.length) throw new Error('leaderboard empty');

    const chosen = selectComparisonModels(rows);
    manifest = {
      base_data_dir: 'data',
      base_data_dirs: {},
      base_symbols: [],
      models: buildModelsFromRows(chosen),
      market_groups: {},
      exit_abbreviations: {},
    };

    initialSelection = getInitialSelection();
    currentMarketFamily = initialSelection.market;
    currentMarket = initialSelection.market;
    currentTimeframe = initialSelection.timeframe;
    currentYear = initialSelection.year;

    modelIndices = {};
    await Promise.all(manifest.models.map(loadModelIndex));

    syncMarketSelect();
    syncTimeframeSelect();
    syncYearSelect();
    await refreshDashboardSelection(initialSelection.symbol);
  } catch (e) {
    console.error('Init error:', e);
    document.getElementById('chart').innerHTML =
      '<div style="text-align:center;padding:40px;color:#888">' +
      'Không tải được dữ liệu từ API.<br>' +
      'Kiểm tra API server đang chạy và endpoint <code>/api/v1/leaderboard</code>.</div>';
  }
}

init();

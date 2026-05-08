// Main app logic — data loading, toggle handlers, init

const DASHBOARD_SELECTION_KEY = 'dashboard.selection';

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
    symbol: normalizeSymbol(query.symbol || stored.symbol || ''),
  };
}

function persistSelection() {
  const selection = {
    market: normalizeMarket(currentMarketFamily),
    timeframe: normalizeTimeframe(currentTimeframe),
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

function findSymbolItem(symbol) {
  const target = normalizeSymbol(symbol).toUpperCase();
  if (!target) return null;
  return allSymbolItems.find((item) => String(item.symbol || '').toUpperCase() === target) || null;
}

let initialSelection = getInitialSelection();
currentMarketFamily = initialSelection.market;
currentMarket = initialSelection.market;
currentTimeframe = initialSelection.timeframe;
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

async function loadModelPayload(dataDir, symbol) {
  const cacheKey = dataDir + '/' + symbol;
  if (modelDataCache[cacheKey]) return modelDataCache[cacheKey];
  try {
    const resp = await fetch('./' + dataDir + '/' + symbol + '.json');
    if (!resp.ok) return {};
    const data = await resp.json();
    modelDataCache[cacheKey] = data;
    return data;
  } catch (_) {
    return {};
  }
}

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

window.loadSymbol = async function(file) {
  if (!file) return;
  const isJsonFile = file.endsWith && file.endsWith('.json');
  try {
    if (isJsonFile) {
      const resp = await fetch(file);
      currentData = await resp.json();
    } else {
      currentData = { symbol: file, ohlcv: [] };
    }
  } catch (_) {
    currentData = { symbol: file.split('/').pop().replace('.json', ''), ohlcv: [] };
  }
  currentSymbol = currentData.symbol;
  persistSelection();

  const symbolModels = getFilteredModels().filter(model => {
    const idx = modelIndices[model.version_key] || [];
    return idx.some(entry => entry.symbol === currentSymbol);
  });
  for (const model of symbolModels) {
    const payload = await loadModelPayload(model.data_dir, currentSymbol);
    for (const key of Object.keys(payload)) {
      if (key === 'symbol') continue;
      if (key.endsWith('_markers') && Array.isArray(payload[key])) {
        currentData[key] = normalizeMarkersForChart(payload[key]);
      } else {
        currentData[key] = payload[key];
      }
    }
  }

  if (!chart) createChart();
  const ohlcv = normalizeOhlcvForChart(currentData.ohlcv || []);
  currentData.ohlcv = ohlcv;
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
};

function getBaseDirForMarket(market) {
  const dirs = manifest.base_data_dirs || {};
  return dirs[market] || (market === 'vn_stock' ? manifest.base_data_dir : null) || manifest.base_data_dir || 'data';
}

async function loadBaseIndex(market) {
  const baseDir = getBaseDirForMarket(market);
  if (baseIndices[market]) return baseIndices[market];
  try {
    const resp = await fetch('./' + baseDir + '/index.json');
    if (!resp.ok) return { symbols: [] };
    const index = await resp.json();
    baseIndices[market] = index;
    return index;
  } catch (_) {
    baseIndices[market] = { symbols: [] };
    return baseIndices[market];
  }
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

  currentSymbol = null;
  const searchInput = document.getElementById('symbolSearchInput');
  if (searchInput && !searchInput.matches(':focus')) searchInput.value = '';
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

async function init() {
  try {
    // Load manifest
    const manifestResp = await fetch('./manifest.json');
    if (!manifestResp.ok) throw new Error('manifest.json not found');
    manifest = await manifestResp.json();
    initialSelection = getInitialSelection();
    currentMarketFamily = initialSelection.market;
    currentMarket = initialSelection.market;
    currentTimeframe = initialSelection.timeframe;

    for (const market of new Set((manifest.models || []).map(getModelMarket).filter(Boolean))) {
      await loadBaseIndex(market);
    }

    modelIndices = {};
    for (const model of manifest.models) {
      try {
        const r = await fetch('./' + model.data_dir + '/index.json');
        if (r.ok) {
          const idx = await r.json();
          modelIndices[model.version_key] = idx.symbols || [];
        }
      } catch (_) {}
    }

    syncMarketSelect();
    syncTimeframeSelect();
    await refreshDashboardSelection(initialSelection.symbol);
  } catch (e) {
    console.error('Init error:', e);
    document.getElementById('chart').innerHTML =
      '<div style="text-align:center;padding:40px;color:#888">' +
      'No manifest.json found. Run:<br>' +
      '<code>cd stock_ml && python -m src.export.unified_export</code></div>';
  }
}

init();

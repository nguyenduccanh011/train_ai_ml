// V22 dashboard app orchestration
async function fetchV22Json(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} → HTTP ${response.status}`);
  return response.json();
}

function normalizeV22Symbols(baseIndex, modelIndex) {
  const baseMap = new Map();
  const prefix = v22CurrentMode.prefix;
  for (const item of (baseIndex && baseIndex.symbols) || []) {
    if (item && item.symbol) baseMap.set(item.symbol, item.file || `data/${item.symbol}.json`);
  }

  return ((modelIndex && modelIndex.symbols) || [])
    .filter(item => item && item.symbol)
    .map(item => ({
      symbol: String(item.symbol).toUpperCase(),
      file: baseMap.get(String(item.symbol).toUpperCase()) || `data/${String(item.symbol).toUpperCase()}.json`,
      pnl: v22Num(item[`${prefix}_pnl`]),
      wr: v22Num(item[`${prefix}_wr`]),
      trades: v22Num(item[`${prefix}_trades`]),
    }));
}

function v22CacheKey(symbol) {
  return `${v22CurrentMode.prefix}:${symbol}`;
}

async function loadV22Mode(modeKey) {
  const nextMode = v22Modes[modeKey] || v22Modes.walk;
  v22CurrentMode = nextMode;
  v22ModelIndex = null;
  v22AllSymbols = [];
  v22FilteredSymbols = [];
  v22DataCache.clear();
  setV22Status(`Đang tải ${nextMode.label}...`);

  try {
    v22ModelIndex = await fetchV22Json(`${nextMode.dataDir}/index.json`);
    v22AllSymbols = normalizeV22Symbols(v22BaseIndex, v22ModelIndex);
    if (!v22AllSymbols.length) throw new Error(`${nextMode.dataDir}/index.json không có symbol`);
    applyV22SymbolFilter();
    const keepCurrent = v22CurrentSymbol && v22AllSymbols.some(s => s.symbol === v22CurrentSymbol);
    await loadV22Symbol(keepCurrent ? v22CurrentSymbol : v22FilteredSymbols[0].symbol);
  } catch (error) {
    console.error(error);
    setV22Status(`Không tải được ${nextMode.label}: ${error.message}`, true);
  }
}

async function loadV22Symbol(symbol) {
  if (!symbol) return;
  v22CurrentSymbol = symbol;
  const input = document.getElementById('v22SymbolSearch');
  input.value = symbol;
  setV22Status(`Đang tải ${symbol}...`);
  applyV22SymbolFilter();

  try {
    const cacheKey = v22CacheKey(symbol);
    if (!v22DataCache.has(cacheKey)) {
      const item = v22AllSymbols.find(s => s.symbol === symbol);
      const basePath = item ? item.file : `data/${symbol}.json`;
      const payloadPath = `${v22CurrentMode.dataDir}/${symbol}.json`;
      const [baseData, payload] = await Promise.all([
        fetchV22Json(basePath),
        fetchV22Json(payloadPath),
      ]);
      v22DataCache.set(cacheKey, { baseData, payload });
    }

    const cached = v22DataCache.get(cacheKey);
    v22CurrentBaseData = cached.baseData;
    v22CurrentPayload = cached.payload;
    setV22ChartData(v22CurrentBaseData, v22CurrentPayload);
    renderV22All(v22CurrentBaseData, v22CurrentPayload);
    const stats = v22CurrentPayload[`${v22CurrentMode.prefix}_stats`] || {};
    setV22Status(`${v22CurrentMode.label} · ${symbol} · ${v22Num(stats.total_trades).toFixed(0)} trades · ${getV22Markers(v22CurrentPayload).length} markers đang hiển thị`);
  } catch (error) {
    console.error(error);
    setV22Status(`Không tải được ${symbol}: ${error.message}`, true);
  }
}

function setupV22Controls() {
  const input = document.getElementById('v22SymbolSearch');
  const dropdown = document.getElementById('v22SymbolDropdown');
  const sortBy = document.getElementById('v22SortBy');
  const modeSelect = document.getElementById('v22Mode');

  input.addEventListener('input', () => {
    applyV22SymbolFilter();
    dropdown.classList.add('open');
  });
  input.addEventListener('focus', () => {
    applyV22SymbolFilter();
    dropdown.classList.add('open');
  });
  input.addEventListener('keydown', event => {
    if (event.key === 'Enter' && v22FilteredSymbols.length) {
      loadV22Symbol(v22FilteredSymbols[0].symbol);
      dropdown.classList.remove('open');
    }
    if (event.key === 'Escape') dropdown.classList.remove('open');
  });
  document.addEventListener('click', event => {
    if (!event.target.closest('.symbol-selector')) dropdown.classList.remove('open');
  });
  sortBy.addEventListener('change', () => {
    applyV22SymbolFilter();
    dropdown.classList.add('open');
  });
  modeSelect.addEventListener('change', () => {
    loadV22Mode(modeSelect.value);
  });

  document.querySelectorAll('.filter-btn[data-filter]').forEach(button => {
    button.addEventListener('click', () => {
      const key = button.dataset.filter;
      v22MarkerFilters[key] = !v22MarkerFilters[key];
      button.classList.toggle('active', v22MarkerFilters[key]);
      refreshV22Markers();
      if (v22CurrentPayload) {
        setV22Status(`${v22CurrentMode.label} · ${v22CurrentSymbol} · ${getV22Markers(v22CurrentPayload).length} markers đang hiển thị`);
      }
    });
  });
}

async function initV22Dashboard() {
  try {
    setV22Status('Đang tải index V22...');
    createV22Chart();
    setupV22Controls();

    v22BaseIndex = await fetchV22Json('data/index.json');
    await loadV22Mode(document.getElementById('v22Mode').value);
  } catch (error) {
    console.error(error);
    setV22Status(`Không khởi tạo được V22 dashboard: ${error.message}`, true);
  }
}

document.addEventListener('DOMContentLoaded', initV22Dashboard);

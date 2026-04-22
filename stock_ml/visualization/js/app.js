// Main app logic — data loading, toggle handlers, init

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

window.loadSymbol = async function(file) {
  if (!file) return;
  // Load base OHLCV data
  const resp = await fetch(file);
  currentData = await resp.json();
  currentSymbol = currentData.symbol;

  // Load each model's data for this symbol
  for (const model of manifest.models) {
    const vk = model.version_key;
    const payload = await loadModelPayload(model.data_dir, currentSymbol);
    // Merge model data into currentData
    for (const key of Object.keys(payload)) {
      if (key !== 'symbol') currentData[key] = payload[key];
    }
  }

  // Also check if base data has model data embedded (legacy format)
  // This handles the old format where v19_1, v19_3, v20, v18, rule data
  // is embedded in the base data/{SYM}.json files

  if (!chart) createChart();
  candleSeries.setData(currentData.ohlcv);
  volumeSeries.setData(currentData.ohlcv.map(d => ({
    time: d.time, value: d.volume,
    color: d.close >= d.open ? 'rgba(38,166,154,0.3)' : 'rgba(239,83,80,0.3)'
  })));
  candleSeries.setMarkers(getMarkers());
  chart.timeScale().fitContent();
  renderStats();
  updateStatsVisibility();
  renderTradePanels();
  // Update search input to show current symbol
  updateSearchInputDisplay();
};

async function init() {
  try {
    // Load manifest
    const manifestResp = await fetch('./manifest.json');
    if (!manifestResp.ok) throw new Error('manifest.json not found');
    manifest = await manifestResp.json();

    // Update title
    const modelNames = manifest.models.map(m => m.name);
    document.querySelector('h1').textContent = modelNames.join(' vs ');

    // Render dynamic UI
    renderToggleButtons();
    renderLegend();

    // Load base index
    const baseDir = manifest.base_data_dir || 'data';
    const baseResp = await fetch('./' + baseDir + '/index.json');
    const baseIndex = await baseResp.json();

    // Load model indices
    const modelIndices = {};
    for (const model of manifest.models) {
      try {
        const r = await fetch('./' + model.data_dir + '/index.json');
        if (r.ok) {
          const idx = await r.json();
          modelIndices[model.version_key] = idx.symbols || [];
        }
      } catch (_) {}
    }

    // Render symbol selector
    renderSymbolSelector(baseIndex, modelIndices);

    // Load first symbol
    if (baseIndex.symbols && baseIndex.symbols.length > 0) {
      loadSymbol(baseIndex.symbols[0].file);
    }
  } catch (e) {
    console.error('Init error:', e);
    document.getElementById('chart').innerHTML =
      '<div style="text-align:center;padding:40px;color:#888">' +
      'No manifest.json found. Run:<br>' +
      '<code>cd stock_ml && python -m src.export.unified_export</code></div>';
  }
}

init();

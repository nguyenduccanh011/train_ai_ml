// UI rendering functions — all dynamic from manifest
function pnlCls(v) { return v >= 0 ? 'positive' : 'negative'; }
function fmt(v) { return v > 0 ? '+' + v : '' + v; }

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1,3), 16);
  const g = parseInt(hex.slice(3,5), 16);
  const b = parseInt(hex.slice(5,7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function getModelMarket(model) {
  if (model.market) return model.market;
  if ((model.matrix_bundle || '').startsWith('derivatives_')) return 'vn_derivatives';
  return 'vn_stock';
}

function getModelMarketFamily(model) {
  if (model.market_family) return model.market_family;
  const market = getModelMarket(model);
  if (market && market.startsWith('vn_derivatives')) return 'vn_derivatives';
  return market || 'vn_stock';
}

function getModelTimeframe(model) {
  return model.timeframe || 'unknown';
}

function getFilteredModels() {
  if (!manifest || !manifest.models) return [];
  return manifest.models.filter(model => {
    const family = getModelMarketFamily(model);
    const timeframe = getModelTimeframe(model);
    if (currentMarketFamily !== 'all' && family !== currentMarketFamily) return false;
    if (currentTimeframe !== 'all' && timeframe !== currentTimeframe) return false;
    return true;
  });
}

function getMarketForSymbol(symbol) {
  for (const [market, index] of Object.entries(baseIndices)) {
    if ((index.symbols || []).some(s => s.symbol === symbol)) return market;
  }
  for (const model of manifest.models || []) {
    const idx = modelIndices[model.version_key] || [];
    if (idx.some(s => s.symbol === symbol)) return getModelMarket(model);
  }
  return 'vn_stock';
}

function getActiveMarketsForBaseData() {
  const markets = new Set(getFilteredModels().map(getModelMarket).filter(Boolean));
  if (markets.size > 0) return Array.from(markets);
  if (currentMarketFamily !== 'all') return [currentMarketFamily];
  return ['vn_stock'];
}

function getTimeframesForFamily(family) {
  if (!manifest || !manifest.models) return [];
  const frames = new Set();
  for (const model of manifest.models) {
    if (family !== 'all' && getModelMarketFamily(model) !== family) continue;
    frames.add(getModelTimeframe(model));
  }
  return Array.from(frames).sort();
}

function renderTimeframeOptions() {
  const sel = document.getElementById('timeframeSelect');
  if (!sel) return;
  const frames = getTimeframesForFamily(currentMarketFamily);
  sel.innerHTML = '<option value="all">All Timeframes</option>';
  for (const frame of frames) {
    const opt = document.createElement('option');
    opt.value = frame;
    opt.textContent = frame;
    sel.appendChild(opt);
  }
  if (currentTimeframe !== 'all' && !frames.includes(currentTimeframe)) currentTimeframe = 'all';
  sel.value = currentTimeframe;
}

function getAvailableYearsForCurrentSymbol() {
  const years = new Set();
  const data = currentRawData || currentData || {};

  for (const candle of data.ohlcv || []) {
    const y = getYearFromDateLike(candle && candle.time);
    if (y) years.add(String(y));
  }

  for (const [key, value] of Object.entries(data)) {
    if (!key.endsWith('_trades') || !Array.isArray(value)) continue;
    for (const trade of value) {
      const entryYear = getYearFromDateLike(trade && trade.entry_date);
      const exitYear = getYearFromDateLike(trade && trade.exit_date);
      if (entryYear) years.add(String(entryYear));
      if (exitYear) years.add(String(exitYear));
    }
  }

  if (years.size === 0) {
    for (const [key, value] of Object.entries(data)) {
      if (!key.endsWith('_markers') || !Array.isArray(value)) continue;
      for (const marker of value) {
        const y = getYearFromDateLike(marker && marker.time);
        if (y) years.add(String(y));
      }
    }
  }

  return Array.from(years).sort((a, b) => Number(b) - Number(a));
}

function renderYearOptions() {
  const sel = document.getElementById('yearSelect');
  if (!sel) return;
  const years = getAvailableYearsForCurrentSymbol();
  sel.innerHTML = '<option value="all">All Years</option>';
  for (const year of years) {
    const opt = document.createElement('option');
    opt.value = year;
    opt.textContent = year;
    sel.appendChild(opt);
  }
  if (currentYear !== 'all' && years.length > 0 && !years.includes(currentYear)) currentYear = 'all';
  sel.value = currentYear;
}

function roundStat(value, digits) {
  const num = Number(value);
  if (!Number.isFinite(num)) return 0;
  const factor = 10 ** digits;
  return Math.round(num * factor) / factor;
}

function computeMedian(values) {
  if (!values || values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return sorted[mid];
  return (sorted[mid - 1] + sorted[mid]) / 2;
}

function buildStatsFromTrades(trades, versionHint) {
  const tradeList = Array.isArray(trades) ? trades : [];
  const pnls = tradeList
    .map(t => Number(t && t.pnl_pct))
    .filter(Number.isFinite);
  const holdDays = tradeList
    .map(t => Number(t && t.holding_days))
    .filter(Number.isFinite);

  if (pnls.length === 0) {
    return {
      total_trades: 0,
      wins: 0,
      losses: 0,
      win_rate: 0,
      total_pnl_pct: 0,
      avg_pnl_pct: 0,
      median_pnl_pct: 0,
      std_pnl_pct: 0,
      avg_win_pct: 0,
      avg_loss_pct: 0,
      payoff_ratio: 0,
      max_win_pct: 0,
      max_loss_pct: 0,
      avg_hold: 0,
      pf: 0,
      version: versionHint || '',
    };
  }

  const wins = pnls.filter(v => v >= 0);
  const losses = pnls.filter(v => v < 0);
  const totalPnl = pnls.reduce((acc, v) => acc + v, 0);
  const avgPnl = totalPnl / pnls.length;
  const avgWin = wins.length ? wins.reduce((acc, v) => acc + v, 0) / wins.length : 0;
  const avgLoss = losses.length ? losses.reduce((acc, v) => acc + v, 0) / losses.length : 0;
  const variance = pnls.reduce((acc, v) => acc + (v - avgPnl) ** 2, 0) / pnls.length;
  const grossWin = wins.reduce((acc, v) => acc + v, 0);
  const grossLoss = Math.abs(losses.reduce((acc, v) => acc + v, 0));
  const pf = grossLoss > 0 ? (grossWin / grossLoss) : 0;
  const payoff = avgLoss !== 0 ? Math.abs(avgWin / avgLoss) : 0;
  const avgHold = holdDays.length ? holdDays.reduce((acc, v) => acc + v, 0) / holdDays.length : 0;

  return {
    total_trades: pnls.length,
    wins: wins.length,
    losses: losses.length,
    win_rate: roundStat((wins.length * 100) / pnls.length, 1),
    total_pnl_pct: roundStat(totalPnl, 2),
    avg_pnl_pct: roundStat(avgPnl, 2),
    median_pnl_pct: roundStat(computeMedian(pnls), 2),
    std_pnl_pct: roundStat(Math.sqrt(variance), 2),
    avg_win_pct: roundStat(avgWin, 2),
    avg_loss_pct: roundStat(avgLoss, 2),
    payoff_ratio: roundStat(payoff, 2),
    max_win_pct: roundStat(wins.length ? Math.max(...wins) : 0, 2),
    max_loss_pct: roundStat(losses.length ? Math.min(...losses) : 0, 2),
    avg_hold: roundStat(avgHold, 1),
    pf: roundStat(pf, 2),
    version: versionHint || '',
  };
}

function timeKeyForLookup(value) {
  const normalized = normalizeTimeForChart(value);
  if (typeof normalized === 'number' && Number.isFinite(normalized)) return `n:${normalized}`;
  if (typeof normalized === 'string' && normalized) return `s:${normalized}`;
  return '';
}

function buildCloseLookup(ohlcv) {
  const lookup = new Map();
  for (const row of ohlcv || []) {
    const key = timeKeyForLookup(row && row.time);
    const close = Number(row && row.close);
    if (!key || !Number.isFinite(close)) continue;
    lookup.set(key, close);
  }
  return lookup;
}

function getTradePrice(trade, priceField, dateField, closeLookup) {
  const direct = Number(trade && trade[priceField]);
  if (Number.isFinite(direct)) return direct;
  const key = timeKeyForLookup(trade && trade[dateField]);
  if (!key || !closeLookup.has(key)) return null;
  const mapped = Number(closeLookup.get(key));
  return Number.isFinite(mapped) ? mapped : null;
}

function getTradePoints(trade, closeLookup) {
  const pnlPct = Number(trade && trade.pnl_pct);
  if (!Number.isFinite(pnlPct)) return null;

  const entryPrice = getTradePrice(trade, 'entry_price', 'entry_date', closeLookup);
  const exitPrice = getTradePrice(trade, 'exit_price', 'exit_date', closeLookup);
  const side = String(trade && (trade.side || trade.position_side || '')).toLowerCase();

  if (Number.isFinite(entryPrice) && Number.isFinite(exitPrice)) {
    if (side.includes('short')) return entryPrice - exitPrice;
    if (side.includes('long')) return exitPrice - entryPrice;

    const delta = exitPrice - entryPrice;
    if (delta === 0 || pnlPct === 0) return delta;
    return Math.sign(delta) === Math.sign(pnlPct) ? delta : -delta;
  }

  if (Number.isFinite(entryPrice)) {
    return entryPrice * (pnlPct / 100);
  }

  if (Number.isFinite(exitPrice)) {
    const denom = 1 + (pnlPct / 100);
    if (denom === 0) return null;
    const impliedEntry = exitPrice / denom;
    return exitPrice - impliedEntry;
  }

  return null;
}

function maxDrawdownFromValues(values) {
  if (!values || values.length === 0) return 0;
  let cum = 0;
  let peak = 0;
  let maxDd = 0;
  for (const v of values) {
    const value = Number(v);
    if (!Number.isFinite(value)) continue;
    cum += value;
    peak = Math.max(peak, cum);
    maxDd = Math.max(maxDd, peak - cum);
  }
  return -roundStat(maxDd, 2);
}

function summarizeTradePoints(trades, ohlcv) {
  const closeLookup = buildCloseLookup(ohlcv);
  const points = [];
  for (const trade of trades || []) {
    const pt = getTradePoints(trade, closeLookup);
    if (Number.isFinite(pt)) points.push(pt);
  }

  const total = roundStat(points.reduce((acc, v) => acc + v, 0), 2);
  const maxDd = maxDrawdownFromValues(points);
  return {
    total_points: total,
    max_drawdown_points: maxDd,
    count: points.length,
  };
}

function maxDrawdownFromTrades(trades) {
  if (!trades || trades.length === 0) return 0;
  let cum = 0, peak = 0, maxDd = 0;
  for (const t of trades) {
    const pnl = Number(t && t.pnl_pct);
    if (!Number.isFinite(pnl)) continue;
    cum += pnl;
    peak = Math.max(peak, cum);
    maxDd = Math.max(maxDd, peak - cum);
  }
  return -Math.round(maxDd * 100) / 100;
}

function renderToggleButtons() {
  const group = document.getElementById('toggleGroup');
  group.innerHTML = '';
  for (const model of getFilteredModels()) {
    const vk = model.version_key;
    const isActive = model.active !== false;
    const btn = document.createElement('button');
    btn.className = isActive ? 'toggle-btn active' : 'toggle-btn retire-btn';
    btn.id = 'btn_' + vk;
    btn.textContent = model.name;
    btn.style.background = hexToRgba(model.color, 0.25);
    btn.style.borderColor = model.color;
    btn.onclick = () => toggleLayer(vk);
    modelVisibility[vk] = isActive;
    group.appendChild(btn);
  }
  const tblBtn = document.createElement('button');
  tblBtn.className = 'toggle-btn';
  tblBtn.id = 'btnTable';
  tblBtn.textContent = 'Trades';
  tblBtn.style.background = '#363a45';
  tblBtn.onclick = () => toggleTable();
  group.appendChild(tblBtn);
}

function renderLegend() {
  const legend = document.getElementById('legendBar');
  let html = '';
  for (const model of getFilteredModels()) {
    const c = model.color;
    const name = model.name;
    const shape = model.marker_shape === 'circle' ? 'dot' : 'arrow';
    if (shape === 'arrow') {
      html += `<div class="legend-item"><span class="legend-arrow" style="color:${c}">&#8593;</span> ${name} Buy</div>`;
    } else {
      html += `<div class="legend-item"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${c}"></span> ${name} Buy</div>`;
    }
    html += `<div class="legend-item"><span class="legend-arrow" style="color:#4caf50">&#8595;</span> ${name} Win</div>`;
    html += `<div class="legend-item"><span class="legend-arrow" style="color:#f44336">&#8595;</span> ${name} Loss</div>`;
    html += '<div class="legend-sep"></div>';
  }
  const abbrevs = manifest.exit_abbreviations || {};
  const abbrStr = Object.entries(abbrevs).filter(([k,v]) => v).map(([k,v]) => `${v}=${k.replace(/_/g,' ')}`).join(' ');
  html += `<div class="legend-item" style="color:#888">${abbrStr}</div>`;
  legend.innerHTML = html;
}

function maxConsecutive(trades, type) {
  if (!trades || trades.length === 0) return 0;
  let max = 0, cur = 0;
  for (const t of trades) {
    const pnl = Number(t && t.pnl_pct);
    if (!Number.isFinite(pnl)) continue;
    const isWin = pnl >= 0;
    if ((type === 'win' && isWin) || (type === 'loss' && !isWin)) {
      cur++;
      max = Math.max(max, cur);
    } else {
      cur = 0;
    }
  }
  return max;
}

function getModelsForCurrentSymbol() {
  return getFilteredModels().filter(model => {
    const idx = modelIndices[model.version_key] || [];
    return idx.some(entry => entry.symbol === currentSymbol);
  });
}

function renderStats() {
  if (!currentData || !manifest) return;
  const bar = document.getElementById('statsBar');
  const baseOhlcvForPoints = (currentRawData && currentRawData.ohlcv) || currentData.ohlcv || [];
  const allStats = [];
  for (const model of getModelsForCurrentSymbol()) {
    const vk = model.version_key;
    const trades = currentData[vk + '_trades'] || [];
    const s = currentData[vk + '_stats'] || buildStatsFromTrades(trades, vk);
    const isDerivatives = getModelMarketFamily(model) === 'vn_derivatives';
    const pointStats = isDerivatives ? summarizeTradePoints(trades, baseOhlcvForPoints) : null;
    allStats.push({ vk, model, stats: s, trades, pointStats });
  }
  const totals = allStats.map(x => x.stats.total_pnl_pct || 0);
  const wrs = allStats.map(x => x.stats.win_rate || 0);
  const avgs = allStats.map(x => x.stats.avg_pnl_pct || 0);
  const pfs = allStats.map(x => x.stats.pf || 0);
  const maxTotal = Math.max(...totals);
  const maxWr = Math.max(...wrs);
  const maxAvg = Math.max(...avgs);
  const maxPf = Math.max(...pfs);

  let html = '';
  allStats.forEach((item, idx) => {
    const s = item.stats;
    const c = item.model.color;
    const maxDd = Number.isFinite(Number(s.max_drawdown_pct))
      ? Number(s.max_drawdown_pct)
      : maxDrawdownFromTrades(item.trades);
    const bestT = totals[idx] === maxTotal && maxTotal > 0 ? ' <span class="best-label">BEST</span>' : '';
    const bestW = wrs[idx] === maxWr && maxWr > 0 ? ' <span class="best-label">BEST</span>' : '';
    const bestA = avgs[idx] === maxAvg && maxAvg > 0 ? ' <span class="best-label">BEST</span>' : '';
    const bestPf = pfs[idx] === maxPf && maxPf > 0 ? ' <span class="best-label">BEST</span>' : '';
    const maxWin = s.max_win_pct || 0;
    const maxLoss = s.max_loss_pct || 0;
    const avgWin = s.avg_win_pct || 0;
    const avgLoss = s.avg_loss_pct || 0;
    const wins = s.wins || 0;
    const losses = s.losses || 0;
    const maxConsecWin = maxConsecutive(item.trades, 'win');
    const maxConsecLoss = maxConsecutive(item.trades, 'loss');
    const totalPtsRow = item.pointStats
      ? `<div class="stat-row"><span class="lbl">Total (pts):</span><span class="${pnlCls(item.pointStats.total_points)}">${fmt(item.pointStats.total_points)} pt</span></div>`
      : '';
    const maxDdPtsRow = item.pointStats
      ? `<div class="stat-row"><span class="lbl">MaxDD (pts):</span><span class="${pnlCls(item.pointStats.max_drawdown_points)}">${fmt(item.pointStats.max_drawdown_points)} pt</span></div>`
      : '';
    html += `<div class="stat-card" style="background:${hexToRgba(c,0.1)};border:1px solid ${hexToRgba(c,0.3)};transition:opacity 0.2s">
      <div class="stat-title" style="color:${c}">${item.model.name}</div>
      <div class="stat-row"><span class="lbl">Trades:</span><span>${s.total_trades||0} <span style="color:#888;font-size:9px">(${wins}W/${losses}L)</span></span></div>
      <div class="stat-row"><span class="lbl">WR:</span><span>${s.win_rate||0}%${bestW}</span></div>
      <div class="stat-row"><span class="lbl">Avg:</span><span class="${pnlCls(s.avg_pnl_pct)}">${fmt(s.avg_pnl_pct||0)}%${bestA}</span></div>
      <div class="stat-row"><span class="lbl">Total:</span><span class="${pnlCls(s.total_pnl_pct)}">${fmt(s.total_pnl_pct||0)}%${bestT}</span></div>
      ${totalPtsRow}
      <div class="stat-row"><span class="lbl">AvgW/L:</span><span><span class="positive">${fmt(avgWin)}%</span> / <span class="negative">${fmt(avgLoss)}%</span></span></div>
      <div class="stat-row"><span class="lbl">MaxW/L:</span><span><span class="positive">${fmt(maxWin)}%</span> / <span class="negative">${fmt(maxLoss)}%</span></span></div>
      <div class="stat-row"><span class="lbl">Streak:</span><span><span class="positive">${maxConsecWin}W</span> / <span class="negative">${maxConsecLoss}L</span></span></div>
      <div class="stat-row"><span class="lbl">Hold:</span><span>${s.avg_hold||0}d</span></div>
      <div class="stat-row"><span class="lbl">PF:</span><span>${s.pf||0}${bestPf}</span></div>
      <div class="stat-row"><span class="lbl">Median:</span><span class="${pnlCls(s.median_pnl_pct||0)}">${fmt(s.median_pnl_pct||0)}%</span></div>
      <div class="stat-row"><span class="lbl">MaxDD:</span><span class="${pnlCls(maxDd)}">${fmt(maxDd)}%</span></div>
      ${maxDdPtsRow}
    </div>`;
  });
  bar.innerHTML = html;
}

function renderTradePanels() {
  const container = document.getElementById('tablesContainer');
  container.innerHTML = '';
  for (const model of getModelsForCurrentSymbol()) {
    const vk = model.version_key;
    const trades = currentData[vk + '_trades'] || [];
    const panel = document.createElement('div');
    panel.className = 'trade-panel';
    const isRuleBased = model.version_key === 'rule' || model.version_key === 'rule_derivatives';
    const cols = isRuleBased
      ? '<th>#</th><th>Entry</th><th>Exit</th><th>Days</th><th>PnL%</th>'
      : '<th>#</th><th>Entry</th><th>Exit</th><th>Days</th><th>PnL%</th><th>Reason</th><th>Trend</th>';
    panel.innerHTML = `<h3 style="color:${model.color}">${model.name} Trades</h3>
      <table><thead><tr>${cols}</tr></thead><tbody>${
        trades.map((t, i) => {
          const cls = t.pnl_pct >= 0 ? 'positive' : 'negative';
          let row = `<td>${i+1}</td><td>${t.entry_date||''}</td><td>${t.exit_date||''}</td><td>${t.holding_days||0}</td><td class="${cls}">${fmt(t.pnl_pct)}%</td>`;
          if (!isRuleBased) row += `<td>${t.exit_reason||''}</td><td>${t.entry_trend||''}</td>`;
          return `<tr>${row}</tr>`;
        }).join('')
      }</tbody></table>`;
    container.appendChild(panel);
  }
}

function updateStatsVisibility() {
  if (!manifest) return;
  const bar = document.getElementById('statsBar');
  const cards = bar.querySelectorAll('.stat-card');
  getModelsForCurrentSymbol().forEach(function(model, idx) {
    const card = cards[idx];
    if (!card) return;
    // Keep stats cards fully visible even when a model layer is toggled off.
    card.style.opacity = '1';
    var titleEl = card.querySelector('.stat-title');
    if (!titleEl) return;
    var badge = titleEl.querySelector('.hidden-badge');
    if (badge) badge.remove();
  });
}

// ============================================================
// Symbol Selector with Search + Sort by PnL
// ============================================================

function renderSymbolSelector(baseIndex, modelIndices) {
  const filteredModels = getFilteredModels();
  const modelSymbolSet = new Set();
  for (const model of filteredModels) {
    const syms = modelIndices[model.version_key] || [];
    for (const s of syms) modelSymbolSet.add(s.symbol);
  }

  const symbolSet = new Set();
  if (baseIndex && baseIndex.symbols) {
    for (const s of baseIndex.symbols) {
      if (modelSymbolSet.has(s.symbol)) symbolSet.add(s.symbol);
    }
  }
  for (const sym of modelSymbolSet) {
    symbolSet.add(sym);
  }
  const symbolList = [...symbolSet].sort();

  allSymbolItems = symbolList.map(sym => {
    const baseFile = baseIndex && baseIndex.symbols ? baseIndex.symbols.find(s => s.symbol === sym) : null;
    const pnlData = {};
    for (const model of filteredModels) {
      const vk = model.version_key;
      const idx = modelIndices[vk];
      if (!idx) continue;
      const entry = idx.find(s => s.symbol === sym);
      pnlData[vk] = entry ? (entry[vk + '_pnl'] || 0) : 0;
    }
    return {
      symbol: sym,
      file: baseFile ? baseFile.file : sym,
      pnlData: pnlData
    };
  });

  // Populate sort model dropdown
  renderSortModelOptions();

  // Apply initial sort and render
  applySortAndFilter();

  // Setup search input events
  setupSearchEvents();
}

function renderSortModelOptions() {
  const sel = document.getElementById('sortModel');
  if (!sel) return;
  sel.innerHTML = '<option value="">A-Z (Tên)</option>';
  for (const model of getFilteredModels()) {
    const opt = document.createElement('option');
    opt.value = model.version_key;
    opt.textContent = model.name + ' PnL';
    sel.appendChild(opt);
  }
  sel.onchange = function() {
    sortByModel = this.value;
    applySortAndFilter();
  };
}

function applySortAndFilter() {
  const searchInput = document.getElementById('symbolSearchInput');
  const query = searchInput ? searchInput.value.trim().toUpperCase() : '';

  // Filter
  if (query) {
    filteredSymbolItems = allSymbolItems.filter(item =>
      item.symbol.toUpperCase().includes(query)
    );
  } else {
    filteredSymbolItems = [...allSymbolItems];
  }

  // Sort
  if (sortByModel && sortByModel !== '') {
    const vk = sortByModel;
    filteredSymbolItems.sort((a, b) => {
      const pnlA = a.pnlData[vk] || 0;
      const pnlB = b.pnlData[vk] || 0;
      return sortDescending ? (pnlB - pnlA) : (pnlA - pnlB);
    });
  } else {
    filteredSymbolItems.sort((a, b) => a.symbol.localeCompare(b.symbol));
  }

  renderDropdownItems();
  updateSearchInputDisplay();
}

function renderDropdownItems() {
  const dropdown = document.getElementById('symbolDropdown');
  if (!dropdown) return;
  dropdown.innerHTML = '';

  // Count header
  const countDiv = document.createElement('div');
  countDiv.className = 'symbol-dropdown-count';
  countDiv.textContent = `${filteredSymbolItems.length} / ${allSymbolItems.length} mã`;
  if (sortByModel) {
    const modelName = getFilteredModels().find(m => m.version_key === sortByModel);
    countDiv.textContent += ` | Sắp xếp: ${modelName ? modelName.name : sortByModel} PnL ${sortDescending ? '↓' : '↑'}`;
  }
  dropdown.appendChild(countDiv);

  // Items
  for (let i = 0; i < filteredSymbolItems.length; i++) {
    const item = filteredSymbolItems[i];
    const div = document.createElement('div');
    div.className = 'symbol-dropdown-item';
    if (currentSymbol === item.symbol) div.classList.add('selected');
    if (i === highlightedIndex) div.classList.add('highlighted');
    div.dataset.index = i;
    div.dataset.file = item.file;
    div.dataset.symbol = item.symbol;

    // Symbol name
    const nameSpan = document.createElement('span');
    nameSpan.className = 'sym-name';
    nameSpan.textContent = item.symbol;
    div.appendChild(nameSpan);

    // PnL info for each model
    const pnlSpan = document.createElement('span');
    pnlSpan.className = 'sym-pnl';
    let pnlParts = [];
    for (const model of getFilteredModels()) {
      const vk = model.version_key;
      const pnl = item.pnlData[vk] || 0;
      const cls = pnl >= 0 ? 'positive' : 'negative';
      const label = vk.toUpperCase().replace('_', '.');
      pnlParts.push(`<span class="${cls}">${label}:${fmt(pnl)}%</span>`);
    }
    pnlSpan.innerHTML = pnlParts.join(' ');
    div.appendChild(pnlSpan);

    div.onclick = function() {
      selectSymbolItem(item);
    };
    dropdown.appendChild(div);
  }
}

function selectSymbolItem(item) {
  const searchInput = document.getElementById('symbolSearchInput');
  const dropdown = document.getElementById('symbolDropdown');
  searchInput.value = item.symbol;
  dropdown.classList.remove('open');
  highlightedIndex = -1;
  loadSymbol(item.file);
}

function updateSearchInputDisplay() {
  const searchInput = document.getElementById('symbolSearchInput');
  if (!searchInput) return;
  if (currentSymbol && !searchInput.matches(':focus')) {
    searchInput.value = currentSymbol;
  }
}

function setupSearchEvents() {
  const searchInput = document.getElementById('symbolSearchInput');
  const dropdown = document.getElementById('symbolDropdown');
  if (!searchInput || !dropdown) return;

  // Focus → open dropdown, clear input to show all symbols
  searchInput.addEventListener('focus', function() {
    this.value = '';
    highlightedIndex = -1;
    applySortAndFilter();
    dropdown.classList.add('open');
  });

  // Input → filter
  searchInput.addEventListener('input', function() {
    highlightedIndex = -1;
    applySortAndFilter();
    dropdown.classList.add('open');
  });

  // Keyboard navigation
  searchInput.addEventListener('keydown', function(e) {
    if (!dropdown.classList.contains('open')) {
      if (e.key === 'ArrowDown' || e.key === 'Enter') {
        dropdown.classList.add('open');
        applySortAndFilter();
        return;
      }
    }

    const items = filteredSymbolItems;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      highlightedIndex = Math.min(highlightedIndex + 1, items.length - 1);
      renderDropdownItems();
      scrollToHighlighted();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      highlightedIndex = Math.max(highlightedIndex - 1, 0);
      renderDropdownItems();
      scrollToHighlighted();
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (highlightedIndex >= 0 && highlightedIndex < items.length) {
        selectSymbolItem(items[highlightedIndex]);
      } else if (items.length === 1) {
        selectSymbolItem(items[0]);
      } else if (items.length > 0) {
        // Try exact match first
        const query = searchInput.value.trim().toUpperCase();
        const exact = items.find(it => it.symbol === query);
        if (exact) selectSymbolItem(exact);
        else selectSymbolItem(items[0]);
      }
    } else if (e.key === 'Escape') {
      dropdown.classList.remove('open');
      highlightedIndex = -1;
      searchInput.blur();
      updateSearchInputDisplay();
    }
  });

  // Click outside → close dropdown
  document.addEventListener('click', function(e) {
    if (!e.target.closest('.symbol-search-container') && !e.target.closest('.sort-controls')) {
      dropdown.classList.remove('open');
      highlightedIndex = -1;
      updateSearchInputDisplay();
    }
  });
}

function scrollToHighlighted() {
  const dropdown = document.getElementById('symbolDropdown');
  if (!dropdown) return;
  const highlighted = dropdown.querySelector('.highlighted');
  if (highlighted) {
    highlighted.scrollIntoView({ block: 'nearest' });
  }
}

// Sort direction toggle
window.toggleSortDir = function() {
  sortDescending = !sortDescending;
  const btn = document.getElementById('sortDirBtn');
  if (btn) {
    btn.textContent = sortDescending ? '↓ Cao→Thấp' : '↑ Thấp→Cao';
    btn.classList.toggle('active', !sortDescending);
  }
  applySortAndFilter();
  // Re-open dropdown if search is focused
  const dropdown = document.getElementById('symbolDropdown');
  const searchInput = document.getElementById('symbolSearchInput');
  if (searchInput && searchInput === document.activeElement) {
    dropdown.classList.add('open');
  }
};

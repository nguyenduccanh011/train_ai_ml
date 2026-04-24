// UI rendering functions — all dynamic from manifest
function pnlCls(v) { return v >= 0 ? 'positive' : 'negative'; }
function fmt(v) { return v > 0 ? '+' + v : '' + v; }

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1,3), 16);
  const g = parseInt(hex.slice(3,5), 16);
  const b = parseInt(hex.slice(5,7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
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
  for (const model of manifest.models) {
    const vk = model.version_key;
    const btn = document.createElement('button');
    btn.className = 'toggle-btn active';
    btn.id = 'btn_' + vk;
    btn.textContent = model.name;
    btn.style.background = hexToRgba(model.color, 0.25);
    btn.style.borderColor = model.color;
    btn.onclick = () => toggleLayer(vk);
    modelVisibility[vk] = true;
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
  for (const model of manifest.models) {
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

function renderStats() {
  if (!currentData || !manifest) return;
  const bar = document.getElementById('statsBar');
  const allStats = [];
  for (const model of manifest.models) {
    const vk = model.version_key;
    const s = currentData[vk + '_stats'] || {};
    allStats.push({ vk, model, stats: s, trades: currentData[vk + '_trades'] || [] });
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
    html += `<div class="stat-card" style="background:${hexToRgba(c,0.1)};border:1px solid ${hexToRgba(c,0.3)};transition:opacity 0.2s">
      <div class="stat-title" style="color:${c}">${item.model.name}</div>
      <div class="stat-row"><span class="lbl">Trades:</span><span>${s.total_trades||0} <span style="color:#888;font-size:9px">(${wins}W/${losses}L)</span></span></div>
      <div class="stat-row"><span class="lbl">WR:</span><span>${s.win_rate||0}%${bestW}</span></div>
      <div class="stat-row"><span class="lbl">Avg:</span><span class="${pnlCls(s.avg_pnl_pct)}">${fmt(s.avg_pnl_pct||0)}%${bestA}</span></div>
      <div class="stat-row"><span class="lbl">Total:</span><span class="${pnlCls(s.total_pnl_pct)}">${fmt(s.total_pnl_pct||0)}%${bestT}</span></div>
      <div class="stat-row"><span class="lbl">AvgW/L:</span><span><span class="positive">${fmt(avgWin)}%</span> / <span class="negative">${fmt(avgLoss)}%</span></span></div>
      <div class="stat-row"><span class="lbl">MaxW/L:</span><span><span class="positive">${fmt(maxWin)}%</span> / <span class="negative">${fmt(maxLoss)}%</span></span></div>
      <div class="stat-row"><span class="lbl">Streak:</span><span><span class="positive">${maxConsecWin}W</span> / <span class="negative">${maxConsecLoss}L</span></span></div>
      <div class="stat-row"><span class="lbl">Hold:</span><span>${s.avg_hold||0}d</span></div>
      <div class="stat-row"><span class="lbl">PF:</span><span>${s.pf||0}${bestPf}</span></div>
      <div class="stat-row"><span class="lbl">Median:</span><span class="${pnlCls(s.median_pnl_pct||0)}">${fmt(s.median_pnl_pct||0)}%</span></div>
      <div class="stat-row"><span class="lbl">MaxDD:</span><span class="${pnlCls(maxDd)}">${fmt(maxDd)}%</span></div>
    </div>`;
  });
  bar.innerHTML = html;
}

function renderTradePanels() {
  const container = document.getElementById('tablesContainer');
  container.innerHTML = '';
  for (const model of manifest.models) {
    const vk = model.version_key;
    const trades = currentData[vk + '_trades'] || [];
    const panel = document.createElement('div');
    panel.className = 'trade-panel';
    const isRule = model.version_key === 'rule';
    const cols = isRule
      ? '<th>#</th><th>Entry</th><th>Exit</th><th>Days</th><th>PnL%</th>'
      : '<th>#</th><th>Entry</th><th>Exit</th><th>Days</th><th>PnL%</th><th>Reason</th><th>Trend</th>';
    panel.innerHTML = `<h3 style="color:${model.color}">${model.name} Trades</h3>
      <table><thead><tr>${cols}</tr></thead><tbody>${
        trades.map((t, i) => {
          const cls = t.pnl_pct >= 0 ? 'positive' : 'negative';
          let row = `<td>${i+1}</td><td>${t.entry_date||''}</td><td>${t.exit_date||''}</td><td>${t.holding_days||0}</td><td class="${cls}">${fmt(t.pnl_pct)}%</td>`;
          if (!isRule) row += `<td>${t.exit_reason||''}</td><td>${t.entry_trend||''}</td>`;
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
  manifest.models.forEach(function(model, idx) {
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
  // Build set of symbols that appear in at least one model's data
  const modelSymbolSet = new Set();
  for (const syms of Object.values(modelIndices)) {
    for (const s of syms) modelSymbolSet.add(s.symbol);
  }

  // Filter base index to only symbols with model data
  const symbolSet = new Set();
  if (baseIndex && baseIndex.symbols) {
    for (const s of baseIndex.symbols) {
      if (modelSymbolSet.has(s.symbol)) symbolSet.add(s.symbol);
    }
  }
  const symbolList = [...symbolSet].sort();

  allSymbolItems = symbolList.map(sym => {
    const file = baseIndex.symbols.find(s => s.symbol === sym);
    const pnlData = {};
    for (const model of manifest.models) {
      const vk = model.version_key;
      const idx = modelIndices[vk];
      if (!idx) continue;
      const entry = idx.find(s => s.symbol === sym);
      pnlData[vk] = entry ? (entry[vk + '_pnl'] || 0) : 0;
    }
    return {
      symbol: sym,
      file: file ? file.file : sym,
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
  for (const model of manifest.models) {
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
  countDiv.textContent = `${filteredSymbolItems.length} / ${allSymbolItems.length} mã cổ phiếu`;
  if (sortByModel) {
    const modelName = manifest.models.find(m => m.version_key === sortByModel);
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
    for (const model of manifest.models) {
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

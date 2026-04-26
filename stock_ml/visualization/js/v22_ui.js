// V22 UI rendering helpers and panels
function v22Num(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function v22PnlCls(value) {
  return v22Num(value) >= 0 ? 'positive' : 'negative';
}

function v22Fmt(value, digits = 1) {
  const n = v22Num(value);
  const s = n.toFixed(digits);
  return n > 0 ? '+' + s : s;
}

function v22Pct(value, digits = 1) {
  return v22Fmt(value, digits) + '%';
}

function v22MaxDrawdownFromTrades(trades) {
  let cum = 0;
  let peak = 0;
  let maxDd = 0;
  for (const trade of trades || []) {
    const pnl = v22Num(trade.pnl_pct, NaN);
    if (!Number.isFinite(pnl)) continue;
    cum += pnl;
    peak = Math.max(peak, cum);
    maxDd = Math.max(maxDd, peak - cum);
  }
  return -Math.round(maxDd * 100) / 100;
}

function v22DaysBetween(a, b) {
  if (!a || !b) return null;
  const d1 = new Date(a + 'T00:00:00');
  const d2 = new Date(b + 'T00:00:00');
  if (Number.isNaN(d1.getTime()) || Number.isNaN(d2.getTime())) return null;
  return Math.round((d2 - d1) / 86400000);
}

function renderV22Summary(payload) {
  const el = document.getElementById('v22Summary');
  const prefix = v22CurrentMode.prefix;
  const trades = Array.isArray(payload && payload[`${prefix}_trades`]) ? payload[`${prefix}_trades`] : [];
  const stats = payload && payload[`${prefix}_stats`] ? payload[`${prefix}_stats`] : {};
  const wins = stats.wins ?? trades.filter(t => v22Num(t.pnl_pct) >= 0).length;
  const losses = stats.losses ?? trades.filter(t => v22Num(t.pnl_pct) < 0).length;
  const totalTrades = stats.total_trades ?? trades.length;
  const totalPnl = stats.total_pnl_pct ?? trades.reduce((sum, t) => sum + v22Num(t.pnl_pct), 0);
  const avgPnl = stats.avg_pnl_pct ?? (totalTrades ? totalPnl / totalTrades : 0);
  const maxDd = Number.isFinite(Number(stats.max_drawdown_pct)) ? Number(stats.max_drawdown_pct) : v22MaxDrawdownFromTrades(trades);

  const cards = [
    ['Trades', totalTrades, `${wins}W / ${losses}L`, 'neutral', 0],
    ['Win rate', `${v22Num(stats.win_rate).toFixed(1)}%`, 'Tỉ lệ lệnh thắng', 'neutral', 0],
    ['Total PnL', v22Pct(totalPnl), 'Tổng PnL theo trades', v22PnlCls(totalPnl), 0],
    ['Avg PnL', v22Pct(avgPnl), 'Trung bình mỗi trade', v22PnlCls(avgPnl), 0],
    ['Profit factor', v22Num(stats.pf).toFixed(2), 'Gross win / gross loss', 'neutral', 0],
    ['Avg hold', `${v22Num(stats.avg_hold).toFixed(1)}d`, 'Số ngày giữ TB', 'neutral', 0],
    ['Max win/loss', `${v22Pct(stats.max_win_pct)} / ${v22Pct(stats.max_loss_pct)}`, 'Biên trade lớn nhất', 'neutral', 0],
    ['Max DD', v22Pct(maxDd), 'Tính từ equity trade', v22PnlCls(maxDd), 0],
  ];

  el.innerHTML = cards.map(([label, value, sub, cls]) => `
    <div class="summary-card">
      <div class="label">${label}</div>
      <div class="value ${cls}">${value}</div>
      <div class="sub">${sub}</div>
    </div>
  `).join('');
}

function renderV22Trades(payload) {
  const el = document.getElementById('v22Trades');
  const prefix = v22CurrentMode.prefix;
  const trades = Array.isArray(payload && payload[`${prefix}_trades`]) ? payload[`${prefix}_trades`] : [];
  if (!trades.length) {
    el.innerHTML = '<div class="empty">Không có trade V22 cho mã này.</div>';
    return;
  }

  el.innerHTML = `<table>
    <thead><tr>
      <th>#</th><th>Entry</th><th>Exit</th><th>Days</th><th>PnL</th><th>Reason</th><th>Trend</th><th>Profile</th><th>Size</th><th>Flags</th>
    </tr></thead>
    <tbody>${trades.map((trade, idx) => {
      const flags = [];
      if (trade.breakout_entry) flags.push('BO');
      if (trade.vshape_entry) flags.push('V');
      if (trade.quick_reentry) flags.push('QR');
      if (trade.entry_choppy_regime) flags.push('Chop');
      return `<tr>
        <td>${idx + 1}</td>
        <td>${trade.entry_date || ''}</td>
        <td>${trade.exit_date || ''}</td>
        <td>${trade.holding_days ?? ''}</td>
        <td class="${v22PnlCls(trade.pnl_pct)}">${v22Pct(trade.pnl_pct)}</td>
        <td><span class="pill">${trade.exit_reason || ''}</span></td>
        <td>${trade.entry_trend || ''}</td>
        <td>${trade.entry_profile || ''}</td>
        <td>${v22Num(trade.position_size).toFixed(2)}</td>
        <td>${flags.map(f => `<span class="pill">${f}</span>`).join(' ')}</td>
      </tr>`;
    }).join('')}</tbody>
  </table>`;
}

function renderV22Timeline(payload) {
  const el = document.getElementById('v22Timeline');
  const prefix = v22CurrentMode.prefix;
  const markers = Array.isArray(payload && payload[`${prefix}_markers`]) ? payload[`${prefix}_markers`] : [];
  if (!markers.length) {
    el.innerHTML = '<div class="empty">Không có marker/tín hiệu V22 cho mã này.</div>';
    return;
  }
  el.innerHTML = markers.slice().sort((a, b) => a.time < b.time ? -1 : a.time > b.time ? 1 : 0).map(marker => {
    const kind = v22MarkerKind(marker);
    const label = kind === 'buy' ? 'BUY' : (kind === 'loss' ? 'LOSS' : 'WIN');
    return `<div class="timeline-item">
      <div>${marker.time || ''}</div>
      <div><span class="marker-dot" style="background:${marker.color || '#888'}"></span> ${label}</div>
      <div>${marker.text || ''}</div>
    </div>`;
  }).join('');
}

function renderV22Breakdown(payload) {
  const el = document.getElementById('v22Breakdown');
  const prefix = v22CurrentMode.prefix;
  const trades = Array.isArray(payload && payload[`${prefix}_trades`]) ? payload[`${prefix}_trades`] : [];
  if (!trades.length) {
    el.innerHTML = '<div class="empty">Chưa có trade để phân tích exit reason.</div>';
    return;
  }

  const groups = new Map();
  for (const trade of trades) {
    const reason = trade.exit_reason || 'unknown';
    if (!groups.has(reason)) groups.set(reason, { count: 0, wins: 0, pnl: 0 });
    const g = groups.get(reason);
    const pnl = v22Num(trade.pnl_pct);
    g.count += 1;
    g.pnl += pnl;
    if (pnl >= 0) g.wins += 1;
  }
  const rows = [...groups.entries()].sort((a, b) => b[1].count - a[1].count);
  const maxCount = Math.max(...rows.map(([, g]) => g.count));
  el.innerHTML = rows.map(([reason, g]) => {
    const wr = g.count ? g.wins / g.count * 100 : 0;
    const avg = g.count ? g.pnl / g.count : 0;
    return `<div>
      <div class="break-row">
        <span>${reason}</span>
        <span>${g.count}</span>
        <span>${wr.toFixed(0)}%</span>
        <span class="${v22PnlCls(avg)}">${v22Pct(avg)}</span>
      </div>
      <div class="bar-shell"><div class="bar-fill" style="width:${maxCount ? g.count / maxCount * 100 : 0}%"></div></div>
    </div>`;
  }).join('');
}

function renderV22Latest(baseData, payload) {
  const el = document.getElementById('v22Latest');
  const ohlcv = Array.isArray(baseData && baseData.ohlcv) ? baseData.ohlcv : [];
  const prefix = v22CurrentMode.prefix;
  const markers = Array.isArray(payload && payload[`${prefix}_markers`]) ? payload[`${prefix}_markers`] : [];
  const trades = Array.isArray(payload && payload[`${prefix}_trades`]) ? payload[`${prefix}_trades`] : [];
  const lastBar = ohlcv[ohlcv.length - 1] || {};
  const lastMarker = markers.slice().sort((a, b) => a.time < b.time ? 1 : a.time > b.time ? -1 : 0)[0] || {};
  const lastTrade = trades.slice().sort((a, b) => String(a.exit_date || a.entry_date) < String(b.exit_date || b.entry_date) ? 1 : -1)[0] || {};
  const daysSince = v22DaysBetween(lastMarker.time, lastBar.time);

  el.innerHTML = `
    <div class="kv-row"><span class="k">Mã</span><span class="v">${payload.symbol || v22CurrentSymbol || ''}</span></div>
    <div class="kv-row"><span class="k">Ngày dữ liệu cuối</span><span class="v">${lastBar.time || ''}</span></div>
    <div class="kv-row"><span class="k">Close cuối</span><span class="v">${lastBar.close ?? ''}</span></div>
    <div class="kv-row"><span class="k">Tín hiệu gần nhất</span><span class="v">${lastMarker.time || ''} · ${lastMarker.text || ''}</span></div>
    <div class="kv-row"><span class="k">Cách hiện tại</span><span class="v">${daysSince == null ? '' : daysSince + ' ngày'}</span></div>
    <div class="kv-row"><span class="k">Trade gần nhất</span><span class="v ${v22PnlCls(lastTrade.pnl_pct)}">${lastTrade.entry_date || ''} → ${lastTrade.exit_date || ''} · ${v22Pct(lastTrade.pnl_pct)}</span></div>
  `;
}

function renderV22SymbolDropdown() {
  const dropdown = document.getElementById('v22SymbolDropdown');
  const items = v22FilteredSymbols.slice(0, 220);
  dropdown.innerHTML = `<div class="symbol-dropdown-count">${v22FilteredSymbols.length} mã · ${v22CurrentMode.label} · hiển thị ${items.length}</div>` + items.map(item => `
    <div class="symbol-dropdown-item ${item.symbol === v22CurrentSymbol ? 'selected' : ''}" data-symbol="${item.symbol}">
      <span class="sym-name">${item.symbol}</span>
      <span class="${v22PnlCls(item.pnl)}">${v22Pct(item.pnl)}</span>
      <span class="sym-muted">WR ${v22Num(item.wr).toFixed(1)}%</span>
      <span class="sym-muted">${item.trades} trades</span>
    </div>
  `).join('');

  dropdown.querySelectorAll('.symbol-dropdown-item').forEach(item => {
    item.addEventListener('mousedown', event => {
      event.preventDefault();
      loadV22Symbol(item.dataset.symbol);
      dropdown.classList.remove('open');
    });
  });
}

function applyV22SymbolFilter() {
  const input = document.getElementById('v22SymbolSearch');
  const sortBy = document.getElementById('v22SortBy').value;
  const query = input.value.trim().toUpperCase();
  v22FilteredSymbols = v22AllSymbols.filter(item => item.symbol.includes(query));
  v22FilteredSymbols.sort((a, b) => {
    if (sortBy === 'symbol') return a.symbol.localeCompare(b.symbol);
    if (sortBy === 'wr') return v22Num(b.wr) - v22Num(a.wr) || a.symbol.localeCompare(b.symbol);
    if (sortBy === 'trades') return v22Num(b.trades) - v22Num(a.trades) || a.symbol.localeCompare(b.symbol);
    return v22Num(b.pnl) - v22Num(a.pnl) || a.symbol.localeCompare(b.symbol);
  });
  renderV22SymbolDropdown();
}

function renderV22All(baseData, payload) {
  document.getElementById('v22ChartTitle').textContent = `V22 · ${v22CurrentMode.label} · ${payload.symbol || v22CurrentSymbol}`;
  renderV22Summary(payload);
  renderV22Trades(payload);
  renderV22Timeline(payload);
  renderV22Breakdown(payload);
  renderV22Latest(baseData, payload);
}

function setV22Status(message, isError = false) {
  const el = document.getElementById('v22Status');
  el.textContent = message;
  el.className = isError ? 'status-line negative' : 'status-line';
}

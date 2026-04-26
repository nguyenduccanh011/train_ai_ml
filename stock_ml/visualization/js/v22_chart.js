// V22 chart creation and marker handling
function createV22Chart() {
  const container = document.getElementById('v22Chart');
  container.innerHTML = '';

  v22Chart = LightweightCharts.createChart(container, {
    width: container.clientWidth,
    height: container.clientHeight,
    layout: { background: { color: '#131722' }, textColor: '#d1d4dc' },
    grid: { vertLines: { color: '#1e222d' }, horzLines: { color: '#1e222d' } },
    crosshair: { mode: 0 },
    timeScale: { timeVisible: false, borderColor: '#2B2B43' },
    rightPriceScale: { borderColor: '#2B2B43' },
  });

  v22CandleSeries = v22Chart.addCandlestickSeries({
    upColor: '#26a69a',
    downColor: '#ef5350',
    borderUpColor: '#26a69a',
    borderDownColor: '#ef5350',
    wickUpColor: '#26a69a',
    wickDownColor: '#ef5350',
  });

  v22VolumeSeries = v22Chart.addHistogramSeries({
    priceFormat: { type: 'volume' },
    priceScaleId: 'vol',
    scaleMargins: { top: 0.85, bottom: 0 },
  });
  v22Chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });

  window.addEventListener('resize', resizeV22Chart);
}

function resizeV22Chart() {
  const container = document.getElementById('v22Chart');
  if (v22Chart && container) {
    v22Chart.resize(container.clientWidth, container.clientHeight);
  }
}

function setV22ChartData(baseData, payload) {
  if (!v22Chart) createV22Chart();
  const ohlcv = Array.isArray(baseData && baseData.ohlcv) ? baseData.ohlcv : [];
  v22CandleSeries.setData(ohlcv);
  v22VolumeSeries.setData(ohlcv.map(bar => ({
    time: bar.time,
    value: bar.volume || 0,
    color: Number(bar.close) >= Number(bar.open) ? '#26a69a55' : '#ef535055',
  })));
  v22CandleSeries.setMarkers(getV22Markers(payload));
  v22Chart.timeScale().fitContent();
}

function getV22Markers(payload) {
  const key = `${v22CurrentMode.prefix}_markers`;
  const markers = Array.isArray(payload && payload[key]) ? payload[key] : [];
  return markers
    .filter(marker => v22ShouldShowMarker(marker))
    .slice()
    .sort((a, b) => a.time < b.time ? -1 : a.time > b.time ? 1 : 0);
}

function v22MarkerKind(marker) {
  const text = String(marker && marker.text || '').toLowerCase();
  const color = String(marker && marker.color || '').toLowerCase();
  const position = String(marker && marker.position || '').toLowerCase();
  const shape = String(marker && marker.shape || '').toLowerCase();

  if (position === 'belowbar' || shape === 'arrowup' || text.includes('buy')) return 'buy';
  if (text.includes('-') || color.includes('f44336') || color.includes('ef5350')) return 'loss';
  if (position === 'abovebar' || shape === 'arrowdown') return 'win';
  return 'win';
}

function v22ShouldShowMarker(marker) {
  const kind = v22MarkerKind(marker);
  return v22MarkerFilters[kind] !== false;
}

function refreshV22Markers() {
  if (!v22CandleSeries || !v22CurrentPayload) return;
  v22CandleSeries.setMarkers(getV22Markers(v22CurrentPayload));
}

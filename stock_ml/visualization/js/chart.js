// Chart creation and management
function createChart() {
  const c = document.getElementById('chart');
  c.innerHTML = '';
  chart = LightweightCharts.createChart(c, {
    width: c.clientWidth, height: c.clientHeight,
    layout: { background: { color: '#131722' }, textColor: '#d1d4dc' },
    grid: { vertLines: { color: '#1e222d' }, horzLines: { color: '#1e222d' } },
    crosshair: { mode: 0 },
    timeScale: { timeVisible: false, borderColor: '#2B2B43' },
    rightPriceScale: { borderColor: '#2B2B43' },
  });
  candleSeries = chart.addCandlestickSeries({
    upColor: '#26a69a', downColor: '#ef5350',
    borderUpColor: '#26a69a', borderDownColor: '#ef5350',
    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
  });
  volumeSeries = chart.addHistogramSeries({
    priceFormat: { type: 'volume' }, priceScaleId: 'vol',
    scaleMargins: { top: 0.85, bottom: 0 },
  });
  chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });
  window.addEventListener('resize', () => {
    const c = document.getElementById('chart');
    if (chart) chart.resize(c.clientWidth, c.clientHeight);
  });
}

function getMarkers() {
  let m = [];
  if (!currentData || !manifest) return m;
  for (const model of manifest.models) {
    const vk = model.version_key;
    if (!modelVisibility[vk]) continue;
    const key = vk + '_markers';
    if (currentData[key]) m = m.concat(currentData[key]);
  }
  m.sort((a, b) => a.time < b.time ? -1 : a.time > b.time ? 1 : 0);
  return m;
}

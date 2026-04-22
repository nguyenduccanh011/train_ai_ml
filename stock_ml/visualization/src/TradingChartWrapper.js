/**
 * TradingChartWrapper - Wrapper for lightweight-charts v5.x
 * Supports OHLCV candlestick charts with series markers (buy/sell signals, custom markers)
 */

export const MarkerShape = {
  ARROW_UP: 'arrowUp',
  ARROW_DOWN: 'arrowDown',
  CIRCLE: 'circle',
  SQUARE: 'square',
};

export const MarkerPosition = {
  ABOVE: 'aboveBar',
  BELOW: 'belowBar',
  IN_BAR: 'inBar',
};

export const MarkerPresets = {
  buy: {
    position: MarkerPosition.BELOW,
    color: '#2196F3',
    shape: MarkerShape.ARROW_UP,
    text: 'Buy',
  },
  sell: {
    position: MarkerPosition.ABOVE,
    color: '#e91e63',
    shape: MarkerShape.ARROW_DOWN,
    text: 'Sell',
  },
  stopLoss: {
    position: MarkerPosition.BELOW,
    color: '#ff5252',
    shape: MarkerShape.CIRCLE,
    text: 'SL',
  },
  takeProfit: {
    position: MarkerPosition.ABOVE,
    color: '#4caf50',
    shape: MarkerShape.CIRCLE,
    text: 'TP',
  },
  trailing_stop: {
    position: MarkerPosition.ABOVE,
    color: '#ff9800',
    shape: MarkerShape.ARROW_DOWN,
    text: 'Trail',
  },
  zombie_exit: {
    position: MarkerPosition.ABOVE,
    color: '#9e9e9e',
    shape: MarkerShape.SQUARE,
    text: 'Zombie',
  },
  signal_exit: {
    position: MarkerPosition.ABOVE,
    color: '#e91e63',
    shape: MarkerShape.ARROW_DOWN,
    text: 'Exit',
  },
};

export class TradingChartWrapper {
  constructor(container, options = {}) {
    this._container = container;
    this._markers = [];
    this._chart = null;
    this._candleSeries = null;
    this._volumeSeries = null;
    this._seriesMarkers = null;
    this._options = options;
    this._lwc = null;
  }

  init(lwc) {
    this._lwc = lwc;
    const chartOpts = {
      width: this._container.clientWidth,
      height: this._container.clientHeight || 500,
      layout: {
        background: { type: 'solid', color: '#1e222d' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: '#2B2B43' },
        horzLines: { color: '#2B2B43' },
      },
      crosshair: { mode: 0 },
      rightPriceScale: { borderColor: '#2B2B43' },
      timeScale: {
        borderColor: '#2B2B43',
        timeVisible: true,
        secondsVisible: false,
      },
      ...this._options.chart,
    };

    this._chart = lwc.createChart(this._container, chartOpts);
    this._candleSeries = this._chart.addSeries(lwc.CandlestickSeries, {
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderDownColor: '#ef5350',
      borderUpColor: '#26a69a',
      wickDownColor: '#ef5350',
      wickUpColor: '#26a69a',
      ...this._options.candlestick,
    });

    if (this._options.volume !== false) {
      this._volumeSeries = this._chart.addSeries(lwc.HistogramSeries, {
        priceFormat: { type: 'volume' },
        priceScaleId: '',
        ...this._options.volume,
      });
      this._volumeSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      });
    }

    this._resizeObserver = new ResizeObserver(() => {
      this._chart.applyOptions({
        width: this._container.clientWidth,
        height: this._container.clientHeight,
      });
    });
    this._resizeObserver.observe(this._container);
    return this;
  }

  setData(data) {
    const ohlc = data.map(({ time, open, high, low, close }) => ({ time, open, high, low, close }));
    this._candleSeries.setData(ohlc);

    if (this._volumeSeries && data.length > 0 && data[0].volume !== undefined) {
      const vol = data.map(d => ({
        time: d.time,
        value: d.volume,
        color: d.close >= d.open ? 'rgba(38,166,154,0.5)' : 'rgba(239,83,80,0.5)',
      }));
      this._volumeSeries.setData(vol);
    }
    this._applyMarkers();
    return this;
  }

  addMarker(type, time, text) {
    const preset = MarkerPresets[type];
    if (!preset) throw new Error(`Unknown marker type: ${type}`);
    this._markers.push({
      time, position: preset.position, color: preset.color,
      shape: preset.shape, text: text || preset.text,
    });
    this._applyMarkers();
    return this;
  }

  addCustomMarker(marker) {
    this._markers.push(marker);
    this._applyMarkers();
    return this;
  }

  setMarkers(markers) {
    this._markers = markers;
    this._applyMarkers();
    return this;
  }

  clearMarkers() {
    this._markers = [];
    this._applyMarkers();
    return this;
  }

  _applyMarkers() {
    if (!this._lwc || !this._candleSeries) return;
    const sorted = [...this._markers].sort((a, b) => {
      if (typeof a.time === 'string' && typeof b.time === 'string') return a.time.localeCompare(b.time);
      return a.time - b.time;
    });
    if (this._seriesMarkers) {
      this._seriesMarkers.detach();
      this._seriesMarkers = null;
    }
    if (sorted.length > 0) {
      this._seriesMarkers = this._lwc.createSeriesMarkers(this._candleSeries, sorted);
    }
  }

  getChart() { return this._chart; }
  getCandleSeries() { return this._candleSeries; }
  getVolumeSeries() { return this._volumeSeries; }

  fitContent() {
    this._chart.timeScale().fitContent();
    return this;
  }

  destroy() {
    if (this._resizeObserver) this._resizeObserver.disconnect();
    if (this._seriesMarkers) this._seriesMarkers.detach();
    if (this._chart) this._chart.remove();
    this._chart = null;
    this._candleSeries = null;
    this._volumeSeries = null;
    this._seriesMarkers = null;
  }
}

export default TradingChartWrapper;

/**
 * TradingChartWrapper - Wrapper for lightweight-charts v5.x
 * Supports OHLCV candlestick charts with series markers (buy/sell signals, custom markers)
 * 
 * @see https://tradingview.github.io/lightweight-charts/tutorials/how_to/series-markers
 */

// Marker shape types
export const MarkerShape = {
  ARROW_UP: 'arrowUp',
  ARROW_DOWN: 'arrowDown',
  CIRCLE: 'circle',
  SQUARE: 'square',
};

// Marker position types
export const MarkerPosition = {
  ABOVE: 'aboveBar',
  BELOW: 'belowBar',
  IN_BAR: 'inBar',
};

// Preset marker configs
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
  info: {
    position: MarkerPosition.ABOVE,
    color: '#ff9800',
    shape: MarkerShape.SQUARE,
    text: 'i',
  },
  alert: {
    position: MarkerPosition.ABOVE,
    color: '#ff5722',
    shape: MarkerShape.SQUARE,
    text: '⚠',
  },
};

export class TradingChartWrapper {
  /**
   * @param {HTMLElement} container - DOM element to render chart into
   * @param {object} [options] - Chart options
   * @param {object} [options.chart] - lightweight-charts createChart options
   * @param {object} [options.candlestick] - Candlestick series options
   * @param {object} [options.volume] - Volume histogram options (set false to disable)
   */
  constructor(container, options = {}) {
    this._container = container;
    this._markers = [];
    this._chart = null;
    this._candleSeries = null;
    this._volumeSeries = null;
    this._seriesMarkers = null; // v5: createSeriesMarkers instance
    this._options = options;
    this._lwc = null;
  }

  /**
   * Initialize the chart. Must be called after construction.
   * @param {object} lwc - The lightweight-charts module (window.LightweightCharts or import)
   */
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

    // v5: use addSeries with series type descriptor
    this._candleSeries = this._chart.addSeries(lwc.CandlestickSeries, {
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderDownColor: '#ef5350',
      borderUpColor: '#26a69a',
      wickDownColor: '#ef5350',
      wickUpColor: '#26a69a',
      ...this._options.candlestick,
    });

    // Volume series (optional)
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

    // Auto-resize
    this._resizeObserver = new ResizeObserver(() => {
      this._chart.applyOptions({
        width: this._container.clientWidth,
        height: this._container.clientHeight,
      });
    });
    this._resizeObserver.observe(this._container);

    return this;
  }

  /**
   * Set OHLCV data
   * @param {Array<{time, open, high, low, close, volume?}>} data
   */
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

    // Re-apply markers after data change
    this._applyMarkers();
    return this;
  }

  /**
   * Add a single marker using a preset type
   * @param {'buy'|'sell'|'stopLoss'|'takeProfit'|'info'|'alert'} type
   * @param {string|number} time - e.g. '2024-01-15' or unix timestamp
   * @param {string} [text] - Override default text
   */
  addMarker(type, time, text) {
    const preset = MarkerPresets[type];
    if (!preset) throw new Error(`Unknown marker type: ${type}. Available: ${Object.keys(MarkerPresets).join(', ')}`);

    this._markers.push({
      time,
      position: preset.position,
      color: preset.color,
      shape: preset.shape,
      text: text || preset.text,
    });
    this._applyMarkers();
    return this;
  }

  /**
   * Add a fully custom marker
   * @param {object} marker - { time, position, color, shape, text, size? }
   */
  addCustomMarker(marker) {
    this._markers.push(marker);
    this._applyMarkers();
    return this;
  }

  /**
   * Set all markers at once (replaces existing)
   * @param {Array} markers
   */
  setMarkers(markers) {
    this._markers = markers;
    this._applyMarkers();
    return this;
  }

  /**
   * Clear all markers
   */
  clearMarkers() {
    this._markers = [];
    this._applyMarkers();
    return this;
  }

  /** Sort and apply markers to the candle series using v5 createSeriesMarkers API */
  _applyMarkers() {
    if (!this._lwc || !this._candleSeries) return;

    const sorted = [...this._markers].sort((a, b) => {
      if (typeof a.time === 'string' && typeof b.time === 'string') return a.time.localeCompare(b.time);
      return a.time - b.time;
    });

    // v5: destroy old markers primitive, create new one
    if (this._seriesMarkers) {
      this._seriesMarkers.detach();
      this._seriesMarkers = null;
    }

    if (sorted.length > 0) {
      this._seriesMarkers = this._lwc.createSeriesMarkers(this._candleSeries, sorted);
    }
  }

  /** Get the underlying chart instance */
  getChart() { return this._chart; }

  /** Get the candlestick series */
  getCandleSeries() { return this._candleSeries; }

  /** Get the volume series */
  getVolumeSeries() { return this._volumeSeries; }

  /** Fit all content */
  fitContent() {
    this._chart.timeScale().fitContent();
    return this;
  }

  /** Destroy the chart and cleanup */
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

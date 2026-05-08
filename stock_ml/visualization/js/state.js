// Global state
var manifest = null;
var chart = null;
var candleSeries = null;
var volumeSeries = null;
var currentData = null;
var currentSymbol = null;
var currentMarket = 'all';
var currentMarketFamily = 'all';
var currentTimeframe = 'all';
var baseIndices = {};
var modelIndices = {};
var modelVisibility = {};
var showTable = false;
var modelDataCache = {};

// Symbol selector state
var allSymbolItems = [];      // Full list of {symbol, file, pnlData} for dropdown
var filteredSymbolItems = []; // Filtered by search
var highlightedIndex = -1;    // Keyboard navigation index
var sortByModel = '';         // '' = A-Z, 'v25' = sort by v25 PnL, etc.
var sortDescending = true;    // true = high-to-low, false = low-to-high

// Shared time helpers for Lightweight Charts.
const BUSINESS_DAY_RE = /^\d{4}-\d{2}-\d{2}$/;
const TZ_SUFFIX_RE = /(Z|[+-]\d{2}:?\d{2})$/i;

function isBusinessDayString(value) {
  return typeof value === 'string' && BUSINESS_DAY_RE.test(value);
}

function isIntradayString(value) {
  return typeof value === 'string' && !isBusinessDayString(value) && value.length > 10;
}

function normalizeTimeForChart(value) {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value !== 'string') return value;
  if (isBusinessDayString(value)) return value;

  let iso = value.trim();
  if (!iso) return value;
  if (iso.includes(' ') && !iso.includes('T')) iso = iso.replace(' ', 'T');
  // Intraday strings without timezone are assumed to be UTC.
  if (!TZ_SUFFIX_RE.test(iso)) iso += 'Z';

  const ms = Date.parse(iso);
  if (Number.isNaN(ms)) return value;
  return Math.floor(ms / 1000);
}

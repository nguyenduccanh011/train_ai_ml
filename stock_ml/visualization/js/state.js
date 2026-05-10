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
var currentRawData = null;
var currentYear = 'all';

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

function normalizeYearSelection(year) {
  if (typeof year !== 'string') return 'all';
  const value = year.trim().toLowerCase();
  if (!value || value === 'all') return 'all';
  return /^\d{4}$/.test(value) ? value : 'all';
}

function getYearFromDateLike(value) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    const epochMs = value > 1e12 ? value : value * 1000;
    const dt = new Date(epochMs);
    const year = dt.getUTCFullYear();
    return Number.isFinite(year) ? year : null;
  }
  if (typeof value !== 'string') return null;
  const raw = value.trim();
  if (!raw) return null;
  const match = raw.match(/^(\d{4})/);
  if (match) return Number(match[1]);
  let iso = raw;
  if (iso.includes(' ') && !iso.includes('T')) iso = iso.replace(' ', 'T');
  if (!TZ_SUFFIX_RE.test(iso)) iso += 'Z';
  const ms = Date.parse(iso);
  if (Number.isNaN(ms)) return null;
  return new Date(ms).getUTCFullYear();
}

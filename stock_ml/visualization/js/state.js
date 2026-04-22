// Global state
var manifest = null;
var chart = null;
var candleSeries = null;
var volumeSeries = null;
var currentData = null;
var currentSymbol = null;
var modelVisibility = {};
var showTable = false;
var modelDataCache = {};

// Symbol selector state
var allSymbolItems = [];      // Full list of {symbol, file, pnlData} for dropdown
var filteredSymbolItems = [];  // Filtered by search
var highlightedIndex = -1;     // Keyboard navigation index
var sortByModel = '';          // '' = A-Z, 'v25' = sort by v25 PnL, etc.
var sortDescending = true;     // true = high→low, false = low→high

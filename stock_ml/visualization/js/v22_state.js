// V22 dashboard state
let v22Chart = null;
let v22CandleSeries = null;
let v22VolumeSeries = null;

const v22Modes = {
  walk: { label: 'Walk-forward', dataDir: 'data_v22', prefix: 'v22' },
  full: { label: 'Full history', dataDir: 'data_v22_full', prefix: 'v22_full' },
};
let v22CurrentMode = v22Modes.walk;
let v22BaseIndex = null;
let v22ModelIndex = null;
let v22AllSymbols = [];
let v22FilteredSymbols = [];
let v22CurrentSymbol = null;
let v22CurrentBaseData = null;
let v22CurrentPayload = null;

const v22DataCache = new Map();
const v22MarkerFilters = {
  buy: true,
  win: true,
  loss: true,
};

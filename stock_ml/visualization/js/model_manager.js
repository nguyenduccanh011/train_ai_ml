const state = {
  rows: [],
  selected: new Set(),
  detailModelId: '',
  search: '',
  status: '',
  sortBy: 'updated_at',
  sortDir: 'desc',
  visibleOnly: false,
  loading: false,
};

const els = {
  searchInput: document.getElementById('searchInput'),
  statusSelect: document.getElementById('statusSelect'),
  sortBySelect: document.getElementById('sortBySelect'),
  sortDirSelect: document.getElementById('sortDirSelect'),
  visibleOnly: document.getElementById('visibleOnly'),
  refreshBtn: document.getElementById('refreshBtn'),
  selectAll: document.getElementById('selectAll'),
  tableBody: document.getElementById('tableBody'),
  selectionHint: document.getElementById('selectionHint'),
  summaryCount: document.getElementById('summaryCount'),
  summaryVisible: document.getElementById('summaryVisible'),
  summaryStatuses: document.getElementById('summaryStatuses'),
  detailEmpty: document.getElementById('detailEmpty'),
  detailView: document.getElementById('detailView'),
  detailKv: document.getElementById('detailKv'),
  bundleEmpty: document.getElementById('bundleEmpty'),
  bundleView: document.getElementById('bundleView'),
  bundleKv: document.getElementById('bundleKv'),
  metricsKv: document.getElementById('metricsKv'),
  artifactList: document.getElementById('artifactList'),
  auditEmpty: document.getElementById('auditEmpty'),
  auditList: document.getElementById('auditList'),
};

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function jsonText(value) {
  return escapeHtml(JSON.stringify(value, null, 2));
}

function fmt(value) {
  return value === null || value === undefined || value === '' ? '-' : escapeHtml(value);
}

function statusClass(value) {
  return `status ${escapeHtml(String(value || 'unknown'))}`;
}

function buildQuery() {
  const url = new URL('/api/models', window.location.href);
  if (state.search) url.searchParams.set('search', state.search);
  if (state.status) url.searchParams.set('status', state.status);
  if (state.visibleOnly) url.searchParams.set('visible_only', '1');
  if (state.sortBy) url.searchParams.set('sort_by', state.sortBy);
  if (state.sortDir) url.searchParams.set('sort_dir', state.sortDir);
  return url;
}

function updateSummary(payload) {
  const rows = payload.rows || [];
  const visible = rows.filter((row) => row.visible_in_dashboard).length;
  const statuses = new Set(rows.map((row) => row.status).filter(Boolean)).size;
  els.summaryCount.textContent = `${rows.length.toLocaleString()} rows`;
  els.summaryVisible.textContent = `${visible.toLocaleString()} visible`;
  els.summaryStatuses.textContent = `${statuses.toLocaleString()} statuses`;
}

function syncSelectionHint() {
  const count = state.selected.size;
  els.selectionHint.textContent = count ? `${count} selected` : 'No selection';
  els.selectAll.checked = count > 0 && count === state.rows.length;
  els.selectAll.indeterminate = count > 0 && count < state.rows.length;
}

function renderTable() {
  const rows = state.rows;
  if (!rows.length) {
    els.tableBody.innerHTML = '<tr><td colspan="12" class="empty">No rows.</td></tr>';
    syncSelectionHint();
    return;
  }

  els.tableBody.innerHTML = rows.map((row) => {
    const selected = state.selected.has(row.id) ? 'selected' : '';
    const visible = row.visible_in_dashboard ? 'yes' : 'no';
    return `
      <tr class="${selected}" data-model-id="${escapeHtml(row.id)}">
        <td><input type="checkbox" data-select="${escapeHtml(row.id)}" ${selected ? 'checked' : ''}></td>
        <td class="wrap">
          <div class="mono">${fmt(row.id)}</div>
          <div class="muted">${fmt(row.name)}</div>
        </td>
        <td><span class="${statusClass(row.status)}">${fmt(row.status)}</span></td>
        <td>${visible}</td>
        <td>${fmt(row.market)}</td>
        <td class="wrap">${fmt(row.strategy)}</td>
        <td class="wrap">${fmt(row.feature_set)}</td>
        <td class="wrap">${fmt(row.entry_model)}</td>
        <td class="wrap">${fmt(row.exit_model)}</td>
        <td class="num">${fmt(row.priority)}</td>
        <td class="mono">${fmt(row.updated_at)}</td>
        <td>
          <button data-row-action="activate" data-model-id="${escapeHtml(row.id)}" type="button">A</button>
          <button data-row-action="archive" data-model-id="${escapeHtml(row.id)}" type="button">Ar</button>
          <button data-row-action="quarantine" data-model-id="${escapeHtml(row.id)}" type="button">Q</button>
        </td>
      </tr>
    `;
  }).join('');
  syncSelectionHint();
}

function renderDetail(model) {
  if (!model) {
    els.detailEmpty.hidden = false;
    els.detailView.hidden = true;
    return;
  }
  const items = [
    ['id', model.id],
    ['name', model.name],
    ['version', model.version],
    ['status', model.status],
    ['visible', String(model.visible_in_dashboard)],
    ['market', model.market],
    ['strategy', model.strategy],
    ['feature_set', model.feature_set],
    ['entry_model', model.entry_model],
    ['exit_model', model.exit_model],
    ['priority', model.priority],
    ['created_at', model.created_at],
    ['updated_at', model.updated_at],
    ['retired_at', model.retired_at],
    ['retired_reason', model.retired_reason],
  ];
  els.detailKv.innerHTML = items.map(([k, v]) => `<div class="k">${fmt(k)}</div><div class="v">${fmt(v)}</div>`).join('');
  els.detailEmpty.hidden = true;
  els.detailView.hidden = false;
}

function renderBundle(bundle) {
  if (!bundle) {
    els.bundleEmpty.hidden = false;
    els.bundleView.hidden = true;
    return;
  }
  const model = bundle.model || {};
  const run = bundle.run || {};
  const metrics = bundle.metrics_snapshot || {};
  els.bundleKv.innerHTML = [
    ['run_id', run.id],
    ['run_name', run.run_name],
    ['status', run.status],
    ['config_hash', run.config_hash],
    ['config_path', run.config_path],
    ['resolved_config_path', run.resolved_config_path],
  ].map(([k, v]) => `<div class="k">${fmt(k)}</div><div class="v">${fmt(v)}</div>`).join('');
  els.metricsKv.innerHTML = [
    ['wr', metrics.wr],
    ['pf', metrics.pf],
    ['total_pnl', metrics.total_pnl],
    ['max_drawdown', metrics.max_drawdown],
    ['sharpe', metrics.sharpe],
    ['mdd_per_symbol', metrics.mdd_per_symbol],
    ['yearly_consistency', metrics.yearly_consistency],
    ['composite_score', metrics.composite_score],
  ].map(([k, v]) => `<div class="k">${fmt(k)}</div><div class="v">${fmt(v)}</div>`).join('');
  const artifacts = bundle.artifacts || [];
  els.artifactList.innerHTML = artifacts.length
    ? artifacts.map((artifact) => `
        <div class="audit-item">
          <div class="audit-head">
            <div><span class="badge">${fmt(artifact.kind)}</span></div>
            <div class="mono muted">${fmt(artifact.size_bytes)} B</div>
          </div>
          <div class="mono wrap">${fmt(artifact.path)}</div>
        </div>
      `).join('')
    : '<div class="empty">No artifacts.</div>';
  els.bundleEmpty.hidden = true;
  els.bundleView.hidden = false;
}

function renderAudit(rows) {
  if (!rows || !rows.length) {
    els.auditEmpty.hidden = false;
    els.auditList.hidden = true;
    return;
  }
  els.auditList.innerHTML = rows.map((row) => `
    <div class="audit-item">
      <div class="audit-head">
        <div>
          <span class="badge">${fmt(row.entity_type)}</span>
          <span class="badge">${fmt(row.action)}</span>
        </div>
        <div class="mono muted">${fmt(row.created_at)}</div>
      </div>
      <div class="mono">${fmt(row.entity_id)} - ${fmt(row.actor)}</div>
      <div class="muted wrap">${fmt(row.reason)}</div>
      <pre class="mono wrap" style="white-space:pre-wrap;margin:6px 0 0;">${jsonText(row.payload_json || {})}</pre>
    </div>
  `).join('');
  els.auditEmpty.hidden = true;
  els.auditList.hidden = false;
}

async function loadModels() {
  state.loading = true;
  els.refreshBtn.disabled = true;
  try {
    const resp = await fetch(buildQuery().toString());
    const payload = await resp.json();
    if (!resp.ok) throw new Error(payload.error || 'failed to load models');
    state.rows = payload.rows || [];
    state.selected = new Set([...state.selected].filter((id) => state.rows.some((row) => row.id === id)));
    updateSummary(payload);
    renderTable();
    if (!state.detailModelId || !state.rows.some((row) => row.id === state.detailModelId)) {
      state.detailModelId = state.rows[0]?.id || '';
    }
    if (state.detailModelId) {
      await loadDetail(state.detailModelId);
    } else {
      renderDetail(null);
      renderBundle(null);
      renderAudit([]);
    }
  } catch (err) {
    els.tableBody.innerHTML = `<tr><td colspan="12" class="error">${escapeHtml(err.message)}</td></tr>`;
  } finally {
    state.loading = false;
    els.refreshBtn.disabled = false;
    syncSelectionHint();
  }
}

async function loadDetail(modelId) {
  state.detailModelId = modelId;
  try {
    const [detailResp, auditResp] = await Promise.all([
      fetch(`/api/models/${encodeURIComponent(modelId)}`),
      fetch(`/api/models/${encodeURIComponent(modelId)}/audit?limit=50`),
    ]);
    const detailPayload = await detailResp.json();
    const auditPayload = await auditResp.json();
    if (!detailResp.ok) throw new Error(detailPayload.error || 'failed to load detail');
    if (!auditResp.ok) throw new Error(auditPayload.error || 'failed to load audit');
    renderDetail(detailPayload.model);
    renderBundle(detailPayload.bundle);
    renderAudit(auditPayload.rows || detailPayload.audit_log || []);
  } catch (err) {
    renderDetail(null);
    renderBundle(null);
    renderAudit([]);
    els.detailEmpty.textContent = err.message;
    els.detailEmpty.hidden = false;
  }
}

async function applyAction(action, ids) {
  const targetIds = ids.length ? ids : [...state.selected];
  if (!targetIds.length) return;
  const plural = targetIds.length > 1 ? 'selected models' : 'model';
  if (action === 'purge') {
    if (!window.confirm(`Purge ${targetIds.length} ${plural}?`)) return;
  }
  const reason = window.prompt(`Reason for ${action}?`, '') ?? '';
  const payload = {
    action,
    model_ids: targetIds,
    actor: 'ui',
    reason,
  };
  const resp = await fetch('/api/models/bulk', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const out = await resp.json();
  if (!resp.ok) throw new Error(out.error || 'bulk action failed');
  state.selected.clear();
  await loadModels();
}

function toggleSelection(modelId, checked) {
  if (checked) state.selected.add(modelId);
  else state.selected.delete(modelId);
  syncSelectionHint();
  renderTable();
}

function bindEvents() {
  const refresh = () => loadModels();
  els.refreshBtn.addEventListener('click', refresh);
  els.selectAll.addEventListener('change', () => {
    state.selected = els.selectAll.checked ? new Set(state.rows.map((row) => row.id)) : new Set();
    renderTable();
  });
  els.searchInput.addEventListener('input', debounce((event) => {
    state.search = event.target.value.trim();
    loadModels();
  }, 250));
  els.statusSelect.addEventListener('change', (event) => {
    state.status = event.target.value;
    loadModels();
  });
  els.sortBySelect.addEventListener('change', (event) => {
    state.sortBy = event.target.value;
    loadModels();
  });
  els.sortDirSelect.addEventListener('change', (event) => {
    state.sortDir = event.target.value;
    loadModels();
  });
  els.visibleOnly.addEventListener('change', (event) => {
    state.visibleOnly = event.target.checked;
    loadModels();
  });
  document.querySelectorAll('[data-action]').forEach((button) => {
    button.addEventListener('click', async () => {
      try {
        await applyAction(button.dataset.action, []);
      } catch (err) {
        window.alert(err.message);
      }
    });
  });
  els.tableBody.addEventListener('click', async (event) => {
    const target = event.target;
    const row = target.closest('tr[data-model-id]');
    if (!row) return;
    const modelId = row.dataset.modelId;
    if (target.matches('input[type="checkbox"][data-select]')) {
      toggleSelection(modelId, target.checked);
      return;
    }
    const actionButton = target.closest('[data-row-action]');
    if (actionButton) {
      try {
        await applyAction(actionButton.dataset.rowAction, [modelId]);
      } catch (err) {
        window.alert(err.message);
      }
      return;
    }
    state.detailModelId = modelId;
    await loadDetail(modelId);
  });
}

function debounce(fn, wait) {
  let timer = null;
  return (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => fn(...args), wait);
  };
}

bindEvents();
loadModels();

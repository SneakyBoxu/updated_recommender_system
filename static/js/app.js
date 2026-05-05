/* ═══════════════════════════════════════════════════════════════════
   MineMatrix — Client-Side JavaScript
   SSE streaming, chart rendering, toast system, UI utilities
   ══════════════════════════════════════════════════════════════════ */

'use strict';

// ─── Helpers ─────────────────────────────────────────────────────────
function showElements(ids) {
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.remove('hidden');
  });
}

function hideElements(ids) {
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.add('hidden');
  });
}

function clearLog(logBodyId) {
  const el = document.getElementById(logBodyId);
  if (el) el.innerHTML = '';
}

function appendLog(logBodyId, msg, level = 'info') {
  const el = document.getElementById(logBodyId);
  if (!el) return;

  const now = new Date();
  const ts = now.toTimeString().slice(0, 8);

  const line = document.createElement('div');
  line.className = `log-line ${level}`;
  line.innerHTML = `<span class="log-ts">[${ts}]</span><span class="log-msg">${escapeHtml(msg)}</span>`;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ─── Toast ────────────────────────────────────────────────────────────
function showToast(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s ease';
    setTimeout(() => toast.remove(), 350);
  }, duration);
}

// ─── SSE Streaming ────────────────────────────────────────────────────
/**
 * Open an SSE stream and pipe log messages to the log panel.
 * @param {string} url        - SSE endpoint URL
 * @param {string} logBodyId  - ID of the .log-body element
 * @param {function} onDone   - called with `results` object when stream ends
 */
function startSSEStream(url, logBodyId, onDone) {
  clearLog(logBodyId);
  appendLog(logBodyId, `Connecting to ${url}…`, 'info');

  const source = new EventSource(url);

  source.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);

      if (data.type === 'log') {
        appendLog(logBodyId, data.msg, data.level || 'info');
      } else if (data.type === 'done') {
        appendLog(logBodyId, '─── Analysis complete ───', 'success');
        source.close();
        if (typeof onDone === 'function') onDone(data.results);
        showToast('Analysis completed successfully!', 'success');
      } else if (data.type === 'error') {
        appendLog(logBodyId, '✗ Error: ' + data.msg, 'error');
        source.close();
        showToast('Analysis failed: ' + data.msg, 'error');
      }
    } catch (e) {
      appendLog(logBodyId, '[parse error] ' + event.data, 'error');
    }
  };

  source.onerror = () => {
    appendLog(logBodyId, '✗ Connection lost. Server may still be computing…', 'error');
    source.close();
  };
}

// ─── Chart Fetching ───────────────────────────────────────────────────
/**
 * Fetch a chart from the server and display it in a wrapper element.
 * @param {string} endpoint    - /api/chart/svd | nmf | pca | stats
 * @param {HTMLElement} wrap   - container element
 */
async function fetchChart(endpoint, wrap) {
  wrap.innerHTML = '<div class="placeholder-inner spinner-text">⏳ Rendering chart…</div>';
  try {
    const res = await fetch(endpoint);
    const data = await res.json();
    if (data.image) {
      wrap.innerHTML = `<img src="${data.image}" class="chart-img" alt="Analysis chart" />`;
    } else {
      wrap.innerHTML = `<div class="placeholder-inner error-text">${escapeHtml(data.error || 'Unknown error')}</div>`;
    }
  } catch (e) {
    wrap.innerHTML = `<div class="placeholder-inner error-text">Failed to load chart: ${escapeHtml(e.message)}</div>`;
  }
}

// ─── Hamburger / Sidebar ──────────────────────────────────────────────
(function initSidebar() {
  const hamburger = document.getElementById('hamburger');
  const sidebar   = document.getElementById('sidebar');
  if (!hamburger || !sidebar) return;

  hamburger.addEventListener('click', () => sidebar.classList.toggle('open'));

  document.addEventListener('click', (e) => {
    if (sidebar.classList.contains('open') &&
        !sidebar.contains(e.target) &&
        e.target !== hamburger) {
      sidebar.classList.remove('open');
    }
  });
})();

// ─── Reload Modal ─────────────────────────────────────────────────────
(function initReloadModal() {
  const reloadBtn   = document.getElementById('reload-btn');
  const modal       = document.getElementById('reload-modal');
  const cancelBtn   = document.getElementById('reload-cancel');
  const autoBtn     = document.getElementById('reload-auto');
  const regenBtn    = document.getElementById('reload-regen');

  if (!reloadBtn || !modal) return;

  reloadBtn.addEventListener('click', () => modal.classList.remove('hidden'));
  cancelBtn.addEventListener('click', () => modal.classList.add('hidden'));
  modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.add('hidden'); });

  async function doReload(mode) {
    modal.classList.add('hidden');
    showToast('Reloading data…', 'info');
    try {
      const res = await fetch('/api/reload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      });
      const data = await res.json();
      if (data.ok) {
        // Update source labels
        const newSrc = data.data_source || '—';
        const sideLabel   = document.getElementById('sidebar-source-label');
        const headerLabel = document.getElementById('header-source-label');
        if (sideLabel)   sideLabel.textContent   = newSrc;
        if (headerLabel) headerLabel.textContent  = newSrc;
        showToast('Data reloaded: ' + newSrc, 'success');
        setTimeout(() => location.reload(), 1200);
      } else {
        showToast('Reload failed', 'error');
      }
    } catch (e) {
      showToast('Reload error: ' + e.message, 'error');
    }
  }

  autoBtn.addEventListener('click', () => doReload('auto'));
  regenBtn.addEventListener('click', () => doReload('regenerate'));
})();

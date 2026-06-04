// ── Storage helpers ──────────────────────────────────────────────────────────

async function loadAllTools() {
  return new Promise(resolve => {
    chrome.storage.local.get('tools', d => resolve(d.tools || {}));
  });
}

async function saveToolState(toolId, patch) {
  return new Promise(resolve => {
    chrome.storage.local.get('tools', d => {
      const tools = d.tools || {};
      tools[toolId] = { ...(tools[toolId] || {}), ...patch };
      chrome.storage.local.set({ tools }, resolve);
    });
  });
}

// ── Main list ────────────────────────────────────────────────────────────────

async function renderList() {
  const allStates = await loadAllTools();
  const list = document.getElementById('toolsList');
  list.innerHTML = '';

  let activeCount = 0;

  for (const tool of TOOLS) {
    const state = { ...tool.defaultState, ...(allStates[tool.id] || {}) };
    if (state.enabled) activeCount++;

    const card = document.createElement('div');
    card.className = 'tool-card' + (state.enabled ? ' enabled' : '');
    card.dataset.toolId = tool.id;
    card.innerHTML = `
      <div class="tool-icon">${tool.icon}</div>
      <div class="tool-info">
        <div class="tool-name">${tool.name}</div>
        <div class="tool-desc">${tool.desc}</div>
      </div>
      <div class="tool-right">
        <div class="tool-status"></div>
        <button class="settings-btn" title="Paramètres">⋯</button>
        ${tool.noToggle ? '' : `
        <label class="toggle" title="${state.enabled ? 'Désactiver' : 'Activer'}">
          <input type="checkbox" ${state.enabled ? 'checked' : ''} />
          <span class="slider"></span>
        </label>`}
      </div>
    `;

    // Toggle (seulement pour les outils qui en ont un)
    const checkbox = card.querySelector('input[type=checkbox]');
    if (checkbox) checkbox.addEventListener('change', async (e) => {
      e.stopPropagation();
      const enabled = e.target.checked;
      const patch = enabled
        ? { enabled: true, startTime: Date.now(), pingCount: 0 }
        : { enabled: false, startTime: null };
      await saveToolState(tool.id, patch);
      chrome.runtime.sendMessage({ action: enabled ? 'tool-start' : 'tool-stop', toolId: tool.id });
      renderList();
    });

    // Settings button → detail panel
    const settingsBtn = card.querySelector('.settings-btn');
    settingsBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      openDetail(tool, state);
    });

    // Click card → also open detail
    card.addEventListener('click', (e) => {
      if (e.target.closest('.toggle') || e.target.closest('.settings-btn')) return;
      openDetail(tool, state);
    });

    list.appendChild(card);
  }

  // Update header count
  document.getElementById('activeCount').textContent = activeCount;
  document.getElementById('activePlural').textContent = activeCount > 1 ? 's' : '';

  // Footer
  const footer = document.getElementById('footerStatus');
  footer.textContent = activeCount === 0
    ? 'Tous les outils sont désactivés.'
    : `${activeCount} outil${activeCount > 1 ? 's' : ''} actif${activeCount > 1 ? 's' : ''}.`;
}

// ── Detail panel ─────────────────────────────────────────────────────────────

function setView(view) {
  const mainView = document.getElementById('mainView');
  const detailPanel = document.getElementById('detailPanel');
  if (view === 'detail') {
    mainView.style.display = 'none';
    detailPanel.classList.add('visible');
  } else {
    mainView.style.display = '';
    detailPanel.classList.remove('visible');
  }
}

function openDetail(tool, state) {
  document.getElementById('detailTitle').textContent = `${tool.icon}  ${tool.name}`;
  const body = document.getElementById('detailBody');
  body.innerHTML = '';

  if (tool.renderDetail) {
    tool.renderDetail(body, state, (patch) => saveToolState(tool.id, patch));
  } else {
    body.innerHTML = '<div class="info-box">Aucun paramètre disponible.</div>';
  }

  setView('detail');
}

document.getElementById('backBtn').addEventListener('click', () => {
  setView('main');
  renderList();
});

// ── Init ─────────────────────────────────────────────────────────────────────
renderList();

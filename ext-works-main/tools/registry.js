// Registre des outils — ajoute ici de nouveaux outils.
// Chaque outil expose : id, name, desc, icon, et optionnellement renderDetail(container, state, save).

const TOOLS = [
  {
    id: 'teams-available',
    name: 'Teams Disponible',
    desc: 'Maintient ton statut sur Disponible',
    icon: '💬',
    defaultState: { enabled: false, interval: 60, pingCount: 0, startTime: null },

    renderDetail(container, state, save) {
      function formatUptime(start) {
        if (!start) return '0m';
        const s = Math.floor((Date.now() - start) / 1000);
        if (s < 60) return `${s}s`;
        const m = Math.floor(s / 60);
        if (m < 60) return `${m}m`;
        return `${Math.floor(m/60)}h${m%60 > 0 ? m%60+'m' : ''}`;
      }

      container.innerHTML = `
        <div class="section-title">Statut</div>
        <div class="stat-grid">
          <div class="stat-box">
            <div class="stat-val" id="ta-pings">${state.pingCount}</div>
            <div class="stat-lbl">Pings</div>
          </div>
          <div class="stat-box">
            <div class="stat-val" id="ta-uptime">${formatUptime(state.enabled ? state.startTime : null)}</div>
            <div class="stat-lbl">Actif depuis</div>
          </div>
          <div class="stat-box">
            <div class="stat-val" id="ta-interval">${state.interval}s</div>
            <div class="stat-lbl">Intervalle</div>
          </div>
        </div>

        <div class="section-title">Intervalle de ping</div>
        <div class="range-wrap">
          <label>Toutes les <span id="ta-range-val">${state.interval}s</span></label>
          <input type="range" id="ta-range" min="10" max="300" step="10" value="${state.interval}" />
          <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-dim)">
            <span>10s</span><span>5 min</span>
          </div>
        </div>

        <div class="info-box">
          Simule un déplacement de souris et une touche Shift sur l'onglet Teams
          (<code>teams.cloud.microsoft</code>) pour prévenir la mise en inactivité.
        </div>
      `;

      const rangeEl = container.querySelector('#ta-range');
      const rangeVal = container.querySelector('#ta-range-val');
      const intervalDisplay = container.querySelector('#ta-interval');

      rangeEl.addEventListener('input', () => {
        const v = parseInt(rangeEl.value);
        rangeVal.textContent = v + 's';
        intervalDisplay.textContent = v + 's';
        save({ interval: v });
        chrome.runtime.sendMessage({ action: 'tool-update', toolId: 'teams-available', interval: v });
      });

      // Live uptime refresh
      const timer = setInterval(() => {
        const ping = container.querySelector('#ta-pings');
        const uptime = container.querySelector('#ta-uptime');
        if (!ping) { clearInterval(timer); return; }
        chrome.storage.local.get('tools', d => {
          const s = (d.tools || {})['teams-available'] || {};
          ping.textContent = s.pingCount ?? 0;
          uptime.textContent = formatUptime(s.enabled ? s.startTime : null);
        });
      }, 1000);
    },
  },

  // ── Prochains outils ici ──

  {
    id: 'time-bomb',
    name: 'Time Bomb',
    desc: 'Jeu de cartes multijoueur local (IELLO)',
    icon: '💣',
    defaultState: {},
    noToggle: true,

    renderDetail(container) {
      container.innerHTML = `
        <div class="info-box">
          Jeu pour 4 à 8 joueurs sur le même réseau local, sans serveur.<br/>
          Le panneau latéral reste ouvert — la connexion est maintenue.
        </div>
        <button id="tb-open" style="width:100%;padding:10px;background:var(--accent);color:#fff;border:none;border-radius:var(--radius);font-size:14px;cursor:pointer;margin-top:8px;">
          💣 Ouvrir Time Bomb
        </button>
      `;
      container.querySelector('#tb-open').addEventListener('click', () => {
        chrome.windows.getCurrent({}, (win) => {
          chrome.sidePanel.open({ windowId: win.id });
        });
      });
    },
  },
];

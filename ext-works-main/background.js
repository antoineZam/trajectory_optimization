// ── Storage ──────────────────────────────────────────────────────────────────

async function getToolState(toolId) {
  return new Promise(resolve => {
    chrome.storage.local.get('tools', d => {
      resolve((d.tools || {})[toolId] || {});
    });
  });
}

async function setToolState(toolId, patch) {
  return new Promise(resolve => {
    chrome.storage.local.get('tools', d => {
      const tools = d.tools || {};
      tools[toolId] = { ...(tools[toolId] || {}), ...patch };
      chrome.storage.local.set({ tools }, resolve);
    });
  });
}

// ── Teams Available ───────────────────────────────────────────────────────────

const TEAMS_ALARM = 'tool-teams-available';
const TEAMS_URL   = 'teams.cloud.microsoft';

async function teamsStart() {
  const state = await getToolState('teams-available');
  const interval = state.interval ?? 60;
  chrome.alarms.create(TEAMS_ALARM, {
    delayInMinutes: interval / 60,
    periodInMinutes: interval / 60,
  });
  teamsPing();
}

async function teamsStop() {
  chrome.alarms.clear(TEAMS_ALARM);
}

async function teamsPing() {
  const state = await getToolState('teams-available');
  if (!state.enabled) return;

  const tabs = await new Promise(r =>
    chrome.tabs.query({}, tabs => r(tabs.filter(t => t.url && t.url.includes(TEAMS_URL))))
  );

  for (const tab of tabs) {
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => {
          const x = Math.floor(Math.random() * window.innerWidth);
          const y = Math.floor(Math.random() * window.innerHeight);
          document.dispatchEvent(new MouseEvent('mousemove', {
            bubbles: true, clientX: x, clientY: y, movementX: 1, movementY: 1,
          }));
          document.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Shift', shiftKey: true }));
          document.dispatchEvent(new KeyboardEvent('keyup',   { bubbles: true, key: 'Shift', shiftKey: true }));
        },
      });
    } catch { /* tab navigated away */ }
  }

  await setToolState('teams-available', { pingCount: (state.pingCount ?? 0) + 1 });
}

// ── Alarm dispatcher ─────────────────────────────────────────────────────────

chrome.alarms.onAlarm.addListener(alarm => {
  if (alarm.name === TEAMS_ALARM) teamsPing();
  // Future tools: add more alarm handlers here
});

// ── Message dispatcher ────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.toolId === 'teams-available') {
    if (msg.action === 'tool-start') teamsStart();
    if (msg.action === 'tool-stop')  teamsStop();
    if (msg.action === 'tool-update') {
      // Re-schedule alarm with new interval
      const interval = msg.interval ?? 60;
      chrome.alarms.clear(TEAMS_ALARM, () => {
        chrome.alarms.create(TEAMS_ALARM, {
          delayInMinutes: interval / 60,
          periodInMinutes: interval / 60,
        });
      });
    }
  }
  // Future tools: add more message handlers here
});

// ── Restore on restart ────────────────────────────────────────────────────────

async function restoreTools() {
  const state = await getToolState('teams-available');
  if (state.enabled) teamsStart();
  // Future tools: restore here
}

chrome.runtime.onStartup.addListener(restoreTools);
chrome.runtime.onInstalled.addListener(restoreTools);

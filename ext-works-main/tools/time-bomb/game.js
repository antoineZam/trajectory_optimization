import { TimeBombHost, TimeBombPeer } from './rtc.js';
import {
  generateDeck, dealCards, assignRoles,
  CARDS_PER_ROUND, MAX_ROUNDS, COLORS,
} from './deck.js';

const MIN_PLAYERS = 4;
const MAX_PLAYERS = 8;

const app = document.getElementById('app');

// ─── État local ───────────────────────────────────────────────────────────────

let isHost = false;
let myPeerId = null;      // 'host' ou 'pN'
let myName = '';
let net = null;           // TimeBombHost | TimeBombPeer
let gameRole = null;      // 'SHERLOCK' | 'MORIARTY' — reçu en début de partie
let myHand = [];          // cardObj[] — main privée du joueur local

// État public reçu via GAME_STATE (commun à hôte et joueurs)
let gs = null;

// État autoritaire : uniquement chez l'hôte
let auth = null;

// ─── Utilitaires DOM ─────────────────────────────────────────────────────────

function render(html) { app.innerHTML = html; }
function $(sel) { return app.querySelector(sel); }

function copyToClipboard(text) {
  navigator.clipboard.writeText(text).catch(() => {
    const ta = Object.assign(document.createElement('textarea'), { value: text });
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
  });
}

function showToast(msg, type = 'info') {
  const el = Object.assign(document.createElement('div'), {
    className: `toast toast-${type}`,
    textContent: msg,
  });
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 2500);
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ─── Écran d'accueil ─────────────────────────────────────────────────────────

function showHome() {
  render(`
    <div class="screen home">
      <div class="logo">💣</div>
      <h1>Time Bomb</h1>
      <p class="subtitle">Jeu de déduction — 4 à 8 joueurs</p>
      <div class="btn-group">
        <button id="btn-host" class="btn btn-primary">Créer une partie</button>
        <button id="btn-join" class="btn btn-secondary">Rejoindre</button>
      </div>
    </div>
  `);
  $('#btn-host').onclick = showHostSetup;
  $('#btn-join').onclick = showJoinSetup;
}

// ─── Hôte : configuration ─────────────────────────────────────────────────────

function showHostSetup() {
  render(`
    <div class="screen setup">
      <button class="btn-back" id="back">← Retour</button>
      <h2>💣 Créer une partie</h2>
      <label>Votre nom
        <input id="host-name" type="text" maxlength="20" placeholder="ex: Alice" autocomplete="off"/>
      </label>
      <button id="btn-start-host" class="btn btn-primary" style="margin-top:12px">Créer le salon</button>
    </div>
  `);
  $('#back').onclick = showHome;
  $('#btn-start-host').onclick = () => {
    const name = $('#host-name').value.trim();
    if (!name) { showToast('Entrez votre nom', 'error'); return; }
    myName = name;
    myPeerId = 'host';
    isHost = true;
    net = new TimeBombHost();
    net.onConnect = () => {};     // l'ajout se fait à la réception de JOIN
    net.onDisconnect = onPeerDisconnect;
    net.onMessage = onHostReceive;
    showHostLobby([{ peerId: 'host', name }]);
  };
}

// ─── Lobby hôte ───────────────────────────────────────────────────────────────

let lobbyPlayers = [];   // [{ peerId, name }]

function showHostLobby(players) {
  lobbyPlayers = players;
  const canStart = players.length >= MIN_PLAYERS;

  render(`
    <div class="screen lobby">
      <h2>💣 Salon — ${players.length}/${MAX_PLAYERS} joueur(s)</h2>

      <div class="section-title">Connecter un joueur</div>
      <div class="connect-box">
        <textarea id="offer-input" rows="3" placeholder="Collez ici le code OFFRE du joueur…"></textarea>
        <button id="btn-accept" class="btn btn-secondary">Accepter → générer réponse</button>
        <div id="answer-area" style="display:none">
          <div class="section-title" style="margin-top:8px">Réponse à donner au joueur :</div>
          <textarea id="answer-out" rows="3" readonly></textarea>
          <button id="btn-copy-answer" class="btn btn-outline">📋 Copier la réponse</button>
        </div>
      </div>

      <div class="section-title">Joueurs connectés</div>
      <ul class="player-list">
        ${players.map(p => `
          <li class="player-item">
            <span class="dot dot-green"></span>
            ${escHtml(p.name)}${p.peerId === 'host' ? ' <em>(vous — hôte)</em>' : ''}
          </li>`).join('')}
      </ul>

      <div class="min-players-hint ${canStart ? 'hidden' : ''}">
        ⏳ En attente de ${MIN_PLAYERS - players.length} joueur(s) supplémentaire(s)…
      </div>

      <button id="btn-launch" class="btn btn-primary btn-launch"
              ${canStart ? '' : 'disabled'}>
        🚀 Lancer la partie (${players.length} joueur(s))
      </button>
    </div>
  `);

  $('#btn-accept').onclick = async () => {
    const offerB64 = $('#offer-input').value.trim();
    if (!offerB64) { showToast('Collez d\'abord le code offre', 'error'); return; }
    const btn = $('#btn-accept');
    btn.disabled = true;
    btn.textContent = '⏳ Connexion…';
    try {
      const answerB64 = await net.acceptOffer(offerB64);
      $('#answer-area').style.display = 'block';
      $('#answer-out').value = answerB64;
      $('#offer-input').value = '';
    } catch {
      showToast('Code invalide', 'error');
    }
    btn.textContent = 'Accepter → générer réponse';
    btn.disabled = false;
  };

  app.addEventListener('click', (e) => {
    if (e.target.id === 'btn-copy-answer') {
      copyToClipboard($('#answer-out')?.value ?? '');
      showToast('Réponse copiée !');
    }
  });

  if (canStart) {
    $('#btn-launch').onclick = startGame;
  }
}

function onPeerDisconnect(peerId) {
  lobbyPlayers = lobbyPlayers.filter(p => p.peerId !== peerId);
  if (auth && auth.phase !== 'LOBBY') {
    auth.players[peerId] && (auth.players[peerId].disconnected = true);
    broadcastGameState();
    updateHostView();
  } else {
    broadcastLobby();
    showHostLobby(lobbyPlayers);
  }
}

function onHostReceive(peerId, msg) {
  switch (msg.type) {
    case 'JOIN': {
      lobbyPlayers.push({ peerId, name: msg.name });
      net.send(peerId, { type: 'WELCOME', peerId, playerList: lobbyPlayers });
      broadcastLobby();
      showHostLobby(lobbyPlayers);
      break;
    }
    case 'FLIP_CARD': {
      let cardId = msg.cardId;
      // Le peer ne connaît pas l'ID exact — on prend la première carte disponible
      if (cardId === '__first__') {
        const hand = auth?.players[msg.targetPeerId]?.hand;
        cardId = hand && hand.length > 0 ? hand[0] : null;
      }
      if (cardId) handleFlipCard(peerId, msg.targetPeerId, cardId);
      break;
    }
  }
}

function broadcastLobby() {
  net.broadcast({
    type: 'LOBBY_UPDATE',
    players: lobbyPlayers,
    canStart: lobbyPlayers.length >= MIN_PLAYERS,
  });
}

// ─── Joueur : rejoindre ───────────────────────────────────────────────────────

function showJoinSetup() {
  render(`
    <div class="screen setup">
      <button class="btn-back" id="back">← Retour</button>
      <h2>💣 Rejoindre une partie</h2>
      <label>Votre nom
        <input id="join-name" type="text" maxlength="20" placeholder="ex: Bob" autocomplete="off"/>
      </label>
      <button id="btn-gen-offer" class="btn btn-primary" style="margin-top:12px">Générer mon code</button>

      <div id="offer-area" style="display:none;margin-top:16px">
        <div class="section-title">Donnez ce code à l'hôte :</div>
        <textarea id="offer-out" rows="3" readonly></textarea>
        <button id="btn-copy-offer" class="btn btn-outline">📋 Copier mon code</button>

        <div class="section-title" style="margin-top:12px">Collez la réponse de l'hôte :</div>
        <textarea id="answer-input" rows="3" placeholder="Code réponse de l'hôte…"></textarea>
        <button id="btn-connect" class="btn btn-primary">Se connecter</button>
      </div>
    </div>
  `);
  $('#back').onclick = showHome;

  $('#btn-gen-offer').onclick = async () => {
    const name = $('#join-name').value.trim();
    if (!name) { showToast('Entrez votre nom', 'error'); return; }
    myName = name;
    isHost = false;
    net = new TimeBombPeer();
    net.onMessage = onPeerReceive;
    net.onConnect = () => {
      net.send({ type: 'JOIN', name: myName });
      showToast('Connecté à l\'hôte !', 'success');
    };
    net.onDisconnect = () => showToast('Déconnecté', 'error');

    const btn = $('#btn-gen-offer');
    btn.disabled = true;
    btn.textContent = '⏳ Génération…';
    try {
      const offerB64 = await net.createOffer();
      $('#offer-area').style.display = 'block';
      $('#offer-out').value = offerB64;
      btn.textContent = 'Code généré ✓';
    } catch {
      showToast('Erreur WebRTC', 'error');
      btn.textContent = 'Générer mon code';
      btn.disabled = false;
    }
  };

  app.addEventListener('click', (e) => {
    if (e.target.id === 'btn-copy-offer') {
      copyToClipboard($('#offer-out')?.value ?? '');
      showToast('Code copié !');
    }
    if (e.target.id === 'btn-connect') {
      const answerB64 = $('#answer-input').value.trim();
      if (!answerB64) { showToast('Collez la réponse de l\'hôte', 'error'); return; }
      net.acceptAnswer(answerB64)
        .catch((e) => showToast('Erreur : ' + (e?.message || 'code invalide'), 'error'));
    }
  });
}

function onPeerReceive(msg) {
  switch (msg.type) {
    case 'WELCOME':
      myPeerId = msg.peerId;
      showPeerLobby(msg.playerList);
      break;
    case 'LOBBY_UPDATE':
      if (gs?.phase !== 'ROUND') showPeerLobby(msg.players);
      break;
    case 'YOUR_ROLE':
      gameRole = msg.role;
      break;
    case 'YOUR_HAND':
      myHand = msg.cards;
      break;
    case 'GAME_STATE':
      gs = msg;
      renderGame();
      break;
    case 'PLAYER_LEFT':
      showToast('Un joueur a quitté la partie', 'error');
      break;
  }
}

function showPeerLobby(players) {
  render(`
    <div class="screen lobby">
      <h2>💣 Salon — ${players.length} joueur(s)</h2>
      <div class="section-title">Joueurs connectés</div>
      <ul class="player-list">
        ${players.map(p => `
          <li class="player-item">
            <span class="dot dot-green"></span>
            ${escHtml(p.name)}${p.peerId === 'host' ? ' <em>(hôte)</em>' : ''}
            ${p.peerId === myPeerId ? ' <em>(vous)</em>' : ''}
          </li>`).join('')}
      </ul>
      <div class="waiting-msg">⏳ En attente du lancement par l'hôte…</div>
    </div>
  `);
}

// ─── Logique de jeu (côté hôte uniquement) ────────────────────────────────────

function startGame() {
  const playerOrder = lobbyPlayers.map(p => p.peerId);
  const round = 0;
  const deck = generateDeck(playerOrder.length, round);
  const cpr = CARDS_PER_ROUND[playerOrder.length][round];
  const { hand, cardMap } = dealCards(playerOrder, deck, cpr);
  const roleMap = assignRoles(playerOrder);

  auth = {
    phase: 'ROUND',
    round,
    playerOrder,
    currentPlayerIndex: 0,
    players: Object.fromEntries(
      playerOrder.map(id => [id, {
        name: lobbyPlayers.find(p => p.peerId === id).name,
        hand: [...hand[id]],
        revealed: [],
        disconnected: false,
      }])
    ),
    cardMap,
    roleMap,
    bombsRevealed: Object.fromEntries(COLORS.map(c => [c, 0])),
    defuseFound: false,
    revealedThisRound: 0,
    winner: null,
    winReason: null,
  };

  // Envoyer rôle + main à chaque joueur distant (pas à l'hôte)
  for (const peerId of playerOrder) {
    if (peerId === 'host') continue;
    net.send(peerId, { type: 'YOUR_ROLE', role: roleMap[peerId] });
    net.send(peerId, { type: 'YOUR_HAND', cards: hand[peerId].map(id => cardMap[id]) });
  }

  // Hôte récupère ses propres données localement
  gameRole = roleMap['host'];
  myHand = (hand['host'] || []).map(id => cardMap[id]);

  broadcastGameState();
  updateHostView();
}

function buildPublicState() {
  const s = auth;
  return {
    type: 'GAME_STATE',
    phase: s.phase,
    round: s.round,
    playerOrder: s.playerOrder,
    currentPlayer: s.playerOrder[s.currentPlayerIndex],
    players: Object.fromEntries(
      s.playerOrder.map(id => [id, {
        name: s.players[id].name,
        handSize: s.players[id].hand.length,
        revealed: s.players[id].revealed.map(cid => s.cardMap[cid]),
        disconnected: s.players[id].disconnected,
      }])
    ),
    bombsRevealed: { ...s.bombsRevealed },
    defuseFound: s.defuseFound,
    winner: s.winner,
    winReason: s.winReason,
    // Révéler les rôles uniquement en fin de partie
    roleMap: s.phase === 'END' ? s.roleMap : null,
  };
}

function broadcastGameState() {
  const pub = buildPublicState();
  gs = pub;
  net.broadcast(pub);
}

function updateHostView() {
  // Met à jour la main de l'hôte depuis l'état autoritaire
  if (auth) {
    myHand = (auth.players['host']?.hand || []).map(id => auth.cardMap[id]);
    gameRole = auth.roleMap?.['host'] ?? gameRole;
  }
  renderGame();
}

function handleFlipCard(flipperId, targetPeerId, cardId) {
  if (!auth || auth.phase !== 'ROUND') return;

  const currentPlayer = auth.playerOrder[auth.currentPlayerIndex];
  if (flipperId !== currentPlayer) {
    if (flipperId !== 'host') net.send(flipperId, { type: 'ERROR', msg: 'Ce n\'est pas votre tour.' });
    return;
  }
  if (targetPeerId === flipperId) {
    if (flipperId !== 'host') net.send(flipperId, { type: 'ERROR', msg: 'Vous ne pouvez pas couper votre propre fil.' });
    else showToast('Vous ne pouvez pas couper votre propre fil.', 'error');
    return;
  }

  const target = auth.players[targetPeerId];
  if (!target || !target.hand.includes(cardId)) {
    if (flipperId !== 'host') net.send(flipperId, { type: 'ERROR', msg: 'Carte introuvable.' });
    return;
  }

  // Déplacer la carte de la main vers les révélées
  target.hand = target.hand.filter(id => id !== cardId);
  target.revealed.push(cardId);

  const card = auth.cardMap[cardId];
  if (card.type === 'DEFUSE') {
    auth.defuseFound = true;
  } else if (card.type === 'BOMB') {
    auth.bombsRevealed[card.color]++;
  }
  auth.revealedThisRound++;

  // Vérifier victoire
  const win = checkWinConditions();
  if (win) {
    auth.phase = 'END';
    auth.winner = win.team;
    auth.winReason = win.reason;
    broadcastGameState();
    updateHostView();
    return;
  }

  // Fin de manche ?
  if (auth.revealedThisRound >= auth.playerOrder.length) {
    if (auth.round >= MAX_ROUNDS - 1) {
      auth.phase = 'END';
      auth.winner = 'MORIARTY';
      auth.winReason = 'TIME_OUT';
      broadcastGameState();
      updateHostView();
    } else {
      advanceRound();
    }
    return;
  }

  advanceTurn();
  broadcastGameState();
  updateHostView();
}

function advanceTurn() {
  const n = auth.playerOrder.length;
  let next = (auth.currentPlayerIndex + 1) % n;
  for (let i = 0; i < n; i++) {
    if (!auth.players[auth.playerOrder[next]].disconnected) break;
    next = (next + 1) % n;
  }
  auth.currentPlayerIndex = next;
}

function advanceRound() {
  auth.round++;
  auth.revealedThisRound = 0;

  const newDeck = generateDeck(auth.playerOrder.length, auth.round);
  const cpr = CARDS_PER_ROUND[auth.playerOrder.length][auth.round];
  const { hand, cardMap } = dealCards(auth.playerOrder, newDeck, cpr);

  for (const peerId of auth.playerOrder) {
    auth.players[peerId].hand = hand[peerId];
  }
  Object.assign(auth.cardMap, cardMap);

  // Envoyer les nouvelles mains
  for (const peerId of auth.playerOrder) {
    if (peerId === 'host') continue;
    net.send(peerId, { type: 'YOUR_HAND', cards: hand[peerId].map(id => auth.cardMap[id]) });
  }

  broadcastGameState();
  updateHostView();
}

function checkWinConditions() {
  if (auth.defuseFound) return { team: 'SHERLOCK', reason: 'DEFUSE_FOUND' };
  for (const color of COLORS) {
    if (auth.bombsRevealed[color] >= 4) return { team: 'MORIARTY', reason: 'BOMB_EXPLODED' };
  }
  return null;
}

// ─── Rendu du plateau de jeu ─────────────────────────────────────────────────

function renderGame() {
  if (!gs) return;
  if (gs.phase === 'END') { renderEndScreen(); return; }

  const isMyTurn = gs.currentPlayer === myPeerId;
  const roleName = gameRole === 'SHERLOCK' ? '🔍 Sherlock' :
                   gameRole === 'MORIARTY' ? '💀 Moriarty' : '❓';

  const bombStatus = COLORS.map(c => {
    const n = gs.bombsRevealed[c] ?? 0;
    const emoji = { RED: '🔴', BLUE: '🔵', YELLOW: '🟡' }[c];
    return `<span class="bomb-count${n >= 3 ? ' danger' : ''}">${emoji}×${n}</span>`;
  }).join('');

  const otherPlayers = gs.playerOrder.filter(pid => pid !== myPeerId);

  const playerRows = otherPlayers.map(pid => {
    const p = gs.players[pid];
    const hiddenCards = Array.from({ length: p.handSize }, (_, i) => {
      if (isMyTurn && !p.disconnected) {
        return `<div class="card card-back clickable" data-peer="${pid}" data-idx="${i}" title="Couper ce fil">🂠</div>`;
      }
      return `<div class="card card-back">🂠</div>`;
    }).join('');
    const revealedCards = p.revealed.map(card => cardFace(card)).join('');
    return `
      <div class="player-row${p.disconnected ? ' disconnected' : ''}">
        <div class="player-name">${escHtml(p.name)}${p.disconnected ? ' ❌' : ''}</div>
        <div class="cards-row">${hiddenCards}${revealedCards}</div>
      </div>`;
  }).join('');

  const ownCards = myHand.map(() => `<div class="card card-back own">🂠</div>`).join('');
  const currentName = gs.players[gs.currentPlayer]?.name ?? '?';
  const turnMsg = isMyTurn
    ? `<span class="turn-you">⚡ C'EST VOTRE TOUR — cliquez une carte d'un autre joueur</span>`
    : `<span class="turn-wait">⏳ Tour de <strong>${escHtml(currentName)}</strong>…</span>`;

  render(`
    <div class="screen game">
      <div class="game-header">
        <span>Manche ${gs.round + 1}/${MAX_ROUNDS}</span>
        <span>Défuse : ${gs.defuseFound ? '✅' : '✗'}</span>
        <span class="bombs-row">${bombStatus}</span>
      </div>
      <div class="role-badge role-${gameRole === 'SHERLOCK' ? 'sherlock' : gameRole === 'MORIARTY' ? 'moriarty' : 'unknown'}">
        ${roleName}
      </div>
      <div class="players-area">${playerRows}</div>
      <div class="own-hand">
        <div class="section-title">Votre main (non coupable)</div>
        <div class="cards-row">${ownCards}</div>
      </div>
      <div class="turn-indicator">${turnMsg}</div>
    </div>
  `);

  if (isMyTurn) {
    app.querySelectorAll('.card-back.clickable').forEach(el => {
      el.onclick = () => {
        const targetPeerId = el.dataset.peer;
        if (isHost) {
          const cardId = auth.players[targetPeerId]?.hand[0];
          if (cardId) handleFlipCard('host', targetPeerId, cardId);
        } else {
          net.send({ type: 'FLIP_CARD', targetPeerId, cardId: '__first__' });
        }
      };
    });
  }
}

function cardFace(card) {
  if (card.type === 'DEFUSE') return `<div class="card card-defuse" title="Fil désamorçant !">✂️</div>`;
  if (card.type === 'BOMB') {
    const emoji = { RED: '🔴', BLUE: '🔵', YELLOW: '🟡' }[card.color] ?? '💣';
    return `<div class="card card-bomb card-bomb-${card.color.toLowerCase()}" title="Bombe ${card.color}">${emoji}</div>`;
  }
  return `<div class="card card-safe" title="Fil neutre">⚡</div>`;
}

// ─── Écran de fin ─────────────────────────────────────────────────────────────

function renderEndScreen() {
  const sherlockWon = gs.winner === 'SHERLOCK';
  const reasons = {
    DEFUSE_FOUND: 'Le fil désamorçant a été trouvé !',
    BOMB_EXPLODED: 'La bombe a explosé !',
    TIME_OUT: 'Le temps est écoulé — la bombe a explosé !',
  };

  const rolesHtml = gs.roleMap
    ? gs.playerOrder.map(pid => {
        const p = gs.players[pid];
        const r = gs.roleMap[pid];
        return `<li>${escHtml(p.name)} — <strong>${r === 'SHERLOCK' ? '🔍 Sherlock' : '💀 Moriarty'}</strong></li>`;
      }).join('')
    : '<li>Rôles non disponibles</li>';

  render(`
    <div class="screen end ${sherlockWon ? 'end-sherlock' : 'end-moriarty'}">
      <div class="end-result">${sherlockWon ? '🎉 SHERLOCK GAGNE !' : '💥 MORIARTY GAGNE !'}</div>
      <div class="end-reason">${reasons[gs.winReason] ?? ''}</div>
      <div class="end-role">
        Votre rôle : <strong>${gameRole === 'SHERLOCK' ? '🔍 Sherlock' : gameRole === 'MORIARTY' ? '💀 Moriarty' : '?'}</strong>
      </div>
      <div class="section-title" style="margin-top:16px">Tous les rôles</div>
      <ul class="role-list">${rolesHtml}</ul>
      ${isHost ? `<button id="btn-replay" class="btn btn-primary" style="margin-top:20px">🔄 Rejouer</button>` : `<div class="waiting-msg" style="margin-top:16px">En attente d'un nouveau lancement par l'hôte…</div>`}
    </div>
  `);

  if (isHost) {
    $('#btn-replay').onclick = startGame;
  }
}

// ─── Init ─────────────────────────────────────────────────────────────────────

showHome();

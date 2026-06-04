// Génération du deck Time Bomb et attribution des rôles.

export const COLORS = ['RED', 'BLUE', 'YELLOW'];

// Cartes par joueur selon la manche (index 0–3)
export const CARDS_PER_ROUND = {
  4: [5, 4, 3, 2],
  5: [4, 3, 2, 1],
  6: [4, 3, 2, 1],
  7: [3, 3, 2, 1],
  8: [3, 2, 2, 1],
};

// Nombre de Moriarty par effectif
export const MORIARTY_COUNT = { 4: 1, 5: 1, 6: 2, 7: 2, 8: 3 };

// Nombre de manches maximum
export const MAX_ROUNDS = 4;

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

let _cardSeq = 0;
function nextId() {
  return `c${String(_cardSeq++).padStart(3, '0')}`;
}

/**
 * Génère un deck mélangé pour la manche donnée.
 * Taille = playerCount × CARDS_PER_ROUND[playerCount][round].
 * Contient toujours exactement 1 fil désamorçant (DEFUSE).
 * Les bombes (BOMB) sont réparties entre 3 couleurs (4 max par couleur).
 * Le reste est des fils neutres (SAFE).
 */
export function generateDeck(playerCount, round) {
  _cardSeq = 0;
  const total = playerCount * CARDS_PER_ROUND[playerCount][round];

  // Nombre de bombes : entre 3 et 12, adapté à la taille du deck
  const bombTotal = Math.min(Math.max(Math.floor(total * 0.3), 3), 12);
  const safeTotal = total - 1 - bombTotal; // -1 pour le fil désamorçant

  const cards = [];

  // 1 fil désamorçant
  cards.push({ id: nextId(), type: 'DEFUSE', color: null });

  // Bombes réparties équitablement entre 3 couleurs (max 4 par couleur)
  const perColor = Math.floor(bombTotal / COLORS.length);
  const extra = bombTotal % COLORS.length;
  COLORS.forEach((color, i) => {
    const count = perColor + (i < extra ? 1 : 0);
    for (let k = 0; k < count; k++) {
      cards.push({ id: nextId(), type: 'BOMB', color });
    }
  });

  // Fils neutres
  for (let i = 0; i < safeTotal; i++) {
    cards.push({ id: nextId(), type: 'SAFE', color: null });
  }

  return shuffle(cards);
}

/**
 * Distribue les cartes aux joueurs.
 * Retourne { hand: Map<peerId, cardObj[]>, cardMap: Map<id, cardObj> }
 */
export function dealCards(playerOrder, deck, cardsPerPlayer) {
  const hand = {};
  const cardMap = {};
  let idx = 0;

  for (const peerId of playerOrder) {
    hand[peerId] = [];
    for (let i = 0; i < cardsPerPlayer; i++) {
      const card = deck[idx++];
      hand[peerId].push(card.id);
      cardMap[card.id] = card;
    }
  }

  return { hand, cardMap };
}

/**
 * Attribue les rôles SHERLOCK / MORIARTY aux joueurs.
 * Retourne { [peerId]: 'SHERLOCK' | 'MORIARTY' }
 */
export function assignRoles(playerIds) {
  const n = playerIds.length;
  const moriartyCount = MORIARTY_COUNT[n] ?? 1;
  const roles = shuffle([
    ...Array(moriartyCount).fill('MORIARTY'),
    ...Array(n - moriartyCount).fill('SHERLOCK'),
  ]);
  return Object.fromEntries(playerIds.map((id, i) => [id, roles[i]]));
}

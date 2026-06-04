<div align="center">

# ⚙ WorkForge

**Ta boîte à outils personnelle pour Chrome.**

![Manifest V3](https://img.shields.io/badge/Manifest-V3-7c6dfa?style=flat-square)
![Chrome](https://img.shields.io/badge/Chrome-Extension-yellow?style=flat-square&logo=googlechrome&logoColor=white)
![Made for me](https://img.shields.io/badge/Made-For%20Me-22c55e?style=flat-square)

</div>

---

WorkForge regroupe dans une seule extension tous les petits outils qui manquent au quotidien. Un tableau de bord minimaliste, chaque outil activable indépendamment, et une architecture pensée pour en ajouter facilement.

---

## Outils

| | Outil | Description | Statut |
|---|---|---|---|
| 💬 | **Teams Disponible** | Maintient ton statut sur Disponible sur `teams.cloud.microsoft` | ✅ Dispo |

---

## Installation

> Pas de Web Store — chargement en mode développeur.

```bash
# 1. Récupérer le projet
git clone <url-du-repo> && cd ext-works

# 2. Générer les icônes (une seule fois)
npm install && node gen-icons.js
```

Ensuite dans Chrome :

1. Ouvre **`chrome://extensions`**
2. Active le **Mode développeur** en haut à droite
3. Clique **"Charger l'extension non empaquetée"**
4. Sélectionne le dossier `ext-works`

Épingle l'icône dans ta barre d'outils et c'est parti.

---

## Structure

```
ext-works/
├── manifest.json        ← Manifest V3, permissions, déclarations
├── background.js        ← Service worker (alarmes, scripts injectés)
├── popup.html/css/js    ← Interface du tableau de bord
├── tools/
│   └── registry.js      ← Registre des outils + UI de config de chacun
├── icons/               ← PNG 16/48/128px (générés)
└── gen-icons.js         ← Générateur d'icônes Node.js
```

---

## Ajouter un outil

Deux fichiers à toucher, c'est tout.

**`tools/registry.js`** — déclarer l'outil et son UI :

```js
{
  id: 'mon-outil',
  name: 'Mon Outil',
  desc: 'Ce que ça fait en une ligne',
  icon: '🔧',
  defaultState: { enabled: false },

  renderDetail(container, state, save) {
    // HTML de la page de configuration de l'outil
    container.innerHTML = `<div class="info-box">Mes paramètres.</div>`;
  },
}
```

**`background.js`** — brancher la logique :

```js
// Dans chrome.runtime.onMessage
if (msg.toolId === 'mon-outil') {
  if (msg.action === 'tool-start') monOutilStart();
  if (msg.action === 'tool-stop')  monOutilStop();
}

// Dans restoreTools() pour survivre aux redémarrages
const state = await getToolState('mon-outil');
if (state.enabled) monOutilStart();
```

La carte s'affiche automatiquement dans le tableau de bord.

---

## Dev

Après chaque modif : bouton **↺** sur `chrome://extensions`.

Logs du service worker : `chrome://extensions` → **Inspecter les vues › Service Worker**.

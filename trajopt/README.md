# TrajOpt — Prédiction & Optimisation de Trajectoire

Système avancé de prédiction et d'optimisation de trajectoire basé sur l'apprentissage par renforcement.

## Installation

### Prérequis

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)

### Installation avec Poetry

```bash
# Cloner le repo et naviguer dans le dossier
cd trajopt

# Installer les dépendances
poetry install

# Activer l'environnement virtuel
poetry shell
```

### Commandes utiles

```bash
# Ajouter une dépendance
poetry add <package>

# Ajouter une dépendance de dev
poetry add --group dev <package>

# Mettre à jour les dépendances
poetry update

# Lancer les tests
poetry run pytest

# Lancer le linter
poetry run ruff check .
```

## Circuits d'entraînement

La bibliothèque `utils/track_library.py` décrit douze circuits, du plus simple
au plus exigeant. Chaque tracé est écrit comme une **suite de lignes droites et
de virages** (rayon, angle, sens), la façon dont les circuits sont publiés : les
rayons de courbe et les longueurs de ligne droite sont donc réels, et c'est la
limite d'adhérence du moteur physique qui fixe la vitesse au point de corde.

| Circuit | Longueur | Virages | Rayon min. | Largeur | Tags |
|---|---|---|---|---|---|
| `indianapolis_oval` | 4022 m | 4 | 256 m | 15,0 m | oval, high-speed, easy |
| `brands_hatch_indy` | 1929 m | 8 | 14 m | 11,0 m | short, hairpin, medium |
| `red_bull_ring` | 4318 m | 10 | 28 m | 13,0 m | permanent, high-speed, easy |
| `monza` | 5794 m | 11 | 28 m | 13,0 m | permanent, high-speed, chicane |
| `spa_francorchamps` | 7004 m | 19 | 22 m | 13,0 m | permanent, high-speed, long |
| `silverstone` | 5891 m | 17 | 25 m | 13,0 m | permanent, high-speed, medium |
| `catalunya` | 4675 m | 14 | 28 m | 13,0 m | permanent, technical, medium |
| `hockenheim` | 4573 m | 13 | 20 m | 13,0 m | permanent, hairpin, medium |
| `zandvoort` | 4258 m | 13 | 22 m | 12,0 m | permanent, narrow, hard |
| `hungaroring` | 4381 m | 14 | 25 m | 12,0 m | permanent, low-speed, hard |
| `interlagos` | 4309 m | 16 | 28 m | 12,0 m | technical, anticlockwise, hard |
| `monaco_street` | 3255 m | 17 | 10 m | 9,0 m | street, narrow, hairpin, hard |

### Ce que ces tracés sont, et ne sont pas

- Ce sont des **reconstructions schématiques** : la séquence de virages, leurs
  rayons, leur sens et les longueurs de ligne droite viennent de la géométrie
  publiée, et la longueur du tour est à moins de 1 % de la longueur réelle. Le
  plan est ressemblant, pas relevé au géomètre.
- La largeur est **uniforme** par circuit (`Track` ne modélise pas de largeur
  variable) ; la valeur retenue est représentative du circuit réel.
- Les circuits en **huit** (Suzuka) sont absents : leur plan se croise, alors que
  la surface roulable est construite comme un anneau (contour extérieur, bord
  intérieur en trou).
- `monaco_street` est un tracé **original** dans le caractère de Monaco, pas une
  reconstruction : le plan réel repasse à moins de 7 m de la ligne droite des
  stands, ce qui fusionne les deux brins sur une piste de 9 m de large. Ce qui
  est conservé, c'est le problème de conduite — 9 m de large, dix-sept virages
  lents, une épingle de 10 m de rayon, une chicane.

### Génération

Les fichiers JSON sont un produit de build, pas une source versionnée :

```bash
poetry run python -m utils.track_library --out data/tracks
```

`mode=optimize` les régénère de lui-même s'ils manquent.

## Entraînement multi-circuits

Un seul modèle est entraîné sur un **ensemble** de circuits : chaque worker
détient tout l'ensemble et **tire un circuit à chaque épisode**, de sorte que la
politique s'ajuste à une distribution de tracés plutôt qu'à un tour appris par
cœur. La ligne optimale, elle, est propre au circuit : l'export produit un
`data/optimal_traj_<circuit>.npy` par tracé.

```bash
# Tous les circuits de la bibliothèque (défaut)
poetry run python main.py

# Progression : d'abord les circuits rapides et larges
poetry run python main.py paths.tracks=easy
poetry run python main.py paths.tracks=hard

# Une sélection explicite, ou un seul circuit comme avant
poetry run python main.py "paths.tracks=[monza,spa_francorchamps]"
poetry run python main.py paths.tracks=null   # uniquement paths.track
```

Deux réglages suivent désormais la longueur du circuit au lieu d'être fixés
(`conf/env/default.yaml`) :

- `max_steps: null` — le budget d'épisode est exprimé en tours
  (`max_steps_lap_factor`, 3 tours au rythme de référence). Les 1500 pas réglés
  sur l'ovale de 607 m ne couvrent que 2 km des 7 km de Spa : chaque épisode y
  serait tronqué au tiers du tour.
- `num_checkpoints: null` — un checkpoint tous les `checkpoint_spacing` mètres
  (150 m). Quatre checkpoints font 150 m d'écart sur l'ovale mais 1,75 km à Spa,
  bien au-delà de l'horizon de `gamma` (0,995, soit ~10 s, ~250 m).

À noter : sur les circuits longs, le bonus de tour complet reste un événement
terminal et son poids relatif diminue (15 points contre 700 de récompense de
progression à Spa). C'est `gamma` qu'il faut augmenter pour qu'il porte plus
loin, pas le barème.

## Structure du projet

```
trajopt/
├── configs/        # Fichiers de configuration
├── data/           # Données (tracks, trajectoires)
├── envs/           # Environnements Gymnasium
├── models/         # Modèles de prédiction
├── physics/        # Moteur physique
├── rl/             # Algorithmes RL
├── telemetry/      # Logs et métriques
└── utils/          # Utilitaires (dont track_library.py : les circuits)
```

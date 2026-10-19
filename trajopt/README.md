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

## Modèle véhicule

`physics/physics_engine.py` est un modèle bicyclette plan : deux essieux, et
pour chacun un seul budget d'adhérence, partagé entre force longitudinale et
force latérale (cercle de friction). Il représente :

- la **propulsion** : tout le couple va à l'essieu arrière ;
- le **transfert de charge longitudinal** `m·a·h/L`, avec un pas de retard,
  le temps que la caisse plonge ;
- la **sensibilité à la charge** : le μ d'un essieu baisse quand sa charge
  dépasse sa charge statique (appui aérodynamique, transfert) ;
- l'**ABS et l'antipatinage** : aucun essieu ne reçoit une demande
  longitudinale supérieure à son adhérence ;
- une **boîte** qui engage le rapport donnant le plus de couple à la roue, et
  un **rupteur** qui coupe le moteur sans toucher à la vitesse.

Il ne représente pas le transfert de charge latéral, l'aéro sensible à
l'assiette, la dynamique de direction ni le temps de passage des rapports. La
sensibilité à la charge est donc un paramètre effectif, calé sur des mesures.

### Calibration GT3

La voiture par défaut (`conf/vehicle/default.yaml`) est une GT3 actuelle
(homologations 2019-2026). Aucune GT3 ne publie son aéro, son centre de
gravité ni sa courbe de couple : ce qui n'est pas sourcé est estimé, et
signalé comme tel dans le fichier.

| Grandeur | Valeur | Origine |
|---|---|---|
| Masse en course | 1 420 kg | BoP 2025, médiane 1 315 kg [1][2], + pilote 85 kg + 30 l |
| Empattement, voie | 2,66 m, 1,70 m | médiane de sept GT3 ; Ferrari 296, Huracán EVO2 |
| Moteur | 4,2 l, ≈ 400 kW, rupteur 9 250 tr/min | Porsche 911 GT3 R (992) [3], ramené au niveau BoP (estim.) |
| Rapports | 3,154 à 1,097, pont 3,789 | Porsche 911 GT3 R (991.2), seule boîte GT3 publiée [4] |
| Pneu arrière | rayon en charge 0,35 m | Michelin 31/71-18, Pirelli 325/705-18 |
| Aéro | SCx 1,1 m², SCz 2,8 m², 40 % avant | SCx : vitesses de pointe [5] ; SCz : 870 kg à 250 km/h pour la 296 Challenge [6] (estim.) |
| Adhérence | μ0 1,55, −22 % par doublement de charge | freinage Brembo [7], ≈ 1,6 g en virage lent (estim.) |
| CdG, répartition | 0,40 m, 45/55 | estim. |

`tests/test_gt3_envelope.py` vérifie que la voiture reste dans l'enveloppe
mesurée :

| Manœuvre | Modèle | Référence |
|---|---|---|
| Vitesse de pointe | 292 km/h | 284-304 km/h, LMGT3 au Mans 2024-2026 [5] |
| 0-200 km/h | 9,0 s | 10,2 s pour la 992 GT3 de route [8] |
| Freinage à 290 km/h | 2,46 g | 2,0-2,5 g, IMSA GTD 2025 [7] |
| Virage à 90 km/h | 1,60 g | ≈ 1,6 g (estim.) |
| Virage à 230 km/h | 1,99 g | 2 à 2,5 g (estim.) |

Le freinage et le virage rapide tirent la sensibilité à la charge en sens
contraires ; elle est calée sur le freinage, qui est mesuré, et le virage
rapide sort à la limite basse de son estimation.

1. [SRO, BoP GT World Challenge America, Austin 2025](https://www.gt-world-challenge-america.com/documents/notice/1438/f8ff82bd4e7af413592def5ffa0eb6.pdf)
2. [FIA, BoP GT3 de la FIA GT World Cup 2025](https://api.fia.com/system/files/documents/2025-11-14_fia_bop_gt3_gtwc_v4_0.pdf)
3. [Porsche, dossier de presse de la 911 GT3 R (992)](https://newsroom.porsche.com/pdf/74b6d8d9-3614-45c0-b195-81b6bb85aee9)
4. [Fiche technique de la Porsche 911 GT3 R (991)](https://www.theracersgroup.com/racecars/porsche-911-gt3-r-991/)
5. [FIA WEC, vitesses de pointe des 24 Heures du Mans 2024](https://fiawec.alkamelsystems.com/Results/13_2024/04_LE%20MANS/541_FIA%20WEC/202406151600_Race/24_Hour%2024/14_BestTopSpeed_Race_Hour%2024.PDF)
6. [Motor Sport, aéro de la Ferrari 296 Challenge](https://www.motorsportmagazine.com/special-article/race-to-road-ferrari/90/ferrari-296-challenge-how-its-aero-and-electronics-make-690bhp-approachable/)
7. [Brembo, données de freinage IMSA à Sebring](https://www.brembo.com/en/motorsport/wec/brembo-brake-facts-sebring)
8. [FastestLaps, Porsche 911 GT3 (992)](https://fastestlaps.com/tests/c5gu9ulf5bum)

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

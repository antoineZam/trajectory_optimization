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
└── utils/          # Utilitaires
```

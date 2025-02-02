# Prédiction des Montants de Retraits avec GRU et MLP

Ce projet implémente un modèle de prédiction des montants de retraits à partir de données historiques. Il utilise des modèles de type **GRU (Gated Recurrent Unit)** et **MLP (Multi-Layer Perceptron)** pour effectuer les prévisions.

## Installation

### Prérequis
Assurez-vous d'avoir **Python 3.8+** installé sur votre machine.

### Installation des dépendances
Installez les dépendances nécessaires avec la commande suivante :

```bash
pip install -r requirements.txt
```

## Structure du projet

```
.
├── modules
│   ├── predict.py          # Contient la classe de prédiction
│   ├── main_script.py      # Contient le script principal d'entraînement
├── script
│   ├── predict.py          # Script pour lancer la prédiction
│   ├── train.py            # Script pour l'entraînement du modèle
├── best_results
│   ├── best_config.json    # Fichier contenant les meilleurs hyperparamètres
│   ├── weight_saved.pt     # Poids du modèle entraîné
├── dataset.csv             # Jeu de données initial
├── augmented_dataset.csv   # Jeu de données pré-traité
├── requirements.txt        # Liste des dépendances Python
├── README.md               # Documentation du projet
```

## Utilisation

### Entraînement du modèle
Pour entraîner le modèle, exécutez :

```bash
python script/train.py
```

Cela chargera les données et entraînera un modèle GRU ou MLP en fonction des hyperparamètres définis dans le fichier de configuration.

### Prédiction
Une fois l'entraînement terminé, vous pouvez générer des prédictions en exécutant :

```bash
python script/predict.py
```

Cela prédira les montants de retraits pour une date donnée et enregistrera les résultats dans un fichier CSV.

## Modèles utilisés
Le projet utilise **deux types de modèles** :
- **GRU (Gated Recurrent Unit)** : Un type de réseau de neurones récurrents efficace pour les séries temporelles.
- **MLP (Multi-Layer Perceptron)** : Un réseau de neurones entièrement connecté, utile pour les données tabulaires.

## Configuration
Les hyperparamètres des modèles sont définis dans `best_results/best_config.json`. Exemple de configuration :

```json
{
    "input_size": 17,
    "num_epochs": 200,
    "model_type": "GRU",
    "hidden_size": 30,
    "num_layers": 2,
    "learning_rate": 0.001,
    "batch_size": 32
}
```




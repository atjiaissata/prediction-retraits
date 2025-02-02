from ray import tune
from modules.bruit import ajout_bruit
from modules.training import Entrainement
import pandas as pd 
import ray
import json
import torch
import os


def short_trial_name(trial):
    """Générer un nom de répertoire court pour les essais"""
    return f"trial_{trial.trial_id}"

def main(data: pd.DataFrame, num_sample, config):


    X = data.drop(columns=["MNT_RETRAIT"])
    y = data["MNT_RETRAIT"]

    data_final = ajout_bruit(data)
    X_final = data_final.drop(columns=["MNT_RETRAIT"])
    y_final = data_final["MNT_RETRAIT"]

    entrain = Entrainement(X_final, y_final)
    train_model = entrain.train

    result = tune.run(train_model, config=config, num_samples=num_sample, metric="val_mae", mode="min",
                       trial_dirname_creator=short_trial_name)

    best_config = result.get_best_config()

    save_dir = "best_results"
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/best_config.json', 'w') as json_file:
        json.dump(best_config, json_file, indent=4)
        print(f"Best config saved to best_config_.json")

    best_trial = result.get_best_trial(metric="val_mae", mode="min")
    best_checkpoint = best_trial.checkpoint  
    best_checkpoint_dir = best_checkpoint.to_directory()  
    checkpoint_path = os.path.join(best_checkpoint_dir, "checkpoint.pt")  
    checkpoint_data = torch.load(checkpoint_path, weights_only=True)

    save_path = os.path.join(save_dir, "weight_saved.pt")

    torch.save(checkpoint_data, save_path)
    print(f"Checkpoint saved to {save_path}")

    
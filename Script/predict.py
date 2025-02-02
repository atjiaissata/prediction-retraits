from modules.predict import Predictor
import pandas as pd


if __name__=="__main__":
    data = pd.read_csv("../augmented_dataset.csv", sep=",")

    config = {
    "date_pred": "30/03/2023", 
    "best_config_path": "./best_results/best_config.json",
    "model_path":"./best_results/weight_saved.pt"
    }

    pred = Predictor(config)
    pred.predict_multiple_days(data)
    

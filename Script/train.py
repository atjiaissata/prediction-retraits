from modules.main_script import main
import pandas as pd
from ray import tune

if __name__=="__main__":

    data = pd.read_csv("../dataset.csv", sep=";")
    data =data[["DATE_HISTO","INDEX_DAY_WEEK","INDEX_WEEK_MONTH","INDEX_WEEK_YEAR","IS_PENSION_DAY","IS_SALARY_DAY","IS_WEEKEND_DAY","VAL_DD","VAL_MM","VAL_YYYY","REF_ID_ATM","MNT_RETRAIT"]]
    config = {

    "input_size": 17,
    "num_epochs": 200,
    "model_type": "GRU",
    "dataset_id": "2",
    "hidden_size": tune.choice([20, 30, 40, 10]),  
    "num_layers": tune.choice([1, 2, 3, 4]),  
    "learning_rate": tune.loguniform(1e-5, 1e-3),  
    "weight_decay": tune.loguniform(1e-6, 1e-3),  
    "batch_size": tune.choice([16, 32, 64]),
    "date_pred": "05/02/2018", 
    "best_config_path": "../best_results/best_config.json",
    "model_path":"../best_results/weight_saved.pt"
    }
    

    main(data, num_sample=3, config=config)

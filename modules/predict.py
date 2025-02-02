import torch
import numpy as np
import pandas as pd
import json
from modules.models import Regressor
from sklearn.preprocessing import MinMaxScaler

class Predictor:
    def __init__(self, config):
        self.config = config
        self.model = self.load_model()

    def week_of_month(self, date):
        """Calcule la semaine du mois pour une date donnée."""
        first_day = date.replace(day=1)
        return (date.day + first_day.weekday()) // 7 + 1

    def create_features_for_date(self, date, augmented_data):
        """Crée les features nécessaires pour la prédiction à partir d'une date donnée."""
        augmented_data['DATE_HISTO'] = pd.to_datetime(augmented_data['DATE_HISTO'])
        new_data = pd.DataFrame({'DATE_HISTO': [date]})

        new_data['INDEX_DAY_WEEK'] = new_data['DATE_HISTO'].dt.weekday
        new_data['INDEX_WEEK_MONTH'] = new_data['DATE_HISTO'].apply(self.week_of_month)
        new_data['INDEX_WEEK_YEAR'] = new_data['DATE_HISTO'].dt.isocalendar().week
        new_data['IS_PENSION_DAY'] = new_data['DATE_HISTO'].dt.day.isin([19, 20, 21]).astype(int)
        new_data['IS_SALARY_DAY'] = new_data['DATE_HISTO'].dt.day.isin([1, 2, 3, 4, 5, 29, 30, 31]).astype(int)
        new_data['IS_WEEKEND_DAY'] = new_data['INDEX_DAY_WEEK'].isin([5, 6]).astype(int)
        new_data['VAL_DD'] = new_data['DATE_HISTO'].dt.day
        new_data['VAL_MM'] = new_data['DATE_HISTO'].dt.month
        new_data['VAL_YYYY'] = new_data['DATE_HISTO'].dt.year
        new_data['REF_ID_ATM'] = 116
        new_data['MNT_RETRAIT'] = (0.5 * augmented_data['MNT_RETRAIT'].shift(1).iloc[-1] +
                                   0.3 * augmented_data['MNT_RETRAIT'].shift(2).iloc[-1] +
                                   0.2 * augmented_data['MNT_RETRAIT'].shift(3).iloc[-1])
        new_data['MOVING_AVERAGE_3D'] = augmented_data['MNT_RETRAIT'].rolling(window=3, closed='left').mean().iloc[-1]
        new_data['MOVING_AVERAGE_7D'] = augmented_data['MNT_RETRAIT'].rolling(window=7, closed='left').mean().iloc[-1]
        new_data['MOVING_STD_7D'] = augmented_data['MNT_RETRAIT'].rolling(window=7, closed='left').std().iloc[-1]
        new_data['WITHDRAWAL_RATIO_3D'] = new_data['MNT_RETRAIT'] / new_data['MOVING_AVERAGE_3D']
        new_data['WITHDRAWAL_RATIO_7D'] = new_data['MNT_RETRAIT'] / new_data['MOVING_AVERAGE_7D']
        new_data['IS_WITHDRAWAL_SPIKE'] = (new_data['MNT_RETRAIT'] > new_data['MOVING_AVERAGE_7D'] * 1.5).astype(int)
        new_data['SUM_WITHDRAWAL_7D'] = augmented_data['MNT_RETRAIT'].rolling(window=7, closed='left').sum().iloc[-1]
        return new_data

    def get_num_days_from_input(self, date_str, augmented_data):
        """Calcule le nombre de jours entre la dernière date des données et la date d'entrée."""
        input_date = pd.to_datetime(date_str, format='%d/%m/%Y')
        last_date = augmented_data["DATE_HISTO"].iloc[-1]
        num_days = (input_date - last_date).days
        return num_days, last_date

    def load_model(self):
        """Charge le modèle de prédiction à partir du fichier de configuration."""
        with open(self.config["best_config_path"], 'r') as f:
            config = json.load(f)
        model = Regressor(model_type=config["model_type"], input_size=config["input_size"],
                          hidden_size=config["hidden_size"], num_layers=config["num_layers"])
        model_state, _ = torch.load(self.config["model_path"], weights_only=True)
        model.load_state_dict(model_state)
        model.eval()
        return model

    def predict_multiple_days(self, historical_data):
        """Effectue les prédictions pour plusieurs jours en fonction des données historiques."""
        num_days, start_date = self.get_num_days_from_input(self.config["date_pred"], historical_data)
        predictions = []
        augmented_data = historical_data.copy()
        current_date = pd.to_datetime(start_date)
        scaler_X = MinMaxScaler()
        scaler_X.fit(augmented_data.drop(columns=['MNT_RETRAIT', 'DATE_HISTO']))
        scaler_y = MinMaxScaler()
        scaler_y.fit(augmented_data[['MNT_RETRAIT']])
        for _ in range(num_days):
            current_date += pd.Timedelta(days=1)
            features = self.create_features_for_date(current_date, augmented_data)
            input_features = features.drop(columns=['MNT_RETRAIT', 'DATE_HISTO'])
            input_features_normalized = scaler_X.transform(input_features)
            input_tensor = torch.tensor(input_features_normalized, dtype=torch.float32).reshape(1, 1, -1)
            with torch.no_grad():
                predicted_withdrawal = self.model(input_tensor).squeeze().item()
            predicted_withdrawal = scaler_y.inverse_transform([[predicted_withdrawal]])[0][0]
            predictions.append((current_date, predicted_withdrawal))
            features['DATE_HISTO'] = current_date
            features['MNT_RETRAIT'] = predicted_withdrawal
            augmented_data = pd.concat([augmented_data, features], ignore_index=True)
        augmented_data.to_csv("augmented_dataset_pred.csv", index=True)
        for date, prediction in predictions:
            print(f'date: {date} prediction: {prediction}')
        return predictions
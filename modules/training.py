import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from ray.air import  session
import os
import tempfile
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import r2_score, mean_absolute_error
from modules.models import Regressor
from ray.train import Checkpoint
import matplotlib.pyplot as plt
import json
from datetime import datetime

class Entrainement:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def transform(self):
        scaler = MinMaxScaler()
        X_normalize = scaler.fit_transform(self.X)
        y_normalize = scaler.fit_transform(self.y.values.reshape(-1, 1)).flatten()

        X_train, X_val, y_train, y_val = train_test_split(X_normalize, y_normalize, test_size=0.2, random_state=42, shuffle=False)

        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        return X_train, X_val, y_train, y_val
    
    def train(self, config):
        
        input_size = config["input_size"]
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        learning_rate = config["learning_rate"]
        weight_decay = config["weight_decay"]
        num_epochs = config["num_epochs"]
        batch_size = config["batch_size"]
        model_type = config["model_type"]

        X_train, X_val, y_train, y_val = self.transform()

        model = Regressor(model_type=model_type, input_size= input_size, hidden_size=hidden_size, num_layers=num_layers)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Entraînement
        for epoch in range(num_epochs):
            model.train()
            for i in range(0, len(X_train), batch_size):
                x_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

          
                outputs = model(x_batch).squeeze()
                loss = criterion(outputs, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train).squeeze()
                train_loss = criterion(train_outputs, y_train).item()
                train_r2 = r2_score(y_train.numpy(), train_outputs.numpy())
                train_mae = mean_absolute_error(y_train.numpy(), train_outputs.numpy())

                val_outputs = model(X_val).squeeze()
                val_loss = criterion(val_outputs, y_val).item()
                val_r2 = r2_score(y_val.numpy(), val_outputs.numpy())
                val_mae = mean_absolute_error(y_val.numpy(), val_outputs.numpy())


                session.report({
                        "loss_train": train_loss,
                        "train_mae": train_mae,
                        "train_r2": train_r2,
                        "loss_val": val_loss,
                        "val_mae": val_mae,
                        "val_r2": val_r2,
                        
                    })
                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                    torch.save(
                        (model.state_dict(), optimizer.state_dict()), path
                    )
                    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                    session.report({
                            "loss_train": train_loss,
                            "train_mae": train_mae,
                            "train_r2": train_r2,
                            "loss_val": val_loss,
                            "val_mae": val_mae,
                            "val_r2": val_r2,
                            
                        }, checkpoint=checkpoint)
        
        self.plot_predictions(model, X_val, y_val)

    def plot_predictions(self, model, X_val, y_val):
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val).squeeze().numpy()
            y_val_numpy = y_val.numpy()

            plt.figure(figsize=(12, 6))
            plt.plot(y_val_numpy, label='Valeurs réelles', marker='o')
            plt.plot(val_predictions, label='Prédictions', marker='x')
            plt.title('Comparaison des Prédictions et des Valeurs Réelles')
            plt.xlabel('Échantillons')
            plt.ylabel('Montant des retraits normalisé')
            plt.legend()
            plt.grid(True)
            if not os.path.exists('C:/Users/Utilisateur/Documents/Projet_mo/images'):
                os.makedirs('C:/Users/Utilisateur/Documents/Projet_mo/images')

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_filename = f"predictions_comparaison_{timestamp}.png"
            image_path = os.path.join('C:/Users/Utilisateur/Documents/Projet_mo/images', image_filename)

            plt.savefig(image_path)

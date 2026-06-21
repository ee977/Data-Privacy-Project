import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

import input_generator
import other_models
import model as main_model
import test

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
BASE_DIR = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = os.path.join(BASE_DIR, 'input')
FIGURES_DIR = os.path.join(BASE_DIR, 'Figures')
MODELS_DIR = os.path.join(BASE_DIR, 'Models')

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

FORCE_RETRAIN = True

def create_sequence_dataset(X, y, seq_len=10):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i: i + seq_len])
        ys.append(y[i + seq_len])
    return (
        torch.tensor(np.array(Xs), dtype=torch.float32),
        torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(1),
    )

def main():
    tickers = ["SISE.IS", "KCHOL.IS", "MGROS.IS", "THYAO.IS", "TSKB.IS"]
    client_train_loaders = []
    client_test_loaders_dict = {}
    scalers_y_dict = {}
    test_dfs_dict = {}
    all_train_X, all_train_y = [], []
    seq_length = 10
    batch_size = 32
    input_dim = 0

    for ticker in tickers:
        df = input_generator.generate_inputs(target_ticker=ticker, input_dir=INPUT_DIR, start_date="2010-01-04")
        exclude_cols = [c for c in df.columns if "Close" in c or "Support" in c or "Resistance" in c]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        target_col = "TARGET_Target_Return"
        input_dim = len(feature_cols)

        train_df = df.loc[df.index.year <= 2024].copy()
        test_df = df.loc[df.index.year >= 2025].copy()
        test_dfs_dict[ticker] = test_df

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        train_X_scaled = scaler_X.fit_transform(train_df[feature_cols])
        train_y_scaled = scaler_y.fit_transform(train_df[[target_col]])
        test_X_scaled = scaler_X.transform(test_df[feature_cols])
        test_y_scaled = scaler_y.transform(test_df[[target_col]])
        scalers_y_dict[ticker] = scaler_y

        X_tr_seq, y_tr_seq = create_sequence_dataset(train_X_scaled, train_y_scaled, seq_len=seq_length)
        X_te_seq, y_te_seq = create_sequence_dataset(test_X_scaled, test_y_scaled, seq_len=seq_length)

        client_train_loaders.append(DataLoader(TensorDataset(X_tr_seq, y_tr_seq), batch_size=batch_size, shuffle=True))
        client_test_loaders_dict[ticker] = DataLoader(TensorDataset(X_te_seq, y_te_seq), batch_size=batch_size, shuffle=False)
        all_train_X.append(X_tr_seq)
        all_train_y.append(y_tr_seq)

    combined_X = torch.cat(all_train_X, dim=0)
    combined_y = torch.cat(all_train_y, dim=0)
    central_train_loader = DataLoader(TensorDataset(combined_X, combined_y), batch_size=batch_size, shuffle=True)

    models = {
        "1_SVM_Centralized":                  other_models.SVMModel(),
        "2_LSTM_Centralized":                 other_models.SimpleLSTM(input_dim),
        "3_MLP-LSTM_Centralized":             other_models.MLPLSTM(input_dim),
        "4_Transformer-LSTM_Centralized":     other_models.TransformerLSTM(input_dim),
        "5_MLP-Transformer-LSTM_Centralized": main_model.MLPLSTMTransformer(input_dim),
        "6_FED-MLP-Transformer-LSTM":         main_model.MLPLSTMTransformer(input_dim),
    }

    central_epochs = 30

    for name in models.keys():
        model_path = os.path.join(MODELS_DIR, f"{name}.pth" if "SVM" not in name else f"{name}.pkl")

        if os.path.exists(model_path) and not FORCE_RETRAIN:
            if "SVM" in name:
                models[name].load(model_path)
            else:
                models[name].load_state_dict(torch.load(model_path, map_location=device))
        else:
            if "SVM" in name:
                models[name] = other_models.train_centralized(models[name], central_train_loader, epochs=1, lr=0.001, device=device)
                models[name].save(model_path)
            elif "FED" in name:
                models[name] = other_models.train_federated(models[name], client_train_loaders, rounds=15, local_epochs=10, lr=0.001, device=device, mu=0.01)
                torch.save(models[name].state_dict(), model_path)
            else:
                models[name] = other_models.train_centralized(models[name], central_train_loader, epochs=central_epochs, lr=0.001, device=device)
                torch.save(models[name].state_dict(), model_path)

    test.evaluate_all(
        models_dict=models,
        test_loaders_dict=client_test_loaders_dict,
        scalers_y_dict=scalers_y_dict,
        test_dfs_dict=test_dfs_dict,
        seq_length=seq_length,
        device=device,
        figures_dir=FIGURES_DIR,
    )

if __name__ == "__main__":
    main()
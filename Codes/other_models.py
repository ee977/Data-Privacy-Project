import copy
import torch
import torch.nn as nn
from sklearn.svm import LinearSVR
import numpy as np
import warnings
import joblib

from model import DualHeadLoss

warnings.filterwarnings("ignore")

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)

class SVMModel:
    def __init__(self):
        self.model = LinearSVR(C=0.5, max_iter=5000)

    def train_model(self, dataloader):
        X_list, y_list = [], []
        for X_batch, y_batch in dataloader:
            X_list.append(X_batch.numpy())
            y_list.append(y_batch.numpy())
        X_train = np.concatenate(X_list, axis=0)
        y_train = np.concatenate(y_list, axis=0).flatten()
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_train_flat, y_train)

    def __call__(self, X_tensor):
        X_flat = X_tensor.cpu().numpy().reshape(X_tensor.shape[0], -1)
        preds = self.model.predict(X_flat)
        return torch.tensor(preds, dtype=torch.float32).unsqueeze(1)

    def save(self, path): joblib.dump(self.model, path)
    def load(self, path): self.model = joblib.load(path)
    def eval(self): pass
    def to(self, device): pass

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim: int):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=128, num_layers=3, batch_first=True, dropout=0.3)
        self.reg_head = nn.Sequential(
            nn.Linear(128, 64), nn.LayerNorm(64), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(64, 16), nn.LeakyReLU(0.1),
            nn.Linear(16, 1)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(128, 32), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        self.apply(init_weights)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        return self.reg_head(last), self.cls_head(last)

class MLPLSTM(nn.Module):
    def __init__(self, input_dim: int):
        super(MLPLSTM, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.LayerNorm(128), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.LayerNorm(64),  nn.LeakyReLU(0.1),
        )
        self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.reg_head = nn.Sequential(nn.Linear(128, 32), nn.LeakyReLU(0.1), nn.Linear(32, 1))
        self.cls_head = nn.Sequential(nn.Linear(128, 32), nn.LeakyReLU(0.1), nn.Dropout(0.3), nn.Linear(32, 1))
        self.apply(init_weights)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        features = self.mlp(x.view(-1, input_dim)).view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        last = lstm_out[:, -1, :]
        return self.reg_head(last), self.cls_head(last)

class TransformerLSTM(nn.Module):
    def __init__(self, input_dim: int):
        super(TransformerLSTM, self).__init__()
        self.proj = nn.Linear(input_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, batch_first=True, dropout=0.3, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.reg_head = nn.Sequential(nn.Linear(128, 32), nn.LeakyReLU(0.1), nn.Linear(32, 1))
        self.cls_head = nn.Sequential(nn.Linear(128, 32), nn.LeakyReLU(0.1), nn.Dropout(0.3), nn.Linear(32, 1))
        self.apply(init_weights)

    def forward(self, x):
        x_proj = self.proj(x)
        attn_out = self.transformer(x_proj)
        lstm_out, _ = self.lstm(attn_out)
        last = lstm_out[:, -1, :]
        return self.reg_head(last), self.cls_head(last)

def train_centralized(model, dataloader, epochs, lr, device):
    if isinstance(model, SVMModel):
        model.train_model(dataloader)
        return model

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = DualHeadLoss(reg_weight=1.0, cls_weight=2.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(1, epochs // 3), T_mult=2)

    for ep in range(epochs):
        epoch_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            reg_pred, cls_logit = model(x)
            loss = criterion(reg_pred, cls_logit, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

    return model

def train_federated(global_model, client_dataloaders, rounds, local_epochs, lr, device, mu=0.01):
    global_model.to(device)
    criterion = DualHeadLoss(reg_weight=1.0, cls_weight=2.0)

    for r in range(rounds):
        local_weights_list = []
        for client_idx, loader in enumerate(client_dataloaders):
            local_model = copy.deepcopy(global_model)
            local_model.to(device)
            local_model.train()
            optimizer = torch.optim.AdamW(local_model.parameters(), lr=lr, weight_decay=1e-3)

            for ep in range(local_epochs):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    reg_pred, cls_logit = local_model(x)
                    base_loss = criterion(reg_pred, cls_logit, y)
                    proximal_term = 0.0
                    for local_p, global_p in zip(local_model.parameters(), global_model.parameters()):
                        proximal_term += torch.sum((local_p - global_p) ** 2)
                    loss = base_loss + (mu / 2.0) * proximal_term
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                    optimizer.step()

            local_weights_list.append(copy.deepcopy(local_model.state_dict()))

        avg_weights = {}
        for key in local_weights_list[0].keys():
            avg_weights[key] = sum(w[key] for w in local_weights_list) / len(local_weights_list)
        global_model.load_state_dict(avg_weights)

    return global_model
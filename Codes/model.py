import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def initialize_weights(m):
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

class DualHeadLoss(nn.Module):
    def __init__(self, reg_weight: float = 1.0, cls_weight: float = 2.0):
        super(DualHeadLoss, self).__init__()
        self.reg_weight = reg_weight
        self.cls_weight = cls_weight
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, reg_pred: torch.Tensor, cls_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_reshaped = target.view(reg_pred.size(0), -1)
        reg_loss = self.mse(reg_pred, target_reshaped)
        direction_label = (target_reshaped > 0).float()
        cls_loss = self.bce(cls_logit, direction_label)
        return self.reg_weight * reg_loss + self.cls_weight * cls_loss

class TimeStepMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.3):
        super(TimeStepMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_dim = x.shape
        x_flat = x.contiguous().view(-1, input_dim)
        features_flat = self.mlp(x_flat)
        return features_flat.view(batch_size, seq_len, -1)

class MLPLSTMTransformer(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.3):
        super(MLPLSTMTransformer, self).__init__()
        self.mlp_extractor = TimeStepMLP(input_dim, dropout=dropout)
        self.lstm = nn.LSTM(input_size=32, hidden_size=128, num_layers=2, batch_first=True, dropout=dropout)
        self.lstm_norm = nn.LayerNorm(128)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True, dropout=dropout * 0.5)
        self.attention_norm = nn.LayerNorm(128)
        self.reg_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor):
        mlp_features = self.mlp_extractor(x)
        lstm_out, _ = self.lstm(mlp_features)
        lstm_out = self.lstm_norm(lstm_out)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attention_norm(attn_out + lstm_out)
        pooled = attn_out.mean(dim=1)
        return self.reg_head(pooled), self.cls_head(pooled)

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 10):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.X) - self.seq_len

    def __getitem__(self, idx: int):
        seq_x = self.X[idx: idx + self.seq_len]
        target_y = self.y[idx + self.seq_len]
        return seq_x, target_y

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int, lr: float, device: torch.device) -> nn.Module:
    model.to(device)
    criterion = DualHeadLoss(reg_weight=1.0, cls_weight=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(1, epochs // 3), T_mult=2)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            reg_pred, cls_logit = model(batch_x)
            loss = criterion(reg_pred, cls_logit, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                reg_pred, cls_logit = model(batch_x)
                loss = criterion(reg_pred, cls_logit, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step()

    return model
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class PredConfig:
    hist_len: int = 20
    fut_len: int = 30
    feat_dim: int = 10 # [x,y,vx,vy,yaw,... actions,... veh embeds]
    hidden: int = 128
    layers: int = 2
    
class LSTMForecaster(nn.Module):
    def __init__(self, cfg: PredConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.LSTM(input_size=cfg.feat_dim, hidden_size=cfg.hidden, num_layers=cfg.layers, batch_first=True)
        self.dec = nn.LSTM(input_size=2, hidden_size=cfg.hidden, num_layers=cfg.layers, batch_first=True)
        self.head = nn.Linear(cfg.hidden, 2)
        
        def forward(self, hist: torch.Tensor, fut_seed: torch.Tensor | None = None):
            # hist: (B, T_h, F)
            B = hist.size(0)
            h, (hT, cT) = self.encoder(hist)
            # teacher-forced decoder (inference: auto-regressive)
            if self.training and fut_seed is not None:
                dec_h, _ = self.dec(fut_seed, (hT, cT))
                dxy = self.head(dec_h)
                return dxy
            
            else:
                T = self.cfg.fut_len
                y = []
                y_t = torch.zeros(B, 1, 2, device=hist.device)
                h_s, c_s = hT, cT

                for _ in range(T):
                    dec_h, (h_s, c_s) = self.dec(y_t, (h_s, c_s))
                    dxy = self.head(dec_h)
                    y_t = torch.cat([y_t, dxy], dim=1)
                    y.append(y_t)
                y = torch.cat(y, dim=1)
                return y
    
    
def loss_fn(pred: torch.Tensor, target: torch.Tensor):
    return ((pred - target) ** 2).mean()
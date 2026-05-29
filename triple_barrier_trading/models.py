import torch
import torch.nn as nn
import torch.nn.functional as F

class TradingPolicyNetwork(nn.Module):
    def __init__(self, feature_dim=4, hidden_dim=256, n_heads=4):
        super(TradingPolicyNetwork, self).__init__()
        
        # Initial LayerNorm to stabilize raw features
        self.input_norm = nn.LayerNorm(feature_dim)
        
        # 1. Local Feature Extraction (1D-CNN)
        self.cnn = nn.Sequential(
            nn.Conv1d(feature_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # 2. Long-range Dependency (Transformer Encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 4, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Final Norm before heads
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # 3. Output Heads
        # Action Head: [Long, Short, Hold]
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # Take-Profit Head: Output (0, 0.15)
        self.tp_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Stop-Loss Head: Output (0, 0.10)
        self.sl_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x shape: [Batch, Window, Feature_Dim]
        """
        # 0. Input Normalization
        x = self.input_norm(x)
        
        # 1. CNN Extraction
        # CNN expects [Batch, Channels, Length]
        x = x.transpose(1, 2) 
        feat = self.cnn(x) # [Batch, Hidden, Window]
        feat = feat.transpose(1, 2) # [Batch, Window, Hidden]
        
        # 2. Transformer encoding
        encoding = self.transformer(feat)
        
        # 3. Use the last time step for decision making
        last_state = self.final_norm(encoding[:, -1, :]) # [Batch, Hidden]
        
        # 4. Heads
        logits = self.action_head(last_state)
        # Numerical stability: use Softmax but ensure no NaN inputs
        probs = F.softmax(logits, dim=-1)
        
        tp_ratio = self.tp_head(last_state) * 0.15
        sl_ratio = self.sl_head(last_state) * 0.10
        
        return probs, tp_ratio, sl_ratio

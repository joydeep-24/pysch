# fusion_model.py
import torch
import torch.nn as nn

class MultimodalFusionModel(nn.Module):
    """
    Fusion Model v3 (Emotion-Aligned)
    Input: [text_VA (2), vision_VA (2), context (4)] = 8 dims
    Output: 5 psychological states
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        return self.net(x)

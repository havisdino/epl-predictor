from torch import nn

from .base import ModelBase


class MLPWithAttention(ModelBase):
    def __init__(self, hidden_size, num_teams, num_venues=3, num_results=3):
        super().__init__(hidden_size, num_teams, num_venues, num_results)
        
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size * 7, hidden_size * 7 * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 7 * 2, num_results)
        )
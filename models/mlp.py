from torch import nn

from .base import ModelBase


class MLPWithAttention(ModelBase):
    def __init__(self, hidden_size, num_teams, num_venues=3, num_results=3):
        super().__init__(hidden_size, num_teams, num_venues, num_results)
        
        input_size = hidden_size * 4 + 3
        
        self.out_proj = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.GELU(),
            nn.Linear(input_size * 2, num_results)
        )
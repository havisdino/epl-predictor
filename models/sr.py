from torch import nn

from .base import ModelBase


class SoftmaxRegressorWithAttention(ModelBase):
    def __init__(self, hidden_size, num_teams, num_venues=3, num_results=3):
        super().__init__(hidden_size, num_teams, num_venues, num_results)
        
        self.out_proj = nn.Linear(7 * hidden_size, num_results)
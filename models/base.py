import torch
from torch import nn
import torch.nn.functional as F
from abc import ABC


class ModelBase(ABC, nn.Module):
    def __init__(self, hidden_size, num_teams, num_venues=3, num_results=3):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.team_embed = nn.Embedding(num_teams, hidden_size)
        self.venue_embed = nn.Embedding(num_venues, hidden_size)
        self.result_embed = nn.Embedding(num_results, hidden_size)
        
        self.k_proj = nn.Linear(4 * hidden_size, 4 * hidden_size)
        self.q_proj = nn.Linear(3 * hidden_size, 4 * hidden_size)
        
        self.out_proj = None
    
    def forward(self, inputs):
        if self.out_proj is None:
            raise NotImplementedError()
        
        past_matches = inputs["past_matches"]
        
        past_venues = self.venue_embed(past_matches["venue"])
        past_teams = self.team_embed(past_matches["team"])
        past_opponents = self.team_embed(past_matches["opponent"])
        past_results = self.result_embed(past_matches["result"])
        
        past_seq = torch.cat([past_venues, past_teams, past_opponents, past_results], dim=-1)
        
        next_match_conditions = inputs["next_match_conditions"]
        
        next_venue = self.venue_embed(next_match_conditions["venue"])
        next_team = self.team_embed(next_match_conditions["team"])
        next_opponent = self.team_embed(next_match_conditions["opponent"])
        
        next_cond = torch.cat([next_venue, next_team, next_opponent], dim=-1)
        
        attn = torch.matmul(self.k_proj(past_seq), self.q_proj(next_cond).transpose(1, 2))
        attn = F.softmax(attn / (self.hidden_size * 3) ** 0.5, dim=1) # BxLx1
        
        past_repr = (attn * past_seq).sum(1) # summate along the length dimension
        
        input = torch.cat([past_repr, next_cond.squeeze(1)], dim=-1) # concatenate with the next match's condition tensor
        
        logits = self.out_proj(input)
        return logits
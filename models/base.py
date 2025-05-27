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
        
        self.out_proj = None    # a mapping to the output logits
    
    def forward(self, inputs):
        r"""Inputs should be a dictionary with the following keys:
        - "past_matches": a dictionary containing:
            - "venue": tensor of shape (B, L) with past match venues
            - "team": tensor of shape (B, L) with past match teams
            - "opponent": tensor of shape (B, L) with past match opponents
            - "result": tensor of shape (B, L) with past match results
            - "goals_for": tensor of shape (B, L) with goals scored by the team
            - "goals_against": tensor of shape (B, L) with goals scored by the opponent
        - "next_match_conditions": a dictionary containing:
            - "venue": tensor of shape (B, 1) with next match venue
            - "team": tensor of shape (B, 1) with next match team
            - "opponent": tensor of shape (B, 1) with next match opponent
        - "stats": tensor of shape (B, 3) where 3 values are win rate, draw rate and loss rate of the team
        """
        if self.out_proj is None:
            raise NotImplementedError()
        
        past_matches = inputs["past_matches"]
        
        past_venues = self.venue_embed(past_matches["venue"])
        past_teams = self.team_embed(past_matches["team"])
        past_opponents = self.team_embed(past_matches["opponent"])
        past_results = self.result_embed(past_matches["result"])
        past_goals_for = past_matches["goals_for"].unsqueeze(-1)  # BxLx1
        past_goals_against = past_matches["goals_against"].unsqueeze(-1)  # BxLx1
        
        past_seq = torch.cat([
            past_venues, past_teams,
            past_opponents, past_results,
            past_goals_for, past_goals_against
        ], dim=-1)
        past_seq_for_attn = past_seq[..., :self.hidden_size * 3]
        
        next_match_conditions = inputs["next_match_conditions"]
        
        next_venue = self.venue_embed(next_match_conditions["venue"])
        next_team = self.team_embed(next_match_conditions["team"])
        next_opponent = self.team_embed(next_match_conditions["opponent"])
        
        next_cond = torch.cat([next_venue, next_team, next_opponent], dim=-1)
        
        attn = torch.matmul(past_seq_for_attn, next_cond).transpose(1, 2)
        attn = F.softmax(attn / (self.hidden_size * 3) ** 0.5, dim=1) # BxLx1
        
        past_repr = (attn * past_seq).sum(1) # summate along the length dimension
        
        stats = inputs["stats"]
        input = torch.cat([past_repr, stats], dim=-1) # concatenate with the next match's condition tensor
        
        logits = self.out_proj(input)
        return logits
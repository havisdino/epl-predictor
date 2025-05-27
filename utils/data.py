import torch
import random
from datasets import load_dataset

from utils.sqlite_tools import get_n_previous_matches_sqlite


TEAMS = [
    "Arsenal", "Aston Villa", "Chelsea",
    "Everton", "Leicester City", "Liverpool",
    "Manchester City", "Manchester Utd", "Newcastle Utd",
    "Southampton", "Tottenham", "West Ham"
]
INDICES = [i for i in range(len(TEAMS))]
TEAM_TO_INDEX = dict(zip(TEAMS, INDICES))

RESULTS = ["W", "L", "D"]
RESULT_TO_INDEX = dict(zip(RESULTS, INDICES))

VENUES = ["Home", "Away", "Neutral"]
VENUE_TO_INDEX = dict(zip(VENUES, INDICES))


def load_and_encode(data_files="data/db2/train_matches.jsonl"):
    return load_dataset("json", data_files=data_files, split="train").map(
        lambda s: dict(
            venue=VENUE_TO_INDEX[s["venue"]],
            result=RESULT_TO_INDEX[s["result"]],
            team=TEAM_TO_INDEX[s["team"]],
            opponent=TEAM_TO_INDEX[s["opponent"]]
        )
    )


def generator(data_files="data/db2/train_matches.jsonl"):
    dataset = load_and_encode(data_files)
    dataset = dataset.sort("date")
    
    while True:
        next_match = random.choice(dataset)
        past_matches = get_n_previous_matches_sqlite(next_match["date"].strftime("%Y-%m-%d"), N=10)
        
        next_match.pop("date")
        
        past_matches = {
            "venue": [VENUE_TO_INDEX[match["venue"]] for match in past_matches],
            "result": [RESULT_TO_INDEX[match["result"]] for match in past_matches],
            "team": [TEAM_TO_INDEX[match["team"]] for match in past_matches],
            "opponent": [TEAM_TO_INDEX[match["opponent"]] for match in past_matches],
            "goals_for": [match["goals_for"] for match in past_matches],
            "goals_against": [match["goals_against"] for match in past_matches]
        }
        
        yield {"past_matches": past_matches, "next_match": next_match, "stats": ...}
        

def process(inputs):
    assert isinstance(inputs, dict)
    
    past_matches = inputs["past_matches"]
    next_match = inputs["next_match"]
    
    past_matches = {k: torch.tensor([v]) for k, v in past_matches.items()}
    
    next_match_result = next_match["result"]
    next_match_result = torch.tensor(next_match_result)
    
    next_match.pop("result")
    next_match_conditions = next_match
    next_match_conditions = {k: torch.tensor([v]) for k, v in next_match_conditions.items()}
    
    return {
        "past_matches": past_matches,
        "next_match_conditions": next_match_conditions,
        "next_match_result": next_match_result,
        "stats": ...
    }


def batch_generator(data_files):
    data_iter = generator(data_files)
    while True:
        yield process(next(data_iter))
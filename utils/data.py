import torch
from random import randint
from datasets import load_dataset


TEAMS = [
    "Arsenal", "Aston Villa", "Chelsea",
    "Everton", "Leicester City", "Liverpool",
    "Manchester City", "Manchester Utd", "Newcastle Utd",
    "Southampton", "Tottenham", "West Ham"
]
INDICES = [i for i in range(len(TEAMS))]
TEAM_TO_INDEX = dict(zip(TEAMS, INDICES))
RESULT_TO_INDEX = dict(zip(["W", "L", "D"], INDICES))
VENUE_TO_INDEX = dict(zip(["Home", "Away", "Neutral"], INDICES))
            

def load_and_encode(data_files="jsonl/*.jsonl"):
    return load_dataset("json", data_files=data_files, split="train").map(
        lambda s: dict(
            venue=VENUE_TO_INDEX[s["venue"]],
            result=RESULT_TO_INDEX[s["result"]],
            team=TEAM_TO_INDEX[s["team"]],
            opponent=TEAM_TO_INDEX[s["opponent"]]
        )
    )


def generator(data_files="jsonl/*.jsonl"):
    dataset = load_and_encode(data_files)
    dataset = dataset.sort("date").remove_columns("date")    
    
    num_samples = len(dataset)

    while True:
        index = randint(10, num_samples - 1)
        past_matches = dataset[index - 10:index]
        next_match = dataset[index:index + 1]
        
        yield {"past_matches": past_matches, "next_match": next_match}
        

def proccess(inputs):
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
        "next_match_result": next_match_result
    }


def batch_generator(data_files):
    data_iter = generator(data_files)
    while True:
        yield proccess(next(data_iter))
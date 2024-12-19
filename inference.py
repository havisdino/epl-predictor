import torch

from utils.data import RESULTS, TEAMS, VENUE_TO_INDEX, VENUES, batch_generator
from models.lightning_wrapper import LightningWrapper
from models.mlp import MLPWithAttention
from models.sr import SoftmaxRegressorWithAttention
from utils.config import Config
from utils.metrics import entropy


team_a = None
team_b = None
result = None


def decode_info(inputs):
    global team_a, team_b, result
    
    print(">>> Past matches")
    
    venue_ids = inputs["past_matches"]["venue"].squeeze().tolist()
    team_ids = inputs["past_matches"]["team"].squeeze().tolist()
    opponent_ids = inputs["past_matches"]["opponent"].squeeze().tolist()
    result_ids = inputs["past_matches"]["result"].squeeze().tolist()
    
    for venue_id, team_id, opponent_id, result_id in zip(venue_ids, team_ids, opponent_ids, result_ids):
        venue = VENUES[venue_id]
        team = TEAMS[team_id]
        opponent = TEAMS[opponent_id]
        result = RESULTS[result_id]
        
        print(f"\tVenue: {venue};\tTeam A: {team};\tTeam B: {opponent};\t\tResult for Team A: {result}")
        
    print("\n>>> Next match")
    
    venue_id = inputs["next_match_conditions"]["venue"].item()
    team_id = inputs["next_match_conditions"]["team"].item()
    opponent_id = inputs["next_match_conditions"]["opponent"].item()
    
    team_a = TEAMS[team_id]
    team_b = TEAMS[opponent_id]
    result = inputs["next_match_result"].item()
    
    print(f"\tVenue: {VENUES[venue_id]};\tTeam A: {team_a};\tTeam B: {team_b};\n")
    

def decode_pred(probs):
    print(">>> Prediction")
    print(f"\tA wins: {probs[0, 0]:.2f}\n\tB wins: {probs[0, 1]:.2f}\n\tDraws: {probs[0, 2]:.2f}")
    
    pred_id = probs.argmax(-1).item()
    verbs = ["wins", "loses", "draws"]
    
    print(f"\t* Entropy: {entropy(probs).item():.2f}")
    
    print(f"\n>>> The most probable:\t{team_a} {verbs[pred_id]} {team_b}")
    print(f">>> Ground truth:\t{team_a} {verbs[result]} {team_b}")


def main(config):
    if config.model_type == "mlp":
        model = MLPWithAttention(**vars(config.model_args))
    elif config.model_type == "sr":
        model = SoftmaxRegressorWithAttention(**vars(config.model_args))
    else:
        raise NotImplementedError()

    wrapper = LightningWrapper.load_from_checkpoint(config.checkpoint, map_location="cpu", model=model)

    data_iter = batch_generator(config.data_files)
    
    inputs = next(data_iter)
    with torch.no_grad():
        wrapper.eval()
        probs = wrapper(inputs).softmax(-1)
        
    decode_info(inputs)
    decode_pred(probs)


if __name__ == "__main__":
    config = Config.from_yaml("config.yml")
    
    main(config)
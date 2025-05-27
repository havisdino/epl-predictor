import torch
from datetime import datetime
from models.lightning_wrapper import LightningWrapper
from flask import Flask, jsonify, request
from models.mlp import MLPWithAttention
from utils.config import Config
from utils.data import TEAM_TO_INDEX, VENUE_TO_INDEX, load_and_encode


def load_model():
    config = Config.from_yaml("config.yml")
    model = MLPWithAttention(**vars(config.model_args))
    wrapper = LightningWrapper.load_from_checkpoint(config.checkpoint, map_location="cpu", model=model)
    return wrapper


app = Flask(__name__)
encoded_data = load_and_encode("data/raw.old/2014-2024/*.jsonl")
encoded_data = load_and_encode("data/db/*.jsonl")
model = load_model()


def search_matches_before(match):
    res = encoded_data[-10:]
    
    for i, past_match in enumerate(encoded_data):
        
        if (
            past_match["team"] == match["team"]
            and past_match["opponent"] == match["opponent"]
            and past_match["date"].strftime("%Y-%m-%d") == match["date"]
        ):
            res = encoded_data[i-10:i]
            break
    
    # debug
    print(past_match["date"].strftime("%Y-%m-%d"), flush=True)
    app.logger.error(f">>> received: {match["date"]}")
    for date in res["date"]:
        app.logger.error(date.strftime("%Y-%m-%d"))
    # end debug
    res.pop("date")
    return res


@app.route("/predict", methods=["POST"])
def predict_match():
    data = request.json
    date = data.get("date")
    team = data.get("team")
    opponent = data.get("opponent")
    venue = data.get("venue")
    
    past_matches = search_matches_before(match={
        "date": date,
        "team": TEAM_TO_INDEX[team],
        "opponent": TEAM_TO_INDEX[opponent]
    })
    
    past_matches = {k: torch.tensor([v]) for k, v in past_matches.items()}
    
    next_match_conditions = {
        "venue": torch.tensor([[VENUE_TO_INDEX[venue]]]),
        "team": torch.tensor([[TEAM_TO_INDEX[team]]]),
        "opponent": torch.tensor([[TEAM_TO_INDEX[opponent]]])
    }
    
    inputs = {
        "past_matches": past_matches,
        "next_match_conditions": next_match_conditions
    }
    
    model.eval()
    with torch.no_grad():
        output = model(inputs)
        probs = output.softmax(-1).cpu().numpy()
    result = {
        "team": team,
        "opponent": opponent,
        "venue": venue,
        "probabilities": {
            "win": float(probs[0, 0]),
            "lose": float(probs[0, 1]),
            "draw": float(probs[0, 2])
        }
    }
    
    return jsonify(result)
    

if __name__ == "__main__":
    app.run(host="localhost", port=8081, debug=True)
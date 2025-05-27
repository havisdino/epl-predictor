import torch
from datetime import datetime
from models.lightning_wrapper import LightningWrapper
from flask import Flask, jsonify, request
from models.mlp import MLPWithAttention
from utils.config import Config
from utils.data import TEAM_TO_INDEX, VENUE_TO_INDEX, load_and_encode


def load_data():
    global raw_data

    from datasets import load_dataset

    raw_data = load_dataset("json", data_files="data/db/*.jsonl", split="train").to_list()
    raw_data = list(sorted(raw_data, key=lambda x: x["date"]))

    return raw_data


def load_model():
    config = Config.from_yaml("config.yml")
    model = MLPWithAttention(**vars(config.model_args))
    wrapper = LightningWrapper.load_from_checkpoint(config.checkpoint, map_location="cpu", model=model)
    return wrapper


app = Flask(__name__)
encoded_data = load_and_encode("data/raw.old/2014-2024/*.jsonl")
encoded_data = load_and_encode("data/db/*.jsonl")
raw_data = load_data()
model = load_model()


@app.route("/matches", methods=["GET"])
def get_matches():
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    team = request.args.get("team")

    filtered_data = raw_data

    if start_date:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        filtered_data = [m for m in filtered_data if datetime.strptime(m["date"], "%Y-%m-%d") >= start_date]

    if end_date:
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        filtered_data = [m for m in filtered_data if datetime.strptime(m["date"], "%Y-%m-%d") <= end_date]

    if team:
        filtered_data = [m for m in filtered_data if team in (m["team_a"], m["team_b"])]

    return jsonify(filtered_data)


@app.route("/teams", methods=["GET"])
def get_teams():
    teams = set()
    for match in raw_data:
        teams.add(match["team"])
        teams.add(match["opponent"])
    return jsonify(sorted(list(teams)))


@app.route("/match/<date>/<team_a>/<team_b>", methods=["GET"])
def get_match(date, team_a, team_b):
    for match in encoded_data[::-1]:
        if match["date"] == date and match["team_a"] == team_a and match["team_b"] == team_b:
            return jsonify(match)
    return jsonify({"error": "Match not found"}), 404


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
    app.run(host="localhost", port=8888, debug=True)

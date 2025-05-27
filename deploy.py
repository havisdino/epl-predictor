from flask import Flask, jsonify, request
from datetime import datetime
from models.lightning_wrapper import LightningWrapper
from models.mlp import MLPWithAttention
from utils.config import Config


def load_data():
    global data

    from datasets import load_dataset

    data = load_dataset("json", data_files="data/db/*.jsonl", split="train").to_list()
    data = list(sorted(data, key=lambda x: x["date"]))

    return data


def load_model():
    config = Config.from_yaml("config.yml")
    model = MLPWithAttention(**vars(config.model_args))
    wrapper = LightningWrapper.load_from_checkpoint(config.checkpoint, map_location="cpu", model=model)
    return wrapper


app = Flask(__name__)
data = load_data()
model = load_model()


@app.route("/matches", methods=["GET"])
def get_matches():
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    team = request.args.get("team")

    filtered_data = data

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
    for match in data:
        teams.add(match["team_a"])
        teams.add(match["team_b"])
    return jsonify(sorted(list(teams)))


@app.route("/match/<date>/<team_a>/<team_b>", methods=["GET"])
def get_match(date, team_a, team_b):
    for match in data[::-1]:
        if match["date"] == date and match["team_a"] == team_a and match["team_b"] == team_b:
            return jsonify(match)
    return jsonify({"error": "Match not found"}), 404


def search_matches_before(date):
    rdata = data[::-1]
    for i, match in enumerate(rdata):
        if match["date"] < date:
            return rdata[1:i+10]
    return None



@app.route("/predict", methods=["POST"])
def predict_match():
    


if __name__ == "__main__":
    data = load_data()
    app.run(host="0.0.0.0", port=8888, debug=True)

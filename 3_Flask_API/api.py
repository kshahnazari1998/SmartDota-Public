import flask
from flask import request, jsonify
from flask_cors import CORS

from flask_talisman import Talisman
import torch
import torch.nn.functional as F
from torch import nn
import operator
import numpy as np

import pickle

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import PolynomialFeatures

app = flask.Flask(__name__)

CORS(app)
# only trigger SSLify if the app is running on Heroku
Talisman(app)


class predict_game:
    def __init__(self, radiant_picks, dire_picks, side, role):
        self.radiant_picks = radiant_picks
        self.dire_picks = dire_picks
        self.side = side
        self.role = role

    def predict(self):
        # return self.side
        radiant, dire = self.trim_data(self.radiant_picks, self.dire_picks)
        picks = self.transform_data(radiant, dire)
        picks_linear = self.transform_data_linear(radiant, dire)
        # prob = self.model_predict(picks)
        pickfile = open("picklist/" + self.role + ".txt", "r")
        ids_str = pickfile.readlines()
        ids = []
        for id in ids_str:
            ids.append(int(id))
        win_prob = []
        for hero in ids:
            if picks[hero] == 0:
                if self.side == "radiant":
                    picks[hero] = -1
                    picks_linear[hero] = 1
                    prob1 = self.model_predict(picks)
                    prob2 = self.model_predict_linear(picks_linear)
                    prob = (prob1 + prob2) / 2
                    picks_linear[hero] = 0
                else:
                    picks[hero] = 1
                    picks_linear[hero + 150] = 1
                    prob1 = 1 - self.model_predict(picks)
                    prob2 = 1 - self.model_predict_linear(picks_linear)
                    prob = (prob1 + prob2) / 2
                    picks_linear[hero + 150] = 0
                picks[hero] = 0
                win_prob.append((hero, prob))

        return sorted(win_prob, key=operator.itemgetter(1), reverse=True)

    def trim_data(self, radiant_picks_data, dire_picks_data):
        try:
            radiant_picks = []
            dire_picks = []

            for pick in radiant_picks_data:
                if pick == 0:
                    break
                else:
                    radiant_picks.append(pick)

            for pick in dire_picks_data:
                if pick == 0:
                    break
                else:
                    dire_picks.append(pick)

            return radiant_picks, dire_picks

        except:
            return "Error"

    def transform_data(self, radiant_picks, dire_picks):
        picks = torch.zeros(150)
        for rad in radiant_picks:
            picks[rad] = -1
        for dir in dire_picks:
            picks[dir] = 1
        return picks

    def transform_data_linear(self, radiant_picks, dire_picks):
        picks = np.zeros(300)
        for rad in radiant_picks:
            picks[rad] = 1
        for dir in dire_picks:
            picks[dir + 150] = 1
        return picks

    def model_predict(self, picks):
        predictor = GamePredictor_final()
        predictor.load_state_dict(torch.load("model.pth"))
        predictor.eval()
        pick = picks.view(-1, 150).float()
        output = predictor(pick).item()
        return output

    def model_predict_linear(self, picks):
        predictor = pickle.load(open("linear_model", "rb"))
        poly = PolynomialFeatures(degree=2)
        x = poly.fit_transform(np.array([picks]))
        output = predictor.predict_proba(x)
        return output[0][0]


class GamePredictor_final(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(150, 50)
        self.l2 = nn.Linear(50, 50)
        self.l3 = nn.Linear(50, 1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return torch.sigmoid(x)


@app.route("/", methods=["GET"])
def home():
    return "Just a simple API"


@app.route("/api/v1/predictor", methods=["GET"])
def api():
    try:
        side = str(request.args["side"])
        role = str(request.args["role"])

        r1 = int(request.args["r1"])
        r2 = int(request.args["r2"])
        r3 = int(request.args["r3"])
        r4 = int(request.args["r4"])
        r5 = int(request.args["r5"])

        d1 = int(request.args["r1"])
        d2 = int(request.args["r2"])
        d3 = int(request.args["r3"])
        d4 = int(request.args["r4"])
        d5 = int(request.args["r5"])

        radiant_picks = [r1, r2, r3, r4, r5]
        dire_picks = [d1, d2, d3, d4, d5]

        res = predict_game(radiant_picks, dire_picks, side, role).predict()
    except Exception as e:
        print(e)
        res = "There was an error in Api Call"

    return jsonify(res)


app.config["UPLOAD_FOLDER"] = "."


@app.route("/file", methods=["POST", "PUT"])
def upload():
    if request.method == "POST":
        password = request.form["pass"]
        if str(password) == "xxx":
            f = request.files["app"]
            f.save((f.filename))
            return "Success"
        else:
            return "Wrong Pass"


if __name__ == "__main__":
    app.run()

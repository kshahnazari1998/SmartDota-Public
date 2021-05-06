import json
import pandas as pd
import numpy as np
import random
from Sqldatabasehandler import sqlhandler
from datetime import datetime

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import PolynomialFeatures
import pickle

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


class ModelUpdater:
    def __init__(self, host, user, passwd, database):
        """
        The constructor for the Class
        """
        self.sqlhand = sqlhandler(host, user, passwd, database)

    def update_model(self, batchsize=100000):
        last_seq = self.get_last_game()
        res = self.sqlhand.SqlQueryExec(
            "SELECT count(*) FROM DotaMatches WHERE GameSEQ> %s",
            True,
            [
                last_seq,
            ],
        )
        if res == -1:
            return -1
        new_games_count = self.sqlhand.get_row_result()
        if new_games_count >= batchsize:
            games = self.sqlhand.get_all_select_rows(
                "SELECT * FROM DotaMatches WHERE GameSEQ>%s order by GameSEQ Limit %s",
                [
                    last_seq,
                    batchsize,
                ],
            )
            cols = self.sqlhand.get_all_select_rows(
                "SHOW columns FROM DotaMatches",
            )
            cols = [x[0] for x in cols]
            games = pd.DataFrame(games)
            games.columns = cols
            games = games.dropna(subset=["Pick1Rad"])
            model, linear = self.train_new_model(games)

            now = datetime.now()
            date_time = now.strftime("%m_%d_%Y_%H")
            max_game_seq = games["GameSEQ"].max()
            self.update_last_game(max_game_seq)

            torch.save(model.state_dict(), "./model.pth")
            torch.save(
                model.state_dict(),
                "./old_models/model_" + date_time + "_" + str(max_game_seq) + ".pth",
            )
            pickle.dump(linear, open(f"linear_model", "wb"))
            pickle.dump(
                linear,
                open(
                    "./old_models/linear_model_"
                    + date_time
                    + "_"
                    + str(max_game_seq)
                    + ".pth",
                    "wb",
                ),
            )
            del games
            self.update_model()
        else:
            return 0

    def get_last_game(self):
        try:
            filepath = "last_game_seq.txt"
            fp = open(filepath)
            last_seq = int(fp.read())
            fp.close()
            return last_seq
        except:
            return -1

    def update_last_game(self, lastseq):
        try:
            filepath = "last_game_seq.txt"
            fp = open(filepath, "w")
            fp.write(str(lastseq))
            fp.close()
            return 0
        except:
            return -1

    def train_new_model(self, df):
        df_no_leavers = df.query("Leavers==0")

        class game_datasets(Dataset):
            def __init__(self, rawdata):
                X = rawdata.loc[:, "Pick1Rad":"Pick5Dir"]
                y = rawdata["RadiantWin"]
                self.x = torch.tensor(X.values)
                self.y = torch.tensor(y.values)

            def __getitem__(self, index):
                return self.x[index], self.y[index]

            def __len__(self):
                return len(self.y)

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

        net = GamePredictor_final()
        net.load_state_dict(torch.load("model.pth"))
        net.train()

        optimizer = optim.Adam(net.parameters(), lr=0.001)

        Epochs = 1

        for epoch in range(0, Epochs):
            train_data_set = game_datasets(df_no_leavers)
            train_data_loader = DataLoader(train_data_set, batch_size=10000)
            train_data_iter = iter(train_data_loader)
            for data in train_data_iter:

                x, y = data
                net.zero_grad()
                x = self.game_datasets_transform_X(x, 10)
                # print(x[100])
                y = self.game_datasets_transform_Y(y, 10)
                x = x.view(-1, 150).float()
                y = y.view(-1, 1).float()
                output = net(x)
                loss_func = nn.MSELoss()
                loss = loss_func(output, y)
                loss.backward()
                optimizer.step()
            print("Done Training")

        # Training SGD
        train_data_set = game_datasets(df_no_leavers)
        train_data_loader = DataLoader(train_data_set, batch_size=2500)
        train_data_iter = iter(train_data_loader)
        poly = PolynomialFeatures(degree=2)
        loaded_model = pickle.load(open(f"linear_model", "rb"))
        del train_data_set
        for data in train_data_iter:
            x, y = data
            x = self.game_datasets_transform_X_SGD(x, 5)
            y = self.game_datasets_transform_Y(y, 5)
            x = x.view(-1, 300).float()
            y = y.view(-1, 1).float()
            x = x.numpy()
            x = poly.fit_transform(x)
            y = y.numpy().ravel()
            loaded_model.partial_fit(x, y, [0, 1])
        print("Done Training")

        return net, loaded_model

    def game_datasets_transform_X(self, data_X, mode=None, device="cpu"):
        # If mode is none only the 10 picks are added.
        # If mode is equal to 10 all possible combinations are added aswell.
        # If mode is either 1,2,3,4,5 the picks with those scenarios are only added.

        if mode is not None:
            picks = data_X.t()
            picks = picks.to(device)
            # 1st picks
            picks_rad = torch.zeros(data_X.shape[0], 150, device=device)
            picks_rad[range(picks_rad.shape[0]), torch.LongTensor(picks[0].long())] = -1
            picks_dire = torch.zeros(data_X.shape[0], 150, device=device)
            picks_dire[
                range(picks_dire.shape[0]), torch.LongTensor(picks[5].long())
            ] = 1
            if mode == 10:
                res = torch.cat([picks_rad, picks_dire], dim=0)
            if mode == 1:
                return torch.cat([picks_rad, picks_dire], dim=0)

            # 2nd picks
            picks_rad[range(picks_rad.shape[0]), torch.LongTensor(picks[1].long())] = -1
            picks_dire[
                range(picks_dire.shape[0]), torch.LongTensor(picks[6].long())
            ] = 1
            if mode == 10:
                res = torch.cat([res, picks_rad, picks_dire], dim=0)
            if mode == 2:
                return torch.cat([picks_rad, picks_dire], dim=0)

            # 3rd picks
            picks_rad[
                range(picks_rad.shape[0]), torch.LongTensor(picks[5:7].long())
            ] = 1
            picks_dire[
                range(picks_dire.shape[0]), torch.LongTensor(picks[0:2].long())
            ] = -1

            picks_rad[range(picks_rad.shape[0]), torch.LongTensor(picks[2].long())] = -1
            picks_dire[
                range(picks_dire.shape[0]), torch.LongTensor(picks[7].long())
            ] = 1
            if mode == 10:
                res = torch.cat([res, picks_rad, picks_dire], dim=0)
            if mode == 3:
                return torch.cat([picks_rad, picks_dire], dim=0)

            # 4th picks
            picks_rad[range(picks_rad.shape[0]), torch.LongTensor(picks[3].long())] = -1
            picks_dire[
                range(picks_dire.shape[0]), torch.LongTensor(picks[8].long())
            ] = 1
            if mode == 10:
                res = torch.cat([res, picks_rad, picks_dire], dim=0)
            if mode == 4:
                return torch.cat([picks_rad, picks_dire], dim=0)

            # 5th picks
            picks_rad[
                range(picks_rad.shape[0]), torch.LongTensor(picks[7:9].long())
            ] = 1
            picks_dire[
                range(picks_dire.shape[0]), torch.LongTensor(picks[2:4].long())
            ] = -1

            picks_rad[range(picks_rad.shape[0]), torch.LongTensor(picks[4].long())] = -1
            picks_dire[
                range(picks_dire.shape[0]), torch.LongTensor(picks[9].long())
            ] = 1
            if mode == 10:
                res = torch.cat([res, picks_rad, picks_dire], dim=0)
            if mode == 5:
                return torch.cat([picks_rad, picks_dire], dim=0)

            # All picks (Only for mode 10)
            picks_rad[range(picks_rad.shape[0]), torch.LongTensor(picks[9].long())] = 1
            res = torch.cat([res, picks_rad], dim=0)
            return res

        else:
            picks = data_X.t()
            picks = picks.to(device)
            picks_all = torch.zeros(data_X.shape[0], 150, device=device)
            picks_all[range(picks_all.shape[0]), picks[0:5]] = -1
            picks_all[range(picks_all.shape[0]), picks[5:10]] = 1
            return picks_all

    def game_datasets_transform_X_SGD(self, data_X, mode=None, device="cpu"):
        # If mode is none only the 10 picks are added.
        # If mode is equal to 10 all possible combinations are added aswell.
        # If mode is either 1,2,3,4,5 the picks with those scenarios are only added.

        if mode is not None:
            picks = data_X.t()
            picks = picks.to(device)
            # picks = data_X
            # 1st picks
            picks_rad = torch.zeros(data_X.shape[0], 300, device=device)
            picks_rad[range(picks_rad.shape[0]), torch.LongTensor(picks[0].long())] = 1
            picks_dire = torch.zeros(data_X.shape[0], 300, device=device)
            picks_dire[
                range(picks_dire.shape[0]), torch.LongTensor((picks[5] + 150).long())
            ] = 1
            if mode == 10:
                res = torch.cat([picks_rad, picks_dire], dim=0)
            if mode == 1:
                return torch.cat([picks_rad, picks_dire], dim=0)

            # 2nd picks
            picks_rad[range(picks_rad.shape[0]), torch.LongTensor(picks[1].long())] = 1
            picks_dire[
                range(picks_dire.shape[0]), torch.LongTensor((picks[6] + 150).long())
            ] = 1
            if mode == 10:
                res = torch.cat([res, picks_rad, picks_dire], dim=0)
            if mode == 2:
                return torch.cat([picks_rad, picks_dire], dim=0)

            # 3rd picks
            picks_rad[
                range(picks_rad.shape[0]), torch.LongTensor((picks[5:7] + 150).long())
            ] = 1
            picks_dire[
                range(picks_dire.shape[0]), torch.LongTensor(picks[0:2].long())
            ] = 1

            picks_rad[range(picks_rad.shape[0]), torch.LongTensor(picks[2].long())] = 1
            picks_dire[
                range(picks_dire.shape[0]), torch.LongTensor((picks[7] + 150).long())
            ] = 1
            if mode == 10:
                res = torch.cat([res, picks_rad, picks_dire], dim=0)
            if mode == 3:
                return torch.cat([picks_rad, picks_dire], dim=0)

            # 4th picks
            picks_rad[range(picks_rad.shape[0]), torch.LongTensor(picks[3].long())] = 1
            picks_dire[
                range(picks_dire.shape[0]), torch.LongTensor((picks[8] + 150).long())
            ] = 1
            if mode == 10:
                res = torch.cat([res, picks_rad, picks_dire], dim=0)
            if mode == 4:
                return torch.cat([picks_rad, picks_dire], dim=0)

            # 5th picks
            picks_rad[
                range(picks_rad.shape[0]), torch.LongTensor((picks[7:9] + 150).long())
            ] = 1
            picks_dire[
                range(picks_dire.shape[0]), torch.LongTensor(picks[2:4].long())
            ] = 1

            picks_rad[range(picks_rad.shape[0]), torch.LongTensor(picks[4].long())] = 1
            picks_dire[
                range(picks_dire.shape[0]), torch.LongTensor((picks[9] + 150).long())
            ] = 1
            if mode == 10:
                res = torch.cat([res, picks_rad, picks_dire], dim=0)
            if mode == 5:
                return torch.cat([picks_rad, picks_dire], dim=0)

            # All picks (Only for mode 10)
            picks_rad[range(picks_rad.shape[0]), torch.LongTensor(picks[9].long())] = 1
            res = torch.cat([res, picks_rad], dim=0)
            return res

        else:
            picks = data_X.t()
            picks = picks.to(device)
            picks_all = torch.zeros(data_X.shape[0], 150, device=device)
            picks_all[range(picks_all.shape[0]), picks[0:5]] = -1
            picks_all[range(picks_all.shape[0]), picks[5:10]] = 1
            return picks_all

    def game_datasets_transform_Y(self, data_Y, mode=None):
        # y_trans = []
        if mode == None:
            return data_Y

        y = data_Y.numpy()
        # for i, y in enumerate(data_Y.numpy()):
        if mode < 10:
            # y_trans.append(y)
            # y_trans.append(y)
            res = np.tile(y, 2)
        else:
            res = np.tile(y, 11)
            # res = np.concatenate([y,y])
            # for _ in range(10):
            # #  y_trans.append(y)
            #   res = np.concatenate([res,y])

        return torch.tensor(res)


if __name__ == "__main__":

    # Define Dota game scraper and create database connection
    try:
        # Define Dota game scraper and create database connection
        with open("keys.json") as f:
            keys = json.load(f)
        host = keys["database"]["host"]
        print(host)
        something = ModelUpdater(
            host=keys["database"]["host"],
            user=keys["database"]["user"],
            passwd=keys["database"]["passwd"],
            database=keys["database"]["database"],
        )

        something.update_model()
    except Exception as e:
        print(f"Error in Dota_skill_scraper.py. Can't start script. Error is {e}")


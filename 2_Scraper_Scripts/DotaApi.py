import time
import requests
import json
from Sqldatabasehandler import sqlhandler


class DotaRankedID:
    def __init__(self, host, user, passwd, database, api_key):
        """
        The constructor for the Class
        """
        # too many parameter is increased every time the api rejects our request
        # and it is increased so that the next request is made with a longer delay
        self.toomany = 1
        # Establish connection to the server and create class SQL handler which
        # Runs our queries for us.
        self.sqlhand = sqlhandler(host, user, passwd, database)
        # Set our steam Api Key
        self.api_key = api_key
        # create variable for minimum seq
        self.min_seq = None
        # Last game used to track last game ID in match history
        self.lastgame = 0

    def scrape_history_games(self):
        """
        call this function
        to scrape the games skill level.
        Runs Forever!
        """

        # Get the 3 different skill level game
        while True:
            self.add_new_games_history(1)
            time.sleep(2)
            self.add_new_games_history(2)
            time.sleep(2)
            self.add_new_games_history(3)
            time.sleep(2)

    def scrape_seq_games(self):
        """
        Repeatedly call this function
        to scrape the games stats.
        """
        # Keep Scraping Just Scrape Based on latest sequence number
        while True:
            time.sleep(2)
            for _ in range(0,5):
                res = self.add_new_game_seq()
                time.sleep(2)
            time.sleep(2)
            self.add_new_games_history(2)
            time.sleep(2)
            self.add_new_games_history(3)
            time.sleep(2)

    def add_new_game_seq(self, seq=None):
        """
        Adds new games to the database based on the last seq num in database
        Args:
            seq (int, optional): Defaults to None.
            if it is None it will scrape from the maximum
            Seq number in the database.
            if a int is given it will scrape from that sequence number
        Returns:
            int or None: returns -1 if a error is was found
            returns 1 if not enough games are there to scrape
            doesen't return anything if the function was a success
        """
        # If seq is given no need to check the database we start with the given key
        if seq == None:
            # res = self.sqlhand.SqlQueryExec(
            #     "SELECT min(GameSEQ) FROM DotaMatches WHERE TimeStamp IS NULL", True
            # )
            # if res == -1:
            #     print("Error 1")
            #     return -1

            # Find max seq number in database
            res = self.sqlhand.SqlQueryExec(
                "SELECT GameSEQ FROM Dota.DotaMatches WHERE TimeStamp is not null order by GameSEQ DESC LIMIT 1",
                True,
            )
            if res == -1:
                print("Error 1")
                return -1
            lastgameseq = self.sqlhand.get_row_result()
            self.lastgameseq = lastgameseq
        else:
            lastgameseq = seq
        try:
            # Define api request parameters
            parameters = {
                "key": self.api_key,
                "start_at_match_seq_num": lastgameseq,
                "matches_requested": 500,
            }
            # Define api request url
            response = requests.get(
                "https://api.steampowered.com/IDOTA2Match_570/GetMatchHistoryBySequenceNum/v001/",
                params=parameters,
            )
            data = response.json()
        except:
            # If request was rejected sleep and add to fail in rows
            time.sleep(self.toomany ** 2 * 5)
            self.toomany += 1
            print("Too Many")
            return -1
        # If nothing is returned then throw an error
        if len(data) == 0:
            self.toomany == 1
            return -1
        # if data[status] == 2 that means an error from server
        try:
            if data["result"]["status"] == 2 and len(data) == 1:
                sqlquery = "UPDATE DotaMatches SET GameSEQ=0 WHERE GameSEQ=%s"
                self.sqlhand.SqlQueryExec(
                    sqlquery,
                    False,
                    [
                        lastgameseq,
                    ],
                    True,
                )
                print("Error 2")
                return -1
        except:
            pass
        # Reset fail in rows number
        self.toomany = 1
        # Add the data to database
        return self.add_games_to_database(data)

    def add_games_to_database(self, data):
        """
        is called from the add_new_game_seq function.
        pass the json data returned from the GetMatchHistoryBySequenceNum to add the
        ranked games which have 10 players and no abandons to the database

        Args:
            data (json list): the data returned from the GetMatchHistoryBySequenceNum call

        returns 1 if not enough games are ready to be scraped
        """
        # list to keep the sqlinputs in
        numberadded = 0
        notranked = 0
        sqlinputs = []
        # Start checking games one by one
        try:
            for matches in data["result"]["matches"]:
                try:
                    # check to be a ranked game and also 10 players are present
                    if matches["lobby_type"] == 7 and matches["human_players"] == 10:
                        # Get some basic data about game
                        matchid = matches["match_id"]
                        matchseq = matches["match_seq_num"]
                        matchtime = matches["start_time"]
                        matchresult = matches["radiant_win"]
                        leaver = False
                        # make leaver=True if there was a leaver in game
                        for player in matches["players"]:
                            if player["leaver_status"] != 0:
                                leaver = True
                        # get the picks of both the teams
                        radiantpicks = []
                        direpicks = []
                        for pick in matches["picks_bans"]:
                            if pick["is_pick"] == True:
                                if pick["team"] == 0:
                                    radiantpicks.append(pick["hero_id"])
                                else:
                                    direpicks.append(pick["hero_id"])
                        row = [matchid, matchtime, matchseq, leaver, matchresult]

                        # Sometimes the last pick is left out from the pick_bans list
                        # due to the error in Api. We would go through the players list
                        # and add that one to the list that is not compelete
                        if (
                            len(radiantpicks) != 5
                            or len(direpicks) != 5
                            and len(radiantpicks) + len(direpicks) == 9
                        ):
                            for players in matches["players"]:
                                hero_id = players["hero_id"]
                                if (
                                    len(radiantpicks) == 5
                                    and hero_id not in radiantpicks
                                    and hero_id not in direpicks
                                ):
                                    direpicks.append(hero_id)
                                if (
                                    len(direpicks) == 5
                                    and hero_id not in radiantpicks
                                    and hero_id not in direpicks
                                ):
                                    radiantpicks.append(hero_id)

                        # If our hero list is not complete then don't add that game
                        if len(radiantpicks) != 5 or len(direpicks) != 5:
                            continue
                        for heroes in radiantpicks:
                            row.append(heroes)
                        for heroes in direpicks:
                            row.append(heroes)
                        # This part copies part of the data to be passed in as SQL arguments for the
                        # ON DUPLICATE KEY UPDATE part of the SQL
                        row = row + row[1:]
                        # we need to have 29 data points to pass to sql query
                        if len(row) == 29:
                            sqlinputs.append(row)
                        else:
                            print("STOR RIGHT HERE")
                        numberadded += 1
                    else:
                        notranked += 1
                except:
                    pass
        except:
            print("error Happened")
        try:

            # Make sure we are not in a loop Important DELETE if the current seq game is corrputed!

            if sqlinputs[0][2] != self.lastgameseq:
                sqlquery = "UPDATE DotaMatches SET GameSEQ=0 WHERE GameSEQ=%s"
                self.sqlhand.SqlQueryExec(
                    sqlquery,
                    False,
                    [
                        self.lastgameseq,
                    ],
                    True,
                )

            # add the games
            sqlquery = (
                "INSERT INTO DotaMatches (GameID,TimeStamp,GameSEQ,Leavers,RadiantWin,Pick1Rad,Pick2Rad,Pick3Rad,Pick4Rad,Pick5Rad,Pick1Dir,Pick2Dir,Pick3Dir,Pick4Dir,Pick5Dir) Values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                + "ON DUPLICATE KEY UPDATE TimeStamp = %s,GameSEQ = %s,Leavers = %s,"
                + "RadiantWin = %s,Pick1Rad = %s,Pick2Rad = %s,Pick3Rad = %s,Pick4Rad = %s,Pick5Rad = %s,Pick1Dir = %s,Pick2Dir = %s,Pick3Dir = %s,Pick4Dir = %s,Pick5Dir = %s"
            )

            for row in sqlinputs:
                self.sqlhand.SqlQueryExec(sqlquery, False, row)
            self.sqlhand.Sql_commit_database()

            # TODO: Change from row by row execution to execute many
            # Suc = self.sqlhand.SqlQueryExecmany(
            #     sqlquery,
            #     sqlinputs,
            # )

        except Exception as e:
            print("error 2")
            print(e)
        print(
            "Number of games added are :", numberadded, " and not ranked are", notranked
        )
        if numberadded + notranked < 50:
            print("Waiting For New Games")
            return 1
            time.sleep(2)

    def add_new_games_history(self, skill_bracket):
        """Scrapes latest dota matches based on the skill brackets
        Args:
            skill_bracket (int): The skill bracket we want to
            scrape the games from.

        Returns:
            int or None:
            if -1 is returned it means there was an error
            if 0 is returned means the function worked fine
        """
        # Skillbracket 2 is very high
        # Skillbracket 1 is high
        # Skillbracket 0 is normal
        for _ in range(0, 10):
            try:
                # Define Api call parameters
                parameters = {
                    "key": self.api_key,
                    "skill": skill_bracket,
                }
                # Get a game from the middle of the list
                if self.lastgame != 0:
                    parameters["start_at_match_id"] = self.lastgame
                # Define API call url
                response = requests.get(
                    "https://api.steampowered.com/IDOTA2Match_570/GetMatchHistory/v001/",
                    params=parameters,
                )
                # Request data
                data = response.json()
            except:
                # if failed then sleep and add to fail in row number
                time.sleep(self.toomany ** 2 * 5)
                self.toomany += 1
                print("Too Many")
                # Make another try
                continue
            # if no data is returned throw an error
            if len(data) == 0:
                self.toomany == 1
                return -1
            # reset fail in row number
            self.toomany = 1
            res = self.add_games_to_database_skill_level(data, skill_bracket)
            # return 0 (success) if we are at the end of the list.
            if res == 0:
                return 0

    def add_games_to_database_skill_level(self, data, skill_bracket):
        """
        Must be called from add_new_games_history function

        Args:
            data (json list): the data returned from the GetMatchHistory call
            skill_bracket (int): The skill bracket the games passed have

        returns: -1 if some error happened
                  0 if all the list has been scrapped
                  1 if scrape was succesful but list still has items inside
        """
        # list to keep the sqlinputs in
        sqlinputs = []
        try:
            print(data["result"]["results_remaining"])
            if data["result"]["results_remaining"] == 0:
                self.lastgame = 0
                return 0
            # Start checking games one by one
            for matches in data["result"]["matches"][1:]:
                matchid = str(matches["match_id"])
                self.lastgame = matchid
                # check to be a ranked game and also 10 players are present
                if matches["lobby_type"] == 7:
                    # We only keep the match id and seq num
                    # The rest of the data must be scraped with the
                    # add_new_game_seq() function because GetMatchHistory
                    # doesen't have basic data like the winner of the game
                    matchseq = str(matches["match_seq_num"])
                    skill = str(skill_bracket)
                    row = (matchid, matchseq, skill, skill)
                    if self.min_seq == None:
                        self.min_seq = matchseq
                    elif matchseq < self.min_seq:
                        self.min_seq = matchseq

                    sqlinputs.append(row)
        except:
            print("error Happened")
            return -1
        try:
            # Add data to database
            for inputs in sqlinputs:
                self.sqlhand.SqlQueryExec(
                    "INSERT INTO DotaMatches (GameID,GameSEQ,skill_level) Values (%s,%s,%s) ON DUPLICATE KEY UPDATE skill_level= %s ",
                    False,
                    inputs,
                )
            print(f"Added {len(sqlinputs)} games for skill bracket of {skill_bracket}")
            self.sqlhand.Sql_commit_database()

            # TODO : Turn row by row execution to execute many
            # Suc = self.sqlhand.SqlQueryExecmany(
            #         "INSERT INTO DotaMatches (GameID,GameSEQ,skill_level) Values (%s,%s,%s) ON DUPLICATE KEY UPDATE skill_level= %s ",
            #         sqlinputs,
            #     )

        except Exception as e:
            print("error 2")
            print(e)
            return -1

        return 1


# TODO: There are 2 todos in middle of the program
# TODO: Need to find a solution so that we get both the picks and the winner. meaning that add_new_games_history
# and add_new_game_seq are not getting data for the same matches therefore there is very little overlap in their data
# therefore we can't use the game level for our machine learning.
# NOTE: One solution could be that two account simutaniously scrape. One just focuses on add_new_games_history and
# The other focuses on add_new_game_seq
from DotaApi import DotaRankedID
import json

if __name__ == "__main__":
    # Define Dota game scraper and create database connection
    try:
        # Define Dota game scraper and create database connection
        with open("keys.json") as f:
            keys = json.load(f)
        host = keys["database"]["host"]
        print(host)
        something = DotaRankedID(
            host=keys["database"]["host"],
            user=keys["database"]["user"],
            passwd=keys["database"]["passwd"],
            database=keys["database"]["database"],
            api_key=keys["api_key"],
        )
        # Call Scrape games

        something.scrape_history_games()
    except Exception as e:
        print(f"Error in Dota_skill_scraper.py. Can't start script. Error is {e}")

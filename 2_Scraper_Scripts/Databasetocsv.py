import pymysql
import pandas
from Sqldatabasehandler import sqlhandler
import json

class Databasetocsv:
    def __init__(self, host, user, passwd, database, filename="dota_games.csv"):
        self.sqlhand = sqlhandler(host, user, passwd, database)
        self.filename = filename

    def export(self):
        query = "SELECT * FROM DotaMatches WHERE RadiantWin is not null"
        results = pandas.read_sql_query(query, self.sqlhand.mydb)
        results.to_csv(self.filename, index=False)

if __name__ == "__main__":
    try:
        # Define Dota game scraper and create database connection
        with open('keys.json') as f:
            keys = json.load(f)
        host = keys['database']['host']
        print(host)
        something = Databasetocsv(
            host= keys['database']['host'],
            user= keys['database']['user'],
            passwd=keys['database']['passwd'],
            database=keys['database']['database'],
        )
        something.export()
    except:
        pass


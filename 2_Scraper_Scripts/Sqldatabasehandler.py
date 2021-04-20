import mysql.connector
import pymysql
import time


class sqlhandler:
    def __init__(self, host, user, passwd, database):
        """
        The constructor for the Class
        Sets the database connection credentials
        """
        self.host = host
        self.user = user
        self.passwd = passwd
        self.database = database
        # Establish connection to the server
        for _ in range(0, 5):
            try:
                self.__init_db()
                break
            except:
                print("Connection Failed to SQL")

    def __init_db(self):
        """
        Re-establish connection to the server
        """
        try:
            self.mydb = pymysql.connect(
                host=self.host,
                user=self.user,
                passwd=self.passwd,
                database=self.database,
            )

            self.mycursor = self.mydb.cursor()
        except:
            pass

    def SqlQueryExec(self, Query, givecount=False, sqlinput=None, commitdatabase=False):
        """Tries to Execute a query and if failed tries 7 times.
        get row result must be used to fetch the result
        Args:
            Query (string): [The Sql query to execute]
            givecount (bool, optional): [if give count is True the ruturn value 0 means the query executes.
            If it's true this command is useful for select query that returns the number of rows]. Defaults to False.
            sqlinput ([type], optional): [The inputs of the query. Defualt None executes query without an input]. Defaults to None.
            commitdatabase (bool, optional): [Set to true if a change in the database has been done and has to be commited]. Defaults to False.

        Returns:
            [type]: [-1 if there was an error. if successful it varies based on the input]
        """
        for _ in range(0, 7):  # try 7 times
            try:
                # execute with or without input based on sqlinput
                if sqlinput is None:
                    self.mycursor.execute(Query)
                else:
                    self.mycursor.execute(Query, sqlinput)
                # if commit is true the database is committed
                if commitdatabase is True:
                    self.mydb.commit()
                # return 0 is givecount is false or rows if givecount is true
                if givecount is False:
                    return 0
                count = 0
                for self.db in self.mycursor:
                    count += 1
                return count
            except Exception as e:
                # If query execution failed reset the database
                print(f"In SqlQueryExec this happened: {e}")
                time.sleep(1)
                self.__init_db()
        # If there was success in Query we shouldn't have reached this line
        print("Couldn't Excecute" + Query + " in SqlQuery Function in Scraper Class")
        return -1

    def SqlQueryExecmany(self, query, sqlinputs=None, commitdatabase=True):
        """[Runs many sql queries at the same time. useful for large amounts of data]

        Args:
            Query (string): [Sql query]
            inputs ([list]) : gives the inputs for the queries
            commitdatabase (bool, optional): [to commit. This should be True unless its and exceptional case]. Defaults to True.
        """
        if sqlinputs is not None:
            Suc = self.mycursor.executemany(
                query,
                sqlinputs,
            )
        else:
            Suc = self.mycursor.executemany(
                query,
            )

        self.mydb.commit()  # Commit to database
        return Suc

    def get_row_result(self, rownumber=0):
        """[summary]

        Args:
            rownumber (int, optional): [The row result we want to be returned]. Defaults to 0.
            Meaning the top row will be returned as default

        Returns:
            [type]: [Returns the result of row number
        """
        return self.db[rownumber]

    def Sql_commit_database(self):
        """
        Commits the database to make the changes
        """
        try:
            for _ in range(5):
                self.mydb.commit()
        except:
            pass
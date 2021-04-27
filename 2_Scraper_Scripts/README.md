In this step, we need to create a scraper that actively scrapes the finished Dota 2 games and adds them to the MySQL database we created in the step before. The instructions are written below in order.

1) First, get a steam API key go from https://steamcommunity.com/dev
2) Put the API key and the MySQL authentication from the step before in keys.json in the folder.
3) Create a Computer Engine (We tested on Google Cloud) with the minimum computational power and Ubuntu operating system. Install Python 3.8.8 on that Engine and make sure the pip command works in the terminal.
4) To handle the dependencies of the scripts, two approaches could be taken. Either pip installing the libraries or using conda. Using pip install is faster because not many dependencies to handle, and install conda would take a lot of unnecessary time. The list below shows the required packages. Run these lines one by one.
```
pip install pandas==1.2.2
pip install pymysql==1.0.2
pip install mysql-connector-python==8.0.23
```
5) Upload the files in this folder to the computer engine.
6) Install screen on the terminal with the code below:
```
apt-get install screen
```
screen needs to be installed on the computer engine so after the terminal is closed, the scripts keep working. create a new session with the command
```
Screen -S Dotasession
```
7) Run the code python Dota_picks_scraper.py and then detach the screen by pressing Ctrl + Alt + D and then close the terminal.

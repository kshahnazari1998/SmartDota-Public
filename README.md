# SmartDota.com

The results of the data pipeline could be seen live on https://smartdota.com

![Smartshot](https://user-images.githubusercontent.com/56759053/112795700-bfb9ac80-901d-11eb-8254-ca92612c7a7d.PNG)


The front end only works for PC users because the users of this websites would be playing on their PC at that moment and would use their PC to work with the website. Although the flutter front end is present because the Flask API exists other things like mobile applications and other softwares could also be built on top of it.

This Repo gives the insturctions to recreate the smartdota.com website.


---
## Description
The Aim of this project is to suggest a dota2 hero to players to maximize their chances of winning.

To get a better idea of how the hero system and picking in dota 2 works you could take at the first 5 minutes of this link https://www.youtube.com/watch?v=mcu3Pp6ZASM

What is great about this project is that because the company that created this game wants to keep the game fresh and players engaged it will give ocassional updates that will change the way the game works therefore the game is always changing and something that was an advantage a month ago might have changed in a way that it is an disadvantage right now. 

With this in mind this project makes a pipeline which collects the games everyday from steam api and adjusts the model to keep up with changes of the game and be always up to date. To make this work there are 5 different components that need to be deployed in order for this system to work.

---

### Components

There are 5 different parts that need to be deployed in order for the system to work. They are in the following order:

1- Database Builder: Create a MySQL database so that newly played games could be stored and used later on.

2- Scraper Scripts: Script that calls the Steam API every second to collect the results of the newly finished games.

3- Flask API: An API that has the latest updated model that would be called by the Flutter front end and gives the predictions to it.

4- Model Updater: After a certain amount of games have been added to the database collects those new datas and trains the model one time on that batch to keep the model up to date with the latest changes in the dota2 game.

5- Flutter Frontend: A front end that people could use and get predictions from the model. The predictions would come from the fluttler app sending a call to the Flask API.

---

### Notes

The instructions of deployment of each folder is put inside the readme file of that folder.

There is a docs folder which does not need to be deployed but is put there to show the process of model selection and the tuning of the models.

---

# Contact Me

I would like to get in touch to discuss anything Data Science related. You could connect with me on linkedin with the name Kevin Shahnazari (Mention in the notes that you are from here) or you could send an email to kshahnazari@gmail.com

Optional: You could delete the old trained models to free up more space. They are just used as a checkpoint. 

The code here reads the database from time to time, and if there is enough data inside, it will train a new model. To deploy your new model, the steps below need to be taken.

1) Fill the keys.json file with the information from the 1st step of the pipeline. Or you could replace this file with the keys.json file used in the 2nd step of the pipeline.
2) Create a Computer engine with a RAM of at least 4 GB and a reasonable amount of CPU (Medium size in most platforms would suffice). 
3) install python 3.8.8 and pip install the libraries below onto that Computer Engine:
```
pip install pandas==1.2.2
pip install pymysql==1.0.2
pip install mysql-connector-python==8.0.23
pip install numpy==1.19.2
pip install pytorch==1.8.0
```

4) Upload the files in this folder to the Computer Engine/

4) We want the script to train and upload the new models every hour. Todo, so we need crontab to do this.
```
#10 * * * * cd FOLDERWITHSCRIPT && python train_new_model.py
#15 * * * * cd FOLDERWITHSCRIPT && curl -X POST https://xxx.herokuapp.com/file -F 'app=@model.pth' -F pass=xxx
```
Change the xxx to the password you set in the 3rd step in the pipeline for the Flask application and change `FOLDERWITHSCRIPT` to the directory you uploaded this folder to Compute Engine.
The last step is to change the xxx in the Heroku app to the URL of your Flask URL.


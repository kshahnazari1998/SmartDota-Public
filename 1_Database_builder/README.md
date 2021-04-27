 
The first part of the pipeline is to create a DataBase for the Games to be saved to. You could use whatever service you want to host your database, but our recommendation is to use either AWS or google cloud. We used AWS to create the database. 

When setting up MySQL, select version 8.0.20; using the newer versions would probably not break the system because basic SQL commands are being used in the scripts. Make sure the name of the database you create is called `Data`

After setting up the database, either copy and paste the SQL code in this folder and execute or use an interface to upload the file and then execute the commands.

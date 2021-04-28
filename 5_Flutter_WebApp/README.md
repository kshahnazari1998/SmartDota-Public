The final step is to deploy the flutter webapp so that the users have a interface to communicate with the models we just built. 

1) After installing flutter create a new flutter Application (with web support, Read more at https://flutter.dev/docs/get-started/web) and replace the assets, lib folder with the ones in this folder. Also match the pubspec.yaml with the ones in this folder.

2) Change line 475 of the url of your heroku App Flask API you got from the 3rd step of the pipeline

3) Build flutter app to test if the website is functioning properly with the command
`flutter build web`

4) The final step is to deploy the flutter app to a server. I prefer firebase because it has the easiest and chepest method of deployment but choose whatever is more suitable for your needs. (https://flutter.dev/docs/deployment/web)

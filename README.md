# Disease-Prediction-App

## Preparing the UI

Pre-Requisite: Node-JS, NPM, yarn

Make sure you are in the ``` UI ``` directory.

Install the dependencies using the command ``` yarn install ``` 

run the following commands to start the server.
```
npm install -g serve
npm run build
serve -s build -l 3000
```

consequently any changes will require the later 2 command to be run. Else only the last cmd will be enough.
Now visit the localhost:3000 to ensure the ui is up and running.

## Starting the flask app.

Make sure you have a virtual environment of python and source into it.
Install the requirements using ``` pip install -r requirements.txt ```

```
FLASK_APP=app.py flask run
```
In case of Powershell try the following two commands if the earlier gives error.
```
$env:FLASK_APP = "app.py"
flask run
```
Running the above command will start up the service to interact with front end.
Note : we will need to run these on two different shells.

## Deployment of a trained Neural Network using Flask to predict Titanic survival

This repository serves as an example how you can deploy a trained ML model using Flask and containerize the microservice using Docker.

First, train 3 models:

    $ python3 run.py train

All models will be saved and the best model's name will be written 
into a text file.

To make predictions using the best trained model run the Flask app:

    $ python3 app.py

The best model will be loaded and used to make predictions. 

The application can also be deployed using a docker container. 

For this, pull this repository. 

    $ git clone --branch main https://<access_token>@github.com/Simon1307/Deployed-ML-Model-Flask.git

Then, cd into the cloned repository and build the docker image.

    $ sudo docker build . -t titanic-prediction

Run the docker image.

    $ sudo docker run -p 5000:5000 titanic-prediction

Now the web application is available at: http://127.0.0.1:5000

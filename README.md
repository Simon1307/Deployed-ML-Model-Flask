## Deployment of a trained Neural Network using Flask to predict Titanic survival
First, train 3 models:

    $ python3 run.py train

All models will be saved and the best model's name will be written 
into a text file.

To make predictions using the best trained model run the Flask app:

    $ python3 app.py

The best model will be loaded and used to make predictions. 

The application can also be deployed using a docker container.

1. For this, pull this repository 


    $ git clone --branch main https://<access_token>@github.com/Simon1307/Deployed-ML-Model-Flask.git


2. cd into the cloned repository
3. Build docker image


    $ sudo docker build . -t titanic-prediction


4. Run the docker image


    $ sudo docker run -p 5000:5000 titanic-prediction


5. Now the web application is available at: http://127.0.0.1:5000


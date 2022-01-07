## Deployment of a Neural Network using Flask
First, train 3 models:

$ python3 run.py train

All models will be saved and the best model's name will be written 
into a text file.

To make predictions using the best trained model run the Flask app:

$ python3 app.py

The best model will be loaded and used to make predictions. 

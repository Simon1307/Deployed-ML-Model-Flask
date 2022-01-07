import numpy as np
from flask import Flask, request, render_template
import json
import tensorflow as tf

app = Flask(__name__)


with open('./src/resources/models/best_model.txt') as f:
    experiment_name = json.load(f)

# load best model
filepath = "./src/resources/models/" + experiment_name
model = tf.keras.models.load_model(filepath)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    new_obs = [float(x) for x in request.form.values()]
    new_obs = np.array([new_obs])
    new_obs = tf.expand_dims(tf.convert_to_tensor(new_obs), axis=0)
    prediction = model.predict(new_obs)
    if prediction >= 0.5:
        prediction = "survived"
    else:
        prediction = "died"
    return render_template('index.html', prediction_text='The person {}!'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)

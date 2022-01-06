import numpy as np
import tensorflow as tf


def prepare_data(dataset, data):
    if dataset == "train":
        # shuffle data
        data = data.sample(frac=1)
       
    columns = ["Pclass",
               "Sex",
               "SibSp",
               "Parch",
               "Fare"]

    x = data[columns]
    x['Sex'] = x['Sex'].map({'male': 1, 'female': 0})
    # Normalize Fare feature
    x["Fare"] =(x["Fare"] - x["Fare"].mean()) / x["Fare"].std()

    x = x.to_numpy()
    x = tf.convert_to_tensor(x, dtype=tf.float32) 
    x = tf.expand_dims(x, axis=1)

    if dataset == "train":
        y = data["Survived"]
        y = y.to_numpy()
        y = tf.convert_to_tensor(y, dtype=tf.int32) 
        return x, y
    else:
        passenger_ids = data["PassengerId"]
        return x, passenger_ids


def parse_observation(obs: list) -> np.array:
    """Reshape a list into a 1 x len(obs) numpy array"""
    new_obs = np.array(obs).reshape(1, -1)
    return new_obs

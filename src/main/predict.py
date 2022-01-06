import numpy as np
import json
import logging
import tensorflow as tf
from src.utils.data_utils import parse_observation
from src.utils.config import get_default

logger = logging.getLogger(__name__)

VALUE = get_default('predict', 'value')


def main():
    with open('./src/resources/models/best_model.txt') as f:
        experiment_name = json.load(f)
            
    # load best model
    filepath = "./src/resources/models/" + experiment_name
    model = tf.keras.models.load_model(filepath)

    # Get observation from config file predict.json
    new_obs = parse_observation(VALUE)
    new_obs = tf.expand_dims(tf.convert_to_tensor(new_obs), axis=0)
    prediction = model.predict(new_obs)
    logger.info(f'Predicted class: {prediction}')
    

if __name__ == '__main__':
    main()

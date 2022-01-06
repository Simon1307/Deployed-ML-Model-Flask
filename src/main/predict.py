import numpy as np
import sys
import json
import tensorflow as tf
sys.path.append('../../src/')


def main():
    with open('../resources/models/best_model.json') as f:
        experiment_name = json.load(f)
            
    # load best model
    filepath = "./resources/models/" + experiment_name
    model = tf.keras.models.load_model(filepath)

    new_obs = np.array([[3., 1.,  0.,  0., -0.49721327]])

    prediction = model.predict(new_obs)
    print('Predicted class:', prediction)
    

if __name__ == '__main__':
    main()

import pandas as pd
import tensorflow as tf
import numpy as np
import os
import json
import sys
import warnings
warnings.filterwarnings('ignore')
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from model import Network
from optimizer import Optimizer
from utils.data_utils import prepare_data
from utils.visualization_utils import plot_experiment_results


# tf.autograph.set_verbosity(0)
# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)


def run_experiment(i, hyperparameter, x_train, y_train, x_val, y_val):
    """Run experiment.
    Runs one experiment with a given hyperparameter combination and stores the trained model in ../resources/models/ .
    Args:
        i: Index of the experiment.
        hyperparameter: Dictionary storing all hyperparameters.
        x_train, y_train, x_val, y_val: Train and test data (tensors).

        Returns:
            History of train and test accuracies and binary cross entropy losses
    """

    # Set seeds
    tf.random.set_seed(1)
    np.random.seed(1)

    batch_size = hyperparameter["mb"][i]
    learning_rate = hyperparameter["lr"][i]
    epochs = hyperparameter["epochs"][i]
    units = hyperparameter["units"][i]
    dropout = hyperparameter["dropout"][i]
    activation = hyperparameter["activation"][i]

    model = Network(units=units, dropout=dropout, activation=activation)
    opt = Optimizer(model,
                    mb=batch_size,
                    lr=learning_rate,
                    loss=tf.keras.losses.BinaryCrossentropy,
                    opt=tf.keras.optimizers.Adam)

    tr_loss, tr_acc, ts_loss, ts_acc = opt.run(x_train,
                                               y_train,
                                               x_val,
                                               y_val,
                                               epochs=epochs,
                                               verbose=0)

    # print(model.summary())
    if hyperparameter["save_model"][i]:
        if not os.path.exists('../resources/models'):
            os.mkdir("../resources/models")
        filepath = "../resources/models/" + hyperparameter["name"][i]
        tf.keras.models.save_model(model, filepath=filepath)
        print(f"Model weights save to: {hyperparameter['name'][i]}")

    print("Validation accuracy:", ts_acc[-1])
    return tr_loss, tr_acc, ts_loss, ts_acc


def main():
    train = pd.read_csv("../resources/train.csv")

    x_train, y_train = prepare_data("train", train)

    # Split train data into train and validation
    x_val = x_train[-150:]
    x_train = x_train[:-150]

    y_val = y_train[-150:]
    y_train = y_train[:-150]

    y_val = tf.reshape(y_val, shape=(len(y_val), 1, 1))
    y_train = tf.reshape(y_train, shape=(len(y_train), 1, 1))

    print("Training data loaded with shapes:")
    print(f"x_train: {x_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"x_train: {x_val.shape}")
    print(f"y_train: {y_val.shape}")

    config_path = '../conf/config.json'  # '../conf/test_config.json'
    with open(config_path) as f:
        experiments = json.load(f)

    all_metrics = []
    for i in range(len(experiments["name"])):
        print(f"Experiment {i + 1} running...")
        tr_loss, tr_acc, ts_loss, ts_acc = run_experiment(i,
                                                          experiments,
                                                          x_train,
                                                          y_train,
                                                          x_val,
                                                          y_val)

        all_metrics.append([tr_loss, tr_acc, ts_loss, ts_acc])
        print("\n")
    print("-" * 20)

    if not os.path.exists('../resources/results'):
        os.mkdir('../resources/results')

    for i in range(len(experiments["name"])):
        epochs = experiments["epochs"][i]
        experiment_name = experiments["name"][i]
        tr_loss = all_metrics[i][0]
        tr_acc = all_metrics[i][1]
        ts_loss = all_metrics[i][2]
        ts_acc = all_metrics[i][3]

        plot_experiment_results(epochs,
                                experiment_name,
                                tr_loss,
                                tr_acc,
                                ts_loss,
                                ts_acc)
        print("-" * 50)

    # determine the best model
    val_accuracies = []
    for i in range(len(all_metrics)):
        # Evaluate model based on validation accuracy in last epoch
        val_accuracies.append(all_metrics[i][-1][-1])
    best_exp_idx = np.argmax(val_accuracies)
    experiment_name = experiments["name"][best_exp_idx]
    print("Best performing model:", experiment_name)

    with open('../resources/models/best_model.txt', 'w') as f:
        json.dump(experiment_name, f)


if __name__ == '__main__':
    main()

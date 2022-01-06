import matplotlib.pyplot as plt
import numpy as np


def plot_experiment_results(epochs, experiment_name, tr_loss, tr_acc, ts_loss, ts_acc):     
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(experiment_name)
    fig.subplots_adjust(wspace=0.6)

    ax1.set_title("Mean BCE")
    ax1.plot(np.arange(1, epochs+1), tr_loss, label="Train")
    ax1.plot(np.arange(1, epochs+1), ts_loss, label="Titanic Challenge")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    
    ax2.set_title("Mean Accuracy")
    ax2.plot(np.arange(1, epochs+1), tr_acc, label="Train")
    ax2.plot(np.arange(1, epochs+1), ts_acc, label="Titanic Challenge")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    
    filepath = './src/resources/results' + experiment_name
    fig.savefig(filepath)

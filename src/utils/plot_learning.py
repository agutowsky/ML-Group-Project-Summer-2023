import matplotlib.pyplot as plt
from utils.config import load_config
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
import os


def plot_metric(history, metric, hypermodel=False, hyperparams=None):
    """
    Plot a graph of the metric vs. epochs.

    Args:
        history (dict): The history object returned by model.fit()
        metric (str): The metric to plot
    """
    # Clear the current figure
    plt.clf()

    # Load the config values
    if hyperparams == None:
        config: dict = load_config(src="./config.yaml")
        epochs = config["params"]["epochs"]
        learningrate = config["params"]["learning_rate"]
        momentum = config["params"]["momentum"]
        hiddenlayers = config["params"]["hidden_layers"]
        hiddenunits = config["params"]["hidden_units"]
        batchsize = config["params"]["batch_size"]
        activation = config["params"]["activation"]
    else:
        # Hyperparam tuning model values
        # epochs = hyperparams.get('epochs')
        epochs = 50
        learningrate = hyperparams.get('learning_rate')
        momentum = hyperparams.get('momentum')
        hiddenlayers = hyperparams.get('hidden_layers')
        hiddenunits = hyperparams.get('hidden_units')
        # batchsize = hyperparams.get('batch_size')
        batchsize = 1
        activation = hyperparams.get('activation')

    # Set the title and y-axis label
    if "accuracy" in metric:
        plt.title("Accuracy per Epoch %")
        plt.ylabel("Accuracy %")
    else:
        plt.title("Loss per Epoch")
        plt.ylabel("Loss")

    # Label the x-axis
    plt.xlabel("Epoch")

    # Plot the graph
    plt.plot(history.history[metric])
    plt.plot(history.history[f"val_{metric}"])

    # Add a legend
    plt.legend(["train", "test"], loc="upper left")

    # Save the graph

    if hypermodel:
        if not os.path.exists(f"graphs/hypermodel/{activation}"):
            os.makedirs(f"graphs/hypermodel/{activation}")
        plt.savefig(
            f"graphs/hypermodel/{activation}/{metric}_e{epochs}_lr{learningrate}_m{momentum}_hl{hiddenlayers}_hu{hiddenunits}_bs{batchsize}.png"
        )
        print(
            f"Saved graph of {metric} per epoch to graphs/hypermodel/{activation}/{metric}_e{epochs}_lr{learningrate}_m{momentum}_hl{hiddenlayers}_hu{hiddenunits}_bs{batchsize}.png"
        )
    else:
        if not os.path.exists(f"graphs/{activation}"):
            os.makedirs(f"graphs/{activation}")
        plt.savefig(
            f"graphs/{activation}/{metric}_e{epochs}_lr{learningrate}_m{momentum}_hl{hiddenlayers}_hu{hiddenunits}_bs{batchsize}.png"
        )
        print(
            f"Saved graph of {metric} per epoch to graphs/{activation}/{metric}_e{epochs}_lr{learningrate}_m{momentum}_hl{hiddenlayers}_hu{hiddenunits}_bs{batchsize}.png"
        )


def plot_confusion_matrix(Y_true, Y_pred_classes, num_classes, hypermodel=False, hyperparams=None):
    # Clear the figure
    plt.clf()

    # Load the config values
    if hyperparams == None:
        config: dict = load_config(src="./config.yaml")
        epochs = config["params"]["epochs"]
        learningrate = config["params"]["learning_rate"]
        momentum = config["params"]["momentum"]
        hiddenlayers = config["params"]["hidden_layers"]
        hiddenunits = config["params"]["hidden_units"]
        batchsize = config["params"]["batch_size"]
        activation = config["params"]["activation"]
    else:
        # Hyperparam tuning model values
        # epochs = hyperparams.get('epochs')
        epochs = 50
        learningrate = hyperparams.get('learning_rate')
        momentum = hyperparams.get('momentum')
        hiddenlayers = hyperparams.get('hidden_layers')
        hiddenunits = hyperparams.get('hidden_units')
        # batchsize = hyperparams.get('batch_size')
        batchsize = 1
        activation = hyperparams.get('activation')

    cm = confusion_matrix(Y_true, Y_pred_classes)
    print("Confusion Matrix:")
    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    # Adding the number of each occurrence to the grid
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    # Create the output path if it doesn't exist
    if not os.path.exists(f"graphs/{activation}"):
        os.makedirs(f"graphs/{activation}")

    # Set the output path and include hyperparameters in the filename
    if hypermodel:
        if not os.path.exists(f"graphs/hypermodel/{activation}"):
            os.makedirs(f"graphs/hypermodel/{activation}")
        output_path = f"./graphs/hypermodel/{activation}/confusion_matrix_e{epochs}_lr{learningrate}_m{momentum}_hl{hiddenlayers}_hu{hiddenunits}_bs{batchsize}.png"

    # Save the plot as an image file
    plt.savefig(output_path)
    print(f"Saved confusion matrix to {output_path}")

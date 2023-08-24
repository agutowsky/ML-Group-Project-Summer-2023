"""
Alexander Gutowsky, Ethan Saftler, Gatlin Newhouse, Manisha Katta, Rohan Reddy
CS 445/545 | Portland State University | ML Group Assignment Summer 2023
"""

import argparse
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
import pandas as pd
import numpy as np
from utils.config import load_config, pretty_print_config
from utils.plot_learning import plot_metric, plot_confusion_matrix
from utils.train_test_split import train_test_split
from utils.preprocess import preprocess_csv, balance_dataset


def main():
    # Command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--preprocess", action="store_true", help="Preprocess CVD Dataset"
    )
    args = parser.parse_args()

    # Reading in configuration options
    print("- Attempting to read in config file from './config.yaml'")
    try:
        config: dict = load_config(src="./config.yaml")
        print(
            "Successfully obtained the following configuration options from './config.yaml'"
        )
        pretty_print_config(config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "The specified config file does not exist. Is it located in './config.yaml'?"
        )
        exit(1)

    # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experimental/SGD
    optimizer = tf.keras.optimizers.experimental.SGD(
        learning_rate=config["params"]["learning_rate"],
        momentum=config["params"]["momentum"],
        nesterov=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="SGD",
    )

    # Preprocessing, if the user supplied the command line argument
    if args.preprocess:
        print(
            f"Attempting to preprocessing datafile from '{config['datasets']['src']}'"
        )
        preprocess_csv(src=config["datasets"]["src"], dest=config["datasets"]["dest"])
        print("Done.")
        print(
            f"Successfully preprocessed dataset. File is stored at '{config['datasets']['dest']}'"
        )

    # Loading dataset into memory
    try:
        data: pd.DataFrame = pd.read_csv(config["datasets"]["dest"])
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "The specified CSV file does not exist. Is the file properly preprocessed?"
        )
        print("Usage: py main.py -p")
        exit(1)

    # X: Features
    # Y: Targets
    X, Y = balance_dataset(data)
    # X_train: Training features
    # X_test:  Testing features
    # Y_train: Training targets
    # Y_test:  Testing targets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, config["params"]["test_size"], config["params"]["seed"]
    )

    input_shape = (18,)
    num_classes = 2
    model = tf.keras.Sequential()

    # Generating the hidden layers
    for _ in range(config["params"]["hidden_layers"]):
        model.add(
            tf.keras.layers.Dense(
                units=config["params"]["hidden_units"],
                activation=config["params"]["activation"],
                input_shape=input_shape,
            )
        )
    # Output Layer
    model.add(tf.keras.layers.Dense(units=num_classes, activation="softmax"))
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Standardize the features using a scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert labels to one-hot encoding
    Y_train_onehot = tf.keras.utils.to_categorical(Y_train, num_classes)
    Y_test_onehot = tf.keras.utils.to_categorical(Y_test, num_classes)

    # Train the model
    print("Beginning Training")
    history = model.fit(
        X_train_scaled,
        Y_train_onehot,
        epochs=config["params"]["epochs"],
        batch_size=config["params"]["batch_size"],
        validation_data=(X_test_scaled, Y_test_onehot),
        verbose=1,
    )

    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(X_test_scaled, Y_test_onehot, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Make predictions on the test data
    Y_pred = model.predict(X_test_scaled)
    Y_pred_classes = np.argmax(Y_pred, axis=1)  # Convert one-hot encoded predictions to classes

    # Calculate and print classification report
    class_report = classification_report(Y_test, Y_pred_classes)
    print("Classification Report:")
    print(class_report)

    # Call the new function to plot the confusion matrix
    num_classes = len(np.unique(Y_test))  # Get the number of unique classes
    plot_confusion_matrix(Y_test, Y_pred_classes, num_classes)

    # Generate graphs per batch size or epoch depending on the number of epochs
    plot_metric(history, "accuracy")
    plot_metric(history, "loss")

    print("Done.")


if __name__ == "__main__":
    main()

import itertools
import os
from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from six.moves import range
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

from consts import (BASE_LOG_DIR, CLASSES, IMAGE_SIZE, MODEL_CHECKPOINTS,
                    TEST_DIR)
from utils import MODELS


class ModelTester():
    """
        Class used to evaluate a model and upload logs to tensorboard
    """

    def __init__(self, model, data):
        """
            Constructor

        Args:
            data (numpy.ndarray):   Data used to create the confusion matrix
                                    and evaluation metrics
                                    (Inputs, Labels)
        """

        # Model which to evaluate
        self.model = model

        # Data used for evaluation
        self.data = data
        self.data = self.data.unbatch()

        self.inputs = []
        self.labels = []

        # Unbatch the evaluation data and convert to numpy array
        for images, labels in self.data.take(-1):
            self.inputs.append(images.numpy())
            self.labels.append(labels.numpy())

        self.inputs = np.array(self.inputs)
        self.labels = np.array(self.labels)

        del self.data

        # Timestamp used for logging
        self.tag = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        # Model prediction
        self.predictions = None

    def plot_to_image(self, figure):
        """
            Converts the matplotlib plot specified by 'figure' to a PNG image
            and returns it. The supplied figure is closed and inaccessible
            after this call.
        """

        buff = BytesIO()

        # Save the plot to a PNG in memory.
        plt.savefig(buff, format='png')

        buff.seek(0)

        # Convert the PNG buffer to a TF image which uses 4 channels
        image_ = tf.image.decode_png(buff.getvalue(), channels=4)

        # Add batch dimension
        image_ = tf.expand_dims(image_, 0)

        return image_

    def plot_confusion_matrix(self, cm):
        """
            Plots the confusion matrix

        Args:
            cm (numpy.ndarray): Confusion matrix

        Returns:
            matplotlib.figure.Figure: Plot of the confusion matrix
        """

        # Create plot figure
        figure = plt.figure(figsize=(12, 12))

        # Plot details
        plt.imshow(cm.T, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        # Plot x's and y's
        tick_marks = np.arange(len(CLASSES))
        plt.xticks(tick_marks, CLASSES, rotation=45)
        plt.yticks(tick_marks, CLASSES)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[1]), range(cm.shape[0])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.xlabel('True Label')
        plt.ylabel('Predicted Label')

        return figure

    def upload_confusion_matrix(self):
        """
            Method to create and upload confusion matrix to
            tensorboard for vizualization
        """

        # Check if the predictions are already present
        if self.predictions is None:
            # If not use the model to get the predictions
            self.predictions = self.model.predict(self.inputs)
            self.predictions = np.argmax(self.predictions, axis=1)

        # Path where the logs are saved to
        LOG_DIR = f"{BASE_LOG_DIR}/testing/cm/{self.model.name}" + self.tag

        # Object used to save the confusion matrix
        self.file_writer = tf.summary.create_file_writer(LOG_DIR)
        self.file_writer.set_as_default()

        # Get the confusion matrix from the predictions and labels
        confusion_matrix_ = confusion_matrix(self.labels, self.predictions)

        # Create figure
        figure = self.plot_confusion_matrix(confusion_matrix_)

        # Plot the figure
        cm_image = self.plot_to_image(figure)

        # Save plot to be vizualized in tensorboard
        tf.summary.image("Confusion Matrix", cm_image, step=-1)

        return

    def upload_metrics(self):
        """
            Method to calculate evaluation metrics and upload them
            to tensorboard for vizualization
        """

        # Check if the predictions are already present
        if self.predictions is None:
            # If not use the model to get the predictions
            self.predictions = self.model.predict(self.inputs)
            self.predictions = np.argmax(self.predictions, axis=1)

        # Path where the logs are saved to
        LOG_DIR = f'{BASE_LOG_DIR}/testing/metrics/{self.model.name}' \
                  + self.tag

        # Object used to save the confusion matrix
        self.file_writer = tf.summary.create_file_writer(LOG_DIR)
        self.file_writer.set_as_default()

        # F1 metric
        test_f1 = f1_score(self.labels,
                           self.predictions,
                           average='macro')

        # Recall metric
        test_recall = recall_score(self.labels,
                                   self.predictions,
                                   average='macro')

        # Precission metric
        test_precission = precision_score(self.labels,
                                          self.predictions,
                                          average='macro')

        # Accuracy metric
        test_accuracy = accuracy_score(self.labels,
                                       self.predictions)

        # Save metrics to be vizualized in tensorboard
        tf.summary.scalar('test_f1 score', data=test_f1, step=-1)
        tf.summary.scalar('test_recall', data=test_recall, step=-1)
        tf.summary.scalar('test_precision', data=test_precission, step=-1)
        tf.summary.scalar('test_accuracy', data=test_accuracy, step=-1)

        return

    def uploat_heatmap(self):
        """
            Method to create and upload class heatmap to tensorboard
            for vizualization
        """

        # Check if the predictions are already present
        if self.predictions is None:
            # If not use the model to get the predictions
            self.predictions = self.model.predict(self.inputs)
            self.predictions = np.argmax(self.predictions, axis=1)

        # Path where the logs are saved to
        LOG_DIR = f'{BASE_LOG_DIR}/testing/report/{self.model.name}' \
                  + self.tag

        # Object used to save the confusion matrix
        self.file_writer = tf.summary.create_file_writer(LOG_DIR)
        self.file_writer.set_as_default()

        # Classification report
        report = classification_report(self.labels, self.predictions,
                                       target_names=CLASSES,
                                       output_dict=True)

        # Create figure
        plt.figure(figsize=(12, 12))

        # Create heatmap from the report
        figure = sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)

        # Convert to tensorflow image format
        plot = self.plot_to_image(figure)

        # Save image to be vizualized in tensorboard
        tf.summary.image("Classification Report", plot, step=-1)

        return


def main():
    # Load test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(TEST_DIR,
                                                          image_size=IMAGE_SIZE,
                                                          batch_size=32,
                                                          seed=42)

    # Normalization layer used to normalize the test dataset
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    # Normalize test dataset
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    # Loop to load model and evaluate it
    for checkpoints_dir in os.listdir(MODEL_CHECKPOINTS):
        # Path where the model checkpoints are saved to
        path = os.path.join(MODEL_CHECKPOINTS, checkpoints_dir)

        # Latest trained model weights
        model_weights = tf.train.latest_checkpoint(path)

        # Model architecture to be used
        architecture = path.split('/')[-1]

        # Get model
        model = MODELS[architecture]

        # Load the trained weights
        model.load_weights(model_weights)

        # Create model tester
        tester = ModelTester(model, test_ds)

        # Evaluate model and upload data to be vizualized
        tester.upload_metrics()
        tester.upload_confusion_matrix()
        tester.uploat_heatmap()


if __name__ == '__main__':
    main()

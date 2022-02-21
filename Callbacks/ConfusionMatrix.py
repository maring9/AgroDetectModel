import itertools
from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from consts import BASE_LOG_DIR, CLASSES
from six.moves import range
from sklearn.metrics import confusion_matrix
from tensorflow import expand_dims
from tensorflow.image import decode_png
from tensorflow.keras.callbacks import Callback
from tensorflow.summary import create_file_writer, image


class ConfusionMatrix(Callback):
    """
        Custom callback class to create and visualize the confusion matrix of
        the classification outputs from the model
    """

    def __init__(self, data):
        """
            Constructor

        Args:
            data (numpy.ndarray):   Data used to create the confusion matrix
                                    Inputs, Labels
        """

        super(ConfusionMatrix, self).__init__()
        self.inputs = data[0]
        self.labels = data[1]

        self.tag = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        LOG_DIR = f"{BASE_LOG_DIR}/confusion_matrix/" + self.tag

        self.file_writer = create_file_writer(LOG_DIR)
        self.file_writer.set_as_default()

    def plot_to_image(self, figure):
        """
            Converts the matplotlib plot specified by 'figure' to a PNG image
            and returns it. The supplied figure is closed and inaccessible
            after this call.
        """

        buff = BytesIO()

        # Use plt.savefig to save the plot to a PNG in memory.
        plt.savefig(buff, format='png')

        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)

        buff.seek(0)

        # Use tf.image.decode_png to convert the PNG buffer
        # to a TF image. Make sure you use 4 channels.
        image_ = decode_png(buff.getvalue(), channels=4)

        # Use tf.expand_dims to add the batch dimension
        image_ = expand_dims(image_, 0)

        return image_

    def plot_confusion_matrix(self, cm):
        """
            Plots the confusion matrix

        Args:
            cm (numpy.ndarray): Confusion matrix

        Returns:
            matplotlib.figure.Figure: Plot of the confusion matrix
        """

        figure = plt.figure(figsize=(8, 8))

        plt.imshow(cm.T, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

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

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        predictions = self.model.predict(self.inputs)
        predictions = np.argmax(predictions, axis=1)

        confusion_matrix_ = confusion_matrix(self.labels, predictions)

        figure = self.plot_confusion_matrix(confusion_matrix_)

        cm_image = self.plot_to_image(figure)

        image("Confusion Matrix", cm_image, step=epoch)

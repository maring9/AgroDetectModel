import itertools
from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from six.moves import range
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from tensorflow import expand_dims
from tensorflow.image import decode_png
from tensorflow.keras.callbacks import (Callback, EarlyStopping,
                                        ModelCheckpoint, ReduceLROnPlateau,
                                        TensorBoard)
from tensorflow.summary import create_file_writer, image, scalar

from consts import BASE_LOG_DIR, CLASSES


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

    def on_train_end(self, epoch, logs=None):
        logs = logs or {}

        predictions = self.model.predict(self.inputs)
        predictions = np.argmax(predictions, axis=1)

        confusion_matrix_ = confusion_matrix(self.labels, predictions)

        figure = self.plot_confusion_matrix(confusion_matrix_)

        cm_image = self.plot_to_image(figure)

        image("Confusion Matrix", cm_image, step=epoch)


def get_earlystopping_callback(monitor='val_loss',
                               patience=0,
                               mode='auto',
                               baseline=None,
                               best_weights=False):
    """
        Helper function to create a callback that stops training when a
        monitored metric has stopped improving

    Args:
        monitor (str, optional):        Quantity to be monitored.
                                        Defaults to 'val_loss'.
        patience (int, optional):       Number of epochs with no improvement
                                        after which training stopps.
                                        Defaults to 0.
        mode (str, optional):           Whether the goal is to minimize or
                                        maximize the monitored metric.
                                        Defaults to 'auto'.
        baseline (_type_, optional):    Baseline value for monitored metric,
                                        training will stop if there is no
                                        improvement over the baseline.
                                        Defaults to None.
        best_weights (bool, optional):  Whether to restore model weights from
                                        the epoch with the best monitored
                                        metric. Defaults to False.

    Returns:
        keras.callbacks.EarlyStopping: EarlyStopping object
    """

    earlystopping_callback = EarlyStopping(monitor=monitor,
                                           patience=patience,
                                           mode=mode,
                                           baseline=baseline,
                                           restore_best_weights=best_weights)
    return earlystopping_callback


class Metrics(Callback):
    """
        Custom callback class to calculate F1-Score, Precision and Recall
        metrics. The metrics are logged and plotted using TesorBoard
    """

    def __init__(self, data):
        """
            Constructor

        Args:
            data (numpy.ndarray):   Data used to calculate the metrics on.
                                    Inputs, Labels

        """

        super(Metrics, self).__init__()
        self.inputs = data[0]
        self.labels = data[1]

        self.tag = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        LOG_DIR = f'{BASE_LOG_DIR}/metrics/' + self.tag
        self.file_writer = create_file_writer(LOG_DIR)
        self.file_writer.set_as_default()

    def on_train_end(self, epoch, logs=None):
        """
            Method to be called at the end of every epoch

        Args:
            epoch (int):            Represents current epoch number
            logs (dict, optional):  Dictionary containing logs.
                                    Defaults to None.
        """

        logs = logs or {}

        validation_predictions = np.argmax(self.model.predict(self.inputs), -1)

        validation_targets = self.labels

        if len(validation_targets.shape) == 2 and \
           validation_targets.shape[1] != 1:
            validation_targets = np.argmax(validation_targets, -1)

        validation_f1 = f1_score(validation_targets,
                                 validation_predictions,
                                 average='macro')

        validation_recall = recall_score(validation_targets,
                                         validation_predictions,
                                         average='macro')

        validation_precission = precision_score(validation_targets,
                                                validation_predictions,
                                                average='macro')

        scalar('f1 score', data=validation_f1, step=epoch)
        scalar('recall', data=validation_recall, step=epoch)
        scalar('precision', data=validation_precission, step=epoch)

        logs['val_f1'] = validation_f1
        logs['val_recall'] = validation_recall
        logs['val_precision'] = validation_precission

        print(" — val_f1: %f — val_precision: %f — val_recall: %f" %
              (validation_f1, validation_precission, validation_recall))

        return


def get_modelcheckpoint_callback(filepath,
                                 monitor='val_loss',
                                 save_best_only=False,
                                 save_weights_only=False,
                                 mode='auto'):
    """
        Helper function to create a callback that saves the model weights.

    Args:
        filepath (string):                  Path to save the model
        monitor (str, optional):            Metric to monitor during training.
                                            Defaults to 'val_loss'.
        save_best_only (bool, optional):    Whether to only save the best
                                            model. Defaults to False.
        save_weights_only (bool, optional): Whether to only save the model
                                            weights. Defaults to False.
        mode (str, optional):               Whether to save the models based on
                                            maximization or minimization of the
                                            monitored metric.
                                            Defaults to 'auto'.

    Returns:
        keras.callbacks.ModelCheckpoint: ModelCheckpoint object
    """

    chekcpoint_callback = ModelCheckpoint(filepath=filepath,
                                          monitor=monitor,
                                          save_best_only=save_best_only,
                                          save_weights_only=save_weights_only,
                                          mode=mode)

    return chekcpoint_callback


def get_reducelr_callback(monitor='val_loss',
                          factor=0.1,
                          patience=10,
                          mode='auto',
                          min_delta=0.0001,
                          cooldown=0,
                          min_lr=0):
    """
        Helper function to create a callback to reduce te learning rate when a
        metric has stopped improving

    Args:
        monitor (str, optional):        Quantity to be monitored.
                                        Defaults to 'val_loss'.
        factor (float, optional):       Factor by which to reduce learning
                                        rate. Defaults to 0.1.
        patience (int, optional):       Number of epochs with no improvement
                                        after which the learning rate will be
                                        reduced. Defaults to 10.
        mode (str, optional):           Whether to save the models based on
                                        maximization or minimization of the
                                        monitored metric.
                                        Defaults to 'auto'.
        min_delta (float, optional):    Threshold for measuring the new
                                        optimum. Defaults to 0.0001.
        cooldown (int, optional):       Number of epochs to wait before
                                        resuming normal operation.
                                        Defaults to 0.
        min_lr (int, optional):         Lower bound on the learning rate.
                                        Defaults to 0.

    Returns:
        keras.callbacks.ReduceLROnPlateau:  ReduceLROnPlateau object
    """

    reduce_lr = ReduceLROnPlateau(monitor=monitor,
                                  factor=factor,
                                  patience=patience,
                                  mode=mode,
                                  min_delta=min_delta,
                                  cooldown=cooldown,
                                  min_lr=min_lr)

    return reduce_lr


def get_tensorboard_callback(log_dir=BASE_LOG_DIR,
                             histogram_freq=0,
                             write_graph=True,
                             write_images=False,
                             update_freq='epoch'):
    """
        Helper function to create a callback that enables visualizations

    Args:
        log_dir (string, optional):     Path of the directory where to save log
                                        files. Defaults to BASE_LOG_DIR.
        histogram_freq (int, optional): Frequency (in epochs) at which to
                                        compute activation and weight
                                        histogram. Defaults to 0.
        write_graph (bool, optional):   Whether to visualize the graph.
                                        Defaults to True.
        write_images (bool, optional):  Whether to write model weighs to
                                        visualize. Defaults to False.
        update_freq (str, optional):    Frequency at which to write metrics
                                        and losses. Defaults to 'epoch'.

    Returns:
        keras.callbacks.TensorBoard: TensorBoard object
    """

    tensorboard_callback = TensorBoard(log_dir=log_dir,
                                       histogram_freq=histogram_freq,
                                       write_graph=write_graph,
                                       write_images=write_images,
                                       update_freq=update_freq)

    return tensorboard_callback

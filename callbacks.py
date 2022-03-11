import gc
import itertools
from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from six.moves import range
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from tensorflow import expand_dims
from tensorflow.image import decode_png
from tensorflow.keras.callbacks import (Callback, EarlyStopping,
                                        ModelCheckpoint, ReduceLROnPlateau,
                                        TensorBoard)
from tensorflow.summary import create_file_writer, scalar

from consts import BASE_LOG_DIR, CLASSES


def get_earlystopping_callback(monitor='val_accuracy',
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

    def on_epoch_end(self, epoch, logs=None):
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
                                 monitor='val_accuracy',
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

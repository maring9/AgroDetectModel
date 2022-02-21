from datetime import datetime

import numpy as np
from consts import BASE_LOG_DIR
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.callbacks import Callback
from tensorflow.summary import create_file_writer, scalar


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

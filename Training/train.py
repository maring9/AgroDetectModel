import itertools
from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

BASE_LOG_DIR = '.logs'
import horovod.keras as hvd
from keras import backend as K
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, MaxPool2D)
from tensorflow.keras.models import Sequential

CLASSES = ['Apple Black rot', 'Apple Cedar rust', 'Apple scab',
           'Apple healthy', 'Blueberry healthy', 'Cherry healthy',
           'Cherry Powdery mildew', 'Corn Cercospora Gray laef spot',
           'Corn Common rust', 'Corn healthy', 'Corn Northern Leaf Bligh',
           'Grape Black rot', 'Grape Esca Black Measles', 'Grape healthy',
           'Grape Leaf blight Isariopsis Leaf Spot',
           'Orange Haunglongbing Citrus', 'Peach Bacterial spot',
           'Peach healthy', 'Pepper bell Bacterial spot',
           'Pepper bell healthy', 'Potato Early blight', 'Potato healthy',
           'Potato Late blight', 'Raspberry healthy', 'Soybean healthy',
           'Squash Powdery mildew', 'Strawberry healthy',
           'Strawberry Leaf scorch', 'Totamo Bacterial spot',
           'Tomato Early blight', 'Tomato healthy', 'Tomato Late blight',
           'Tomato Leaf Mold', 'Tomat mosaic virus', 'Tomato Target Spot',
           'Tomato Two spotted spider mite', 'Tomato Yellow Leaf Curl Virus']
import tensorflow as tf
from six.moves import range
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from tensorflow import expand_dims
from tensorflow.image import decode_png
from tensorflow.keras.callbacks import (Callback, EarlyStopping,
                                        ModelCheckpoint, TensorBoard)
from tensorflow.summary import create_file_writer, image, scalar

NUM_CLASSES = 38

IMAGE_DIMS = (256, 256, 3)
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


def get_alexnet_architecture():
    """
       Helper function to create AlexNet model

    Returns:
        keras.Model: Model instance (AlexNet Architecture)
    """

    alexnet = Sequential([
        Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4),
               activation='relu', input_shape=IMAGE_DIMS),
        BatchNormalization(),
        MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1),
               activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
               activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
               activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    return alexnet

def train():
    # hvd.init()
    config = tf.compat.v1.ConfigProto()
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    config.gpu_options.allow_growth = True
    # config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.compat.v1.Session(config=config))
    batch_size = 1
    num_classes = 38
    train_dir = '/home/marin/Desktop/Dataset/PlantVillage/train'
    val_dir = '/home/marin/Desktop/Dataset/PlantVillage/validation'
    test_dir = '/home/marin/Desktop/Dataset/PlantVillage/testing'
    train_dataset = keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                 labels="inferred",
                                                                 label_mode='int',
                                                                 color_mode='rgb',
                                                                 batch_size=batch_size,
                                                                 image_size=(256, 256),
                                                                 shuffle=True,
                                                                 seed=42)
    val_dataset = keras.preprocessing.image_dataset_from_directory(directory=val_dir,
                                                                 labels="inferred",
                                                                 label_mode='int',
                                                                 color_mode='rgb',
                                                                 batch_size=batch_size,
                                                                 image_size=(256, 256),
                                                                 shuffle=True,
                                                                 seed=42)
    #AUTOTUNE = tf.data.AUTOTUNE
    # train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    # val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)                                                                 
    normalization = keras.layers.Rescaling(1./255)
    normalized_train_dataset = train_dataset.map(lambda x, y: (normalization(x), y))
    normalized_val_dataset = val_dataset.map(lambda x, y: (normalization(x), y))
    val_x = []
    val_y = []
    for image_batch, labels_batch in normalized_val_dataset:
        val_x.append(image_batch.numpy())
        # val_y.append(keras.utils.to_categorical(labels_batch.numpy(), num_classes=38))
        val_y.append(labels_batch.numpy())

    val_x = np.asarray(val_x, dtype=object)
    val_y = np.asarray(val_y)

    print('shape: ', val_x.shape)
    # print('val x: ', val_x)
    net = get_alexnet_architecture()

    tensorboard = get_tensorboard_callback()
    earlystopping = get_earlystopping_callback(patience=10)
    checkpoint = get_modelcheckpoint_callback("Checkpoints/save_at_{epoch}.h5")
    cm = ConfusionMatrix((val_x, val_y))
    metrics = Metrics((val_x, val_y))

    net.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

    net.fit(normalized_train_dataset, validation_data=normalized_val_dataset, epochs=100, callbacks=[tensorboard, earlystopping,checkpoint, cm, metrics])


if __name__=='__main__':
    train()

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

from consts import BASE_LOG_DIR, CLASSES, MODEL_CHECKPOINTS, TEST_DIR
from models import (get_alexnet_architecture, get_efficientnetb7_architecture,
                    get_inceptionresnetv2_architecture,
                    get_inceptionv3_architecture, get_mobilenetv2_architecture,
                    get_resnet152v2_architecture, get_vgg16_architecture)


class ModelTester():
    def __init__(self, model, data):
        """
            Constructor

        Args:
            data (numpy.ndarray):   Data used to create the confusion matrix
                                    Inputs, Labels
        """

        self.model = model

        self.data = data
        self.data = self.data.unbatch()

        self.inputs = []
        self.labels = []

        for images, labels in self.data.take(-1):
            self.inputs.append(images.numpy())
            self.labels.append(labels.numpy())

        self.inputs = np.array(self.inputs)
        self.labels = np.array(self.labels)

        del self.data

        self.tag = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        self.predictions = None

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
        # plt.close(figure)

        buff.seek(0)

        # Use tf.image.decode_png to convert the PNG buffer
        # to a TF image. Make sure you use 4 channels.
        image_ = tf.image.decode_png(buff.getvalue(), channels=4)

        # Use tf.expand_dims to add the batch dimension
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

        figure = plt.figure(figsize=(12, 12))

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

    def upload_confusion_matrix(self):
        if self.predictions is None:
            self.predictions = self.model.predict(self.inputs)
            self.predictions = np.argmax(self.predictions, axis=1)

        LOG_DIR = f"{BASE_LOG_DIR}/testing/cm/{self.model.name}" + self.tag
        self.file_writer = tf.summary.create_file_writer(LOG_DIR)
        self.file_writer.set_as_default()

        confusion_matrix_ = confusion_matrix(self.labels, self.predictions)

        figure = self.plot_confusion_matrix(confusion_matrix_)

        cm_image = self.plot_to_image(figure)

        tf.summary.image("Confusion Matrix", cm_image, step=-1)

        return

    def upload_metrics(self):
        if self.predictions is None:
            self.predictions = self.model.predict(self.inputs)
            self.predictions = np.argmax(self.predictions, axis=1)

        LOG_DIR = f'{BASE_LOG_DIR}/testing/metrics/{self.model.name}' \
                  + self.tag
        self.file_writer = tf.summary.create_file_writer(LOG_DIR)
        self.file_writer.set_as_default()

        test_f1 = f1_score(self.labels,
                           self.predictions,
                           average='macro')

        test_recall = recall_score(self.labels,
                                   self.predictions,
                                   average='macro')

        test_precission = precision_score(self.labels,
                                          self.predictions,
                                          average='macro')

        test_accuracy = accuracy_score(self.labels,
                                       self.predictions)

        tf.summary.scalar('test_f1 score', data=test_f1, step=-1)
        tf.summary.scalar('test_recall', data=test_recall, step=-1)
        tf.summary.scalar('test_precision', data=test_precission, step=-1)
        tf.summary.scalar('test_accuracy', data=test_accuracy, step=-1)

        return

    def uploat_heatmap(self):
        if self.predictions is None:
            self.predictions = self.model.predict(self.inputs)
            self.predictions = np.argmax(self.predictions, axis=1)

        LOG_DIR = f'{BASE_LOG_DIR}/testing/report/{self.model.name}' \
                  + self.tag
        self.file_writer = tf.summary.create_file_writer(LOG_DIR)
        self.file_writer.set_as_default()

        report = classification_report(self.labels, self.predictions,
                                       target_names=CLASSES,
                                       output_dict=True)

        plt.figure(figsize=(12, 12))

        figure = sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)

        plot = self.plot_to_image(figure)

        tf.summary.image("Classification Report", plot, step=-1)

        return


def main():
    test_ds = tf.keras.utils.image_dataset_from_directory(TEST_DIR,
                                                          image_size=(150, 150),
                                                          batch_size=32,
                                                          seed=42)

    normalization_layer = tf.keras.layers.Rescaling(1./255)

    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    models = {
        'InceptionResnetV2': get_inceptionresnetv2_architecture(weights=None),
        'AlexNet': get_alexnet_architecture(),
        'MobileNetv2': get_mobilenetv2_architecture(weights=None),
        'VGG16': get_vgg16_architecture(weights=None),
        'ResNet152V2': get_resnet152v2_architecture(weights=None),
        'InceptionV3': get_inceptionv3_architecture(weights=None),
        'EfficientNetB7': get_efficientnetb7_architecture(weights=None)

    }

    for checkpoints_dir in os.listdir(MODEL_CHECKPOINTS):
        path = os.path.join(MODEL_CHECKPOINTS, checkpoints_dir)
        model_weights = tf.train.latest_checkpoint(path)
        architecture = path.split('/')[-1]

        model = models[architecture]

        model.load_weights(model_weights)

        tester = ModelTester(model, test_ds)

        tester.upload_metrics()
        tester.upload_confusion_matrix()
        tester.uploat_heatmap()


if __name__ == '__main__':
    main()

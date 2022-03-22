

import os

import numpy as np
import tensorflow as tf

from consts import IMAGE_SIZE
from models import (get_alexnet_architecture, get_efficientnetb7_architecture,
                    get_inceptionresnetv2_architecture,
                    get_inceptionv3_architecture, get_mobilenetv2_architecture,
                    get_resnet152v2_architecture, get_vgg16_architecture)

# Dictionary to create model architectures
MODELS = {'AlexNet': get_alexnet_architecture(),
          'EfficientNetB7': get_efficientnetb7_architecture(),
          'InceptionResnetV2': get_inceptionresnetv2_architecture(),
          'InceptionV3': get_inceptionv3_architecture(),
          'ResNet152V2': get_resnet152v2_architecture(),
          'VGG16': get_vgg16_architecture(),
          'MobileNetv2': get_mobilenetv2_architecture()
          }


def get_latest_model(model_dir, model_name):
    """
        Function to load the latest model from checkpoints directory

    Args:
        model_dir (str):    Directory containing model checkpoints.
        model_name (str):   One of the trained model architectures:
                            AlexNet, EfficientNetB7, InceptionResnetV2,
                            InceptionV3, ResNet152V2, VGG16, MobileNetv2

    Raises:
        ValueError: Invalid model directory.
        ValueError: Invalid model name.

    Returns:
        tf.keras.Model: Model architecture with trained weights.
    """

    # Sanity Checks
    # TODO check if checkpoints are present
    if not os.path.isdir(model_dir):
        raise ValueError('Invalid Model Directory')

    if model_name not in list(MODELS.keys()):
        raise ValueError('Invalid Model Name')

    # Get the model weights from the latest checkpoint
    latest_model_weights = tf.train.latest_checkpoint(model_dir)

    # Create model architecture
    model = MODELS[model_name]

    # Load trained model weights
    model.load_weights(latest_model_weights)

    return model


def preprocess_input(image_path):
    """
        Function to preprocess an image for inference

    Args:
        image_path (str): Path of the image to be preprocessed.

    Returns:
        numpy.ndarray:  Preprocessed image for model prediction.
                        Image is converted to float and normalized to
                        [0, 1] and added extra batch dimension for inference
    """

    # Load image from path
    image = tf.keras.preprocessing.image.load_img(image_path,
                                                  target_size=IMAGE_SIZE)

    # Transform image to an array
    image_array = tf.keras.preprocessing.image.img_to_array(image)

    # Convert to floating point and normalize to [0, 1]
    image_array = image_array.astype('float32') / 255

    # Add batch dimension needed for inference (tensor shape: (1, 150, 150, 3))
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

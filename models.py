from tensorflow.keras import Model
from tensorflow.keras.applications import (VGG16, EfficientNetB7,
                                           InceptionResNetV2, InceptionV3,
                                           ResNet152V2)
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, MaxPool2D)
from tensorflow.keras.models import Sequential

from consts import IMAGE_DIMS, NUM_CLASSES


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


def get_efficientnetb7_architecture(include_top=True, weights=None,
                                    input_shape=IMAGE_DIMS):
    """
        Helper functinon to create EfficientNetV2B0 model

    Args:
        include_top (bool, optional):   Whether to include the fully-connected
                                        layer at the top of the network.
                                        Defaults to True.
        weights (str, optional):        Path to the weights file to be loaded.
                                        Defaults to 'imagenet'.
        input_shape (tuple, optional):  Should have exactly 3 inputs channels.
                                        Input shape for the model.
                                        Defaults to None.

    Returns:
        keras.Model:                    Model instance (EfficientNet-b0
                                                        Architecture)
    """

    efficient_net = EfficientNetB7(include_top=include_top,
                                   weights=weights,
                                   input_shape=input_shape)

    outputlayer = Dense(NUM_CLASSES, activation='softmax')(efficient_net.layers[-2].output)

    model = Model(efficient_net.input, outputlayer)

    return model


def get_inceptionresnetv2_architecture(include_top=True,
                                       weights=None,
                                       input_shape=IMAGE_DIMS):
    """
        Helper functinon to create InceptionResNetV2 model

    Args:
        include_top (bool, optional):   Whether to include the fully-connected
                                        layer at the top of the network.
                                        Defaults to True.
        weights (str, optional):        Path to the weights file to be loaded.
                                        Defaults to 'imagenet'.
        input_shape (tuple), optional): Should have exactly 3 inputs channels,
                                        and width and height should be no
                                        smaller than 75.
                                        Defaults to (299, 299, 3).

    Returns:
        keras.Model:                    Model instance (InceptionResNetV2
                                                        Architecture)
    """

    inceptionresnetv2 = InceptionResNetV2(include_top=include_top,
                                          weights=weights,
                                          input_shape=input_shape)

    outputlayer = Dense(NUM_CLASSES, activation='softmax')(inceptionresnetv2.layers[-2].output)

    model = Model(inceptionresnetv2.input, outputlayer)

    return model


def get_inceptionv3_architecture(include_top=True,
                                 weights=None,
                                 input_shape=IMAGE_DIMS):
    """
        Helper functinon to create InceptionV3 model

    Args:
        include_top (bool, optional):   Whether to include the fully-connected
                                        layer at the top. Defaults to True.
        weights (str, None):            Path to the weights file to be loaded.
                                        Defaults to 'imagenet'.
        input_shape (tuple, optional):  Exactly 3 inputs channels, and width
                                        and height should be no smaller
                                        than 75.
                                        Defaults to (299, 299, 3).

    Returns:
        keras.Mode:                     Model instance (InceptionV3
                                                        Architecture)
    """

    inceptionv3 = InceptionV3(include_top=include_top,
                              weights=weights,
                              input_shape=input_shape)

    outputlayer = Dense(NUM_CLASSES, activation='softmax')(inceptionv3.layers[-2].output)

    model = Model(inceptionv3.input, outputlayer)

    return model


def get_resnet152v2_architecture(include_top=True,
                                 weights=None,
                                 input_shape=IMAGE_DIMS):
    """
        Helper functinon to create ResNet152V2 model

    Args:
        include_top (bool, optional):   Whether to include the fully-connected
                                        layer at the top of the network.
                                        Defaults to True.
        weights (str, optional):        Path to the weights file to be loaded.
                                        Defaults to 'imagenet'.
        input_shape (tuple, optional):  Should have exactly 3 inputs channels,
                                        and width and height should be no
                                        smaller than 32.
                                        Defaults to (224, 224, 3).

    Returns:
        keras.Model:                    Model instance (ResNet152V2
                                                        Architecture)
    """

    resnet152v2 = ResNet152V2(include_top=include_top,
                              weights=weights,
                              input_shape=input_shape)

    outputlayer = Dense(NUM_CLASSES, activation='softmax')(resnet152v2.layers[-2].output)

    model = Model(resnet152v2.input, outputlayer)

    return model


def get_vgg16_architecture(include_top=True,
                           weights=None,
                           input_shape=IMAGE_DIMS):
    """
        Helper functinon to create VGG16 model

    Args:
        include_top (bool, optional):   Whether to include the 3 fully-
                                        connected layers at the top of the
                                        network. Defaults to True.
        weights (str, None):            Path to the weights file to be loaded.
                                        Defaults to 'imagenet'.
        input_shape (tuple, optional):  3 input channels, and width and height
                                        should be no smaller than 32.
                                        Defaults to (224, 224, 3).

    Returns:
        keras.Model:                    Model instance (VGG Architecture)
    """

    vgg16 = VGG16(include_top=include_top,
                  weights=weights,
                  input_shape=input_shape)

    outputlayer = Dense(NUM_CLASSES, activation='softmax')(vgg16.layers[-2].output)

    model = Model(vgg16.input, outputlayer)

    return model

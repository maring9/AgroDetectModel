from consts import IMAGE_DIMS, NUM_CLASSES
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.layers import Dense


def get_resnet152v2_architecture(include_top=True,
                                 weights='imagenet',
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

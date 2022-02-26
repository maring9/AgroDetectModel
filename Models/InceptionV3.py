from consts import IMAGE_DIMS, NUM_CLASSES
from tensorflow.keras import Model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense


def get_inceptionv3_architecture(include_top=True,
                                 weights='imagenet',
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

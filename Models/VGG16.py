from tensorflow.keras.applications import VGG16


def get_vgg16_architecture(include_top=True, weights='imagenet',
                           input_shape=(224, 224, 3)):
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

    return vgg16

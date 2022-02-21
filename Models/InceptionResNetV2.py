from tensorflow.keras.applications import InceptionResNetV2


def get_inceptionresnetv2_architecture(include_top=True,
                                       weights='imagenet',
                                       input_shape=(299, 299, 3)):
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
    return inceptionresnetv2

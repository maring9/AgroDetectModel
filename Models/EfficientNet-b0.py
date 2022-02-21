from tensorflow.keras.applications import EfficientNetV2B0


def get_efficientnetb0_architecture(include_top=True, weights='imagenet',
                                    input_shape=None):
    """

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

    efficient_net = EfficientNetV2B0(include_top=include_top,
                                     weights=weights,
                                     input_shape=input_shape)

    return efficient_net

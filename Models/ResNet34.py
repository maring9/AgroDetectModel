from classification_models.keras import Classifiers


def get_resnet34_architecture():
    """
        Helper function to create ResNet34 model

    Returns:
        keras.Model:    Model instance (ResNet34 Architecture)
    """

    ResNet34, preprocess_input = Classifiers.get('resnet34')

    return ResNet34

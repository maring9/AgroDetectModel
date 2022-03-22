
import base64
import json
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image

from consts import CLASSES, IMAGE_SIZE

# Path to load model from
model_file = '/opt/ml/model'

# Load model, will only be loaded once per cold start
model = tf.keras.models.load_model(model_file)


def preprocess_payload(event):
    """


    Args:
        event (request):    Request containing the image as payload to run
                            inference on

    Returns:
        numpy.ndarray:      Preprocessed image ready for inference.
                            The image is resized to (150, 150), normalized
                            and added extra batch dimension needed for
                            inference
    """

    # Get payload from the request
    image_bytes = event['body'].encode('utf-8')

    # Open the image gotten as payload
    image = Image.open(BytesIO(base64.b64decode(image_bytes)))
    image = image.convert(mode='RGB')

    # Resize image to the resolution expected by the model
    image = image.resize(IMAGE_SIZE)

    # Convert image to numpy array and normalize to [0, 1]
    image = np.array(image).astype('float32') / 255

    # Add batch dimension for inference
    image = np.expand_dims(image, axis=0)

    return image


def lambda_handler(event, context):
    image = preprocess_payload(event)
    probabilities = model.predict(image)
    prediction = np.argmax(probabilities)

    label = CLASSES[prediction]
    return {
        'statusCode': 200,
        'body': json.dumps(
            {
                "predicted_label": label,
            }
        )
    }

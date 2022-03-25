
import base64
import json
from io import BytesIO

import boto3
import numpy as np
import tensorflow as tf
from PIL import Image

from aws_s3 import upload_image
from consts import CLASSES, IMAGE_SIZE

# Path to load model from
MODEL_FILE = '/opt/ml/model'

# Load model, will only be loaded once per cold start
MODEL = tf.keras.models.load_model(MODEL_FILE)

# S3 client used to log input image
S3_CLIENT = boto3.client('s3',
                         aws_access_key_id='AKIAZDAXB3AOIAVMQ7VK',
                         aws_secret_access_key='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')

# Bucket name where the image is saved
BUCKET_NAME = 'plant-disease-input-images'


def preprocess_payload(event):
    """
        Function to preprocess the image payload for inference.
        The image is saved to an S3 bucket for logging

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

    image_bytes = BytesIO(base64.b64decode(image_bytes))

    inference_image = Image.open(image_bytes)
    inference_image = inference_image.convert(mode='RGB')

    # Resize image to the resolution expected by the model
    inference_image = inference_image.resize(IMAGE_SIZE)

    # Convert image to numpy array and normalize to [0, 1]
    inference_image = np.array(inference_image).astype('float32') / 255

    # Add batch dimension for inference
    inference_image = np.expand_dims(inference_image, axis=0)

    return inference_image, image_bytes


def lambda_handler(event, context):
    inference_image, image_bytes = preprocess_payload(event)
    probabilities = MODEL.predict(inference_image)
    prediction = np.argmax(probabilities)

    label = CLASSES[prediction]

    _ = upload_image(image_bytes, S3_CLIENT, BUCKET_NAME, label)

    return {
        'statusCode': 200,
        'body': json.dumps(
            {
                "predicted_label": label,
            }
        )
    }

# Constants

BASE_LOG_DIR = 'logs'

TRAIN_DIR = '/home/marin/Desktop/Dataset/PlantVillageResized/train'
VAL_DIR = '/home/marin/Desktop/Dataset/PlantVillageResized/validation'
TEST_DIR = '/home/marin/Desktop/Dataset/PlantVillageResized/testing'

BATCH_SIZE = 32

EPOCHS = 50

MODEL_CHECKPOINTS = '/home/marin/Desktop/BachelorThesis/Checkpoints'

CLASSES = sorted(['Apple Black rot', 'Apple Cedar rust', 'Apple healthy',
                  'Apple scab', 'Blueberry healthy', 'Cherry healthy',
                  'Cherry Powdery mildew', 'Corn Cercospora Gray laef spot',
                  'Corn Common rust', 'Corn healthy',
                  'Corn Northern Leaf Bligh', 'Grape Black rot',
                  'Grape Esca Black Measles', 'Grape healthy',
                  'Grape Leaf blight Isariopsis Leaf Spot',
                  'Orange Haunglongbing Citrus', 'Peach Bacterial spot',
                  'Peach healthy', 'Pepper bell Bacterial spot',
                  'Pepper bell healthy', 'Potato Early blight',
                  'Potato healthy', 'Potato Late blight', 'Raspberry healthy',
                  'Soybean healthy', 'Squash Powdery mildew',
                  'Strawberry healthy', 'Strawberry Leaf scorch',
                  'Tomato Bacterial spot', 'Tomato Early blight',
                  'Tomato healthy', 'Tomato Late blight', 'Tomato Leaf Mold',
                  'Tomato mosaic virus', 'Tomato Septoria leaf spot',
                  'Tomato Target Spot', 'Tomato Two spotted spider mite',
                  'Tomato Yellow Leaf Curl Virus'])

NUM_CLASSES = 38

IMAGE_DIMS = (150, 150, 3)
IMAGE_SIZE = (150, 150)

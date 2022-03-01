BASE_LOG_DIR = 'logs'

TRAIN_DIR = '/home/marin/Desktop/Dataset/PlantVillage/train'
VAL_DIR = '/home/marin/Desktop/Dataset/PlantVillage/validation'
TEST_DIR = '/home/marin/Desktop/Dataset/PlantVillage/testing'

TEST_IMAGES = '/home/marin/Desktop/BachelorThesis/test data/inputs.npy'
TEST_LABELS = '/home/marin/Desktop/BachelorThesis/test data/labels.npy'

CLASSES = ['Apple Black rot', 'Apple Cedar rust', 'Apple scab',
           'Apple healthy', 'Blueberry healthy', 'Cherry healthy',
           'Cherry Powdery mildew', 'Corn Cercospora Gray laef spot',
           'Corn Common rust', 'Corn healthy', 'Corn Northern Leaf Bligh',
           'Grape Black rot', 'Grape Esca Black Measles', 'Grape healthy',
           'Grape Leaf blight Isariopsis Leaf Spot',
           'Orange Haunglongbing Citrus', 'Peach Bacterial spot',
           'Peach healthy', 'Pepper bell Bacterial spot',
           'Pepper bell healthy', 'Potato Early blight', 'Potato healthy',
           'Potato Late blight', 'Raspberry healthy', 'Soybean healthy',
           'Squash Powdery mildew', 'Strawberry healthy',
           'Strawberry Leaf scorch', 'Totamo Bacterial spot',
           'Tomato Early blight', 'Tomato healthy', 'Tomato Late blight',
           'Tomato Leaf Mold', 'Tomat mosaic virus', 'Tomato Target Spot',
           'Tomato Two spotted spider mite', 'Tomato Yellow Leaf Curl Virus']

NUM_CLASSES = 38

IMAGE_DIMS = (256, 256, 3)

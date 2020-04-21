import sys, os
import numpy as np

from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.preprocessing import image

# indicates where model is saved to (should correspond to that in train.py)
FILENAME = 'butterfly_classification.h5'
# default testing data's path (to get data, run ../data/label_data.py)
TEST_DATA = "../data/test"
# default image size to use
IMG_SIZE = 224

BATCH_SIZE = 32

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def predict_butterfly(img_path, weights=FILENAME):
    img = preprocess_image(img_path)
    model = load_model(weights)
    result = model.predict(img)
    print(result)

def evaluate_butterfly(images_path=TEST_DATA, weights=FILENAME):
    test_datagen = image.ImageDataGenerator(
        preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_directory(images_path, 
        target_size=(IMG_SIZE, IMG_SIZE), 
        batch_size=BATCH_SIZE)
    model = load_model(weights)
    model.compile(optimizer='Adam', 
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy', 'acc'])
    model.evaluate(test_generator)

if __name__ == "__main__":
    if len(sys.argv) != 2 and os.path.isdir(TEST_DATA):
        # evaluate model using default test files (generated by running ../data/label_data.py)
        evaluate_butterfly()
    elif len(sys.argv) == 2:
        path = sys.argv[1]      
        if os.path.isdir(path):
            # evaluate model using given directory
            evaluate_butterfly(path)
        else:
            # predict label of img
            predict_butterfly(path)
    elif len(sys.argv) == 3:
        path = sys.argv[1]
        weights = sys.argv[2]
        if os.path.isdir(path):
            # evaluate model using given directory
            evaluate_butterfly(path, weights)
        else:
            # predict label of img
            predict_butterfly(path, weights)
    else:
        print('Please input image file or run ../data/label_data.py')

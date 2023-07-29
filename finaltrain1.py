import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import pickle

TRAIN_DIR = 'D:/Users/Public/Documents/100%pythonCode/Sourcecode/code/train/train'
TEST_DIR = 'D:/Users/Public/Documents/100%pythonCode/Sourcecode/code/test/test'
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'healthyvsunhealthy-{}-{}.model'.format(LR, '2conv-basic')

def label_img(img):
    word_label = img.split('.')[0]
    if word_label == 'h':
        return [1, 0, 0, 0]
    elif word_label == 'b':
        return [0, 1, 0, 0]
    elif word_label == 'v':
        return [0, 0, 1, 0]
    elif word_label == 'l':
        return [0, 0, 0, 1]

'''def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data'''

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    shuffle(testing_data)
    return testing_data

# Uncomment the following line if you need to create the train_data.npy file
# train_data = create_train_data()

# If you have already created the dataset:
train_data = np.load('train_data.npy', allow_pickle=True)

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 128, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 4, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model loaded!')

# Load or create the testing data using pickle
if os.path.exists('test_data.pkl'):
    with open('test_data.pkl', 'rb') as f:
        testing_data = pickle.load(f)
else:
    testing_data = process_test_data()
    with open('test_data.pkl', 'wb') as f:
        pickle.dump(testing_data, f)

test_images = np.array([i[0] for i in testing_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
test_ids = [i[1] for i in testing_data]

predictions = model.predict(test_images)

for i in range(len(test_images)):
    img_id = test_ids[i]
    prediction = predictions[i]
    if np.argmax(prediction) == 0:
        print('{img_id}: Healthy')
    elif np.argmax(prediction) == 1:
        print('{img_id}: Bacterial Spot')
    elif np.argmax(prediction) == 2:
        print('{img_id}: Late Blight')
    elif np.argmax(prediction) == 3:
        print('{img_id}: Yellow Vein Curl Virus')

print('Prediction completed!')


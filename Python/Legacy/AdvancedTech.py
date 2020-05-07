# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import joblib
from keras.preprocessing.image import ImageDataGenerator
import os 
import pyautogui
import time
import numpy as np
from keras.preprocessing import image


os.getcwd()
os.chdir('C:/Users/Mike/Documents/University/Year 3/AT2/AT2/Python')
dirpath = 'C:/Users/Mike/Documents/University/Year 3/AT2/AT2/Python'

def Save(_classifier):
    filename = 'completedClassifier.sav'
    joblib.dump(_classifier, filename)
    return;

def Screenshot(i):
    screenshot = pyautogui.screenshot()
    screenshot.save("dataset/test_set/firework/" + str(i) + ".png")
    print(str(i) + " taken")
    time.sleep(0.25)


#def TestImage(_classifier, _training_set):
#    classifier = _classifier    
#    training_set = _training_set
#    path = ('./dataset/test.png')
#    test_image = image.load_img(path, target_size = (64, 64))
#    test_image = image.img_to_array(test_image)
#    test_image = np.expand_dims(test_image, axis = 0)
#    result = classifier.predict(test_image)
#    training_set.class_indices
#    
#    if result[0][0] >=0.5:
#        prediction = 'dog'
#    else:
#        prediction = 'cat'
#        print(prediction)
#
#    Save(classifier);
#
#
#def RunClassifier():
##if (path.isfile())
#    classifier = Sequential()
#
#
##Convolution
#    classifier.add(Convolution2D(32, 3, 3, input_shape = (64,64,3), activation = 'relu'))
#
##Pooling
#    classifier.add(MaxPooling2D(pool_size = (2, 2)))
#
##Flattening
#    classifier.add(Flatten())
#
##connection
#    classifier.add(Dense(output_dim = 128, activation = 'relu'))
#    classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
#
##compile
#    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#
##Fitting CNN To image
#
#    train_datagen = ImageDataGenerator(
#        rescale=1./255,
#        shear_range=0.2, 
#        zoom_range=0.2,
#        horizontal_flip=True)
#
#    test_datagen = ImageDataGenerator(rescale=1./255)
#
#    training_set = train_datagen.flow_from_directory(
#        dirpath + '/dataset/training_set',
#        target_size=(64, 64),
#        batch_size=32,
#        class_mode='binary')
#
#
#    test_set= test_datagen.flow_from_directory(
#       dirpath + '/dataset/test_set',
#        target_size=(64,64),
#        batch_size=32,
#        class_mode='binary')
#
##from IPython.display import display
##from PIL import Image
#    classifier.fit_generator(
#        training_set,
#        steps_per_epoch=1000,
#        epochs=3,
#        validation_data=test_set,
#        validation_steps=100)
#
#    TestImage(classifier, training_set)
#    
#RunClassifier() 

for i in range(8000):
    Screenshot(i)
    




    




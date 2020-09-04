#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import os
import pickle
from PIL import Image
import time

def read_img(img_name):
    img = Image.open(img_name)
    np_img = np.array(img) # (224, 224, 3)
    np_img = np.asarray([np_img], dtype=np.int32) # (1, 224, 224, 3)
    return np_img


# In[ ]:


filePath = "D:\\Juypter\\ColorClustered\\" 
fileNames = os.listdir(filePath)
filedirs = []

for filename in fileNames:
    newfile_path=filename
    filedirs.append(newfile_path)
print(filedirs)


# In[ ]:


def use_color_classify(filePath):
    model = keras.models.load_model('D:\\Juypter\\model\\model.h5')
    new_image = read_img(filePath)
    img = plt.imshow(np.vstack(new_image))
    test_data = np.vstack(new_image)
    predictions = model.predict(np.array( [test_data] ))
    list_index = []
    for i in range(len(filedirs)):
        list_index.append(i)
    
    x = predictions
    for i in range(len(filedirs)):
        for j in range(len(filedirs)):
            if x[0][list_index[i]] > x[0][list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp
    #Show the sorted labels in order from highest probability to lowest
    print(list_index)
    classification = ['oil black','orange','gary','light white','light blue',
                      'gray blue','oil black','oil vblack','oil brown','gary brown',
                      'gary brown','reddish brown','bright beige','oil brown', 'oil black',
                      'oil gary', 'reddish brown', 'oil blue black','white and black','gary blue',
                     'pink']
    i=0
    for i in range(5):
        print(classification[list_index[i]], ':', round(predictions[0][list_index[i]] * 100, 2), '%')
    name = filePath.split("\\")[-2].split("-")[1]
    this_str = "I saw a "+name+" with the color "+classification[list_index[0]]
    return this_str


# In[ ]:





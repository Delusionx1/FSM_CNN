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


# In[ ]:


def read_img(img_name):
    img = Image.open(img_name)
    np_img = np.array(img) # (224, 224, 3)
    np_img = np.asarray([np_img], dtype=np.int32) # (1, 224, 224, 3)
    return np_img


# In[ ]:


filePath = "D:\\Juypter\\ColorClustered\\" 
fileNames = os.listdir(filePath)
filedirs = []


filePath_test = "D:\\Juypter\\Test\\" 
fileNames_test = os.listdir(filePath_test)
test_filedirs = []

for filename in fileNames:
    newfile_path=os.path.join(filePath,filename)
    filedirs.append(newfile_path)
for filename in fileNames_test:
    newfile_path_test=os.path.join(filePath_test,filename)
    test_filedirs.append(newfile_path_test)
print(filedirs)


# In[ ]:


#Load all of the data source
all_data = []
all_labels = []
for filedir in filedirs:
    imageNames = os.listdir(filedir)
    for imageName in imageNames:
        newImage_path=os.path.join(filedir,imageName)
        this_data = read_img(newImage_path)
        this_label = filedir.split("\\")[-1]
        all_data.append(this_data)
        all_labels.append(this_label)
data = np.vstack(all_data)
# data = data / 127.5 - 1
labels = np.hstack(all_labels)
npData =  np.array(all_data)
print(npData.shape)
print(labels)


# In[ ]:


test_data = []
test_labels = []
for test_filedir in test_filedirs:
    imageNames = os.listdir(test_filedir)
    for imageName in imageNames:
        newImage_path=os.path.join(test_filedir,imageName)
        this_data = read_img(newImage_path)
        this_label = test_filedir.split("\\")[-1]
        test_data.append(this_data)
        test_labels.append(this_label)
T_data = np.vstack(test_data)
# data = data / 127.5 - 1
T_labels = np.hstack(test_labels)
T_npData =  np.array(test_data)
print(T_npData.shape)
print(T_labels)


# In[ ]:


index = 0
print(data[index])
img = plt.imshow(data[index])


# In[ ]:


print('The image label is: ', labels[12])


# In[ ]:


y_train_one_hot = to_categorical(labels)
y_test_one_hot = to_categorical(T_labels)


# In[ ]:


print('The one hot label is:', y_train_one_hot[0])
print(len(y_train_one_hot[0]))


# In[ ]:


#Normalization
data = data / 255
x_test = T_data / 255


# In[ ]:


#Define the classification model
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(250, activation='relu'))
model.add(Dense(len(y_train_one_hot[0]), activation='softmax'))


# In[ ]:


model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:


# Training
hist = model.fit(data, y_train_one_hot, 
           batch_size=5, epochs=10, validation_split=0.2 )


# In[ ]:


#Save the model as a local weight file
model.save('D:/Juypter/model/model.h5')


# In[ ]:





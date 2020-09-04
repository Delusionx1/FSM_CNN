#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import os
import time
import threading
from cv2 import cv2 as cv2
import datetime
import subprocess
from glob import glob
from PIL import Image
import Ipynb_importer
import imp
import UseColorClassify as ucc
import UseTextureClassify as utc
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import tensorflow as tf
from tensorflow import keras


# In[2]:


now_hour = datetime.datetime.now().hour
print(now_hour)


# In[3]:


def takephoto():
    cap = cv2.VideoCapture(0)
    index = 0
    ret, frame = cap.read()
    while ret and int(now_hour)>=8 and int(now_hour)<=22:
        for index in range(100):
            ISOTIMEFORMAT = '%Y-%m-%d-%H-%M-%S'
            theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
            print(str(theTime))
            resize = cv2.resize(frame, (1600,1000), interpolation=cv2.INTER_NEAREST)
            os.chdir("D:\\Juypter\\darknet-master\\build\darknet\\x64\\collectedImages")
            os.makedirs(str(theTime))
            imgpath = str(theTime)+'/'+str(index)+'.jpg'
            cv2.imwrite(imgpath, resize)
            time.sleep(2)
            print(imgpath)
            os.chdir("D:\\Juypter\\darknet-master\\build\\darknet\\x64")
            subprocess.Popen("darknet detect cfg/yolov3.cfg yolov3.weights "+"collectedImages/"+str(imgpath), shell = True)
            time.sleep(15)

            #Root path
            root_path = "D:\\Juypter\\darknet-master\\build\\darknet\\x64\\splitedImages"
            #Used to store all file paths
            root_path = new_report(root_path)
            print(root_path)
            file_list = []
            #Used to store all file paths
            dir_list = []
            get_file_path(root_path,file_list,dir_list)
            print(file_list)

            target_dir = 'D:\\result_img_224\\'
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            file_list1 = []
            color_restlts = []
            texture_results = []

            index = 0
            for filename in file_list:
                print(filename)
                filesize = os.path.getsize(filename)
                with Image.open(filename) as im:
                        width, height = im.size
                        new_width = 224
                        new_height = 224
                        resized_im = im.resize((new_width, new_height))
                        output_filename = target_dir + str(index) + filename.split("\\")[-1]
                        resized_im.save(output_filename)
                        print(output_filename)
                        file_list1.append(output_filename)
                        index+=1;
            os.chdir("D:\\Juypter")
            #Create file to save feature image
            if not os.path.exists('Texture'):
                    os.mkdir('Texture')
            if not os.path.exists('Color'):
                    os.mkdir('Color')
            #Feature image etxeraction  
            import vgg16cnn as vcn
            for filename in file_list1:
                print(filename)
                output_dir2_2 = 'Texture\\' + filename.split("\\")[-1]
                if not os.path.exists(output_dir2_2):
                    os.mkdir(output_dir2_2)
                style_img_path = filename
                vcn.style_val = vcn.read_img(style_img_path)
                vcn.training_CNN(output_dir2_2)
                texture_comp = utc.use_texture_classify(output_dir2_2+"\\result-00100.jpg")
                texture_results.append(texture_comp)
            vcn.tf.reset_default_graph()
            
            import vgg16cnn2 as vcn2
            for filename in file_list1:
                print(filename)
                output_dir1_1 = 'Color\\' + filename.split("\\")[-1]
                if not os.path.exists(output_dir1_1):
                    os.mkdir(output_dir1_1)
                style_img_path = filename
                vcn2.style_val = vcn2.read_img(style_img_path)
                vcn2.training_CNN(output_dir1_1)
                color_comp = ucc.use_color_classify(output_dir1_1+"\\result-00100.jpg")
                color_restlts.append(color_comp)
            vcn2.tf.reset_default_graph()
            
            #Classify strategy
            for i in range(len(color_restlts)):
                memory_str = str(datetime.datetime.now()) +" "+color_restlts[i]+texture_results[i]
                file = r'D:\\Juypter\\Memory_Model.txt'
                print(memory_str)
                with open(file, 'a+') as f:
                     f.write(memory_str+'\n')

            
            ret, frame = cap.read()
            index += 1
    cap.release()
    cv2.destroyAllWindows()
    return 0 
def get_file_path(root_path,file_list,dir_list):
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path,dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            #Recursively get the path of all files and directories
            get_file_path(dir_file_path,file_list,dir_list)
        else:
            if(dir_file != "predictions.jpg"):
                file_list.append(dir_file_path)
def new_report(test_report):
    lists = []
    file_list = os.listdir(test_report)
    for dir_file in file_list:
        dir_file_path = os.path.join(test_report,dir_file)
        #Determine whether the path is a file or a path
        if os.path.isdir(dir_file_path):
            lists.append(dir_file_path)
                
    lists.sort(key=lambda fn: os.path.getmtime(fn))
    file_new = os.path.join(test_report, lists[-1])                    
    return file_new


# In[ ]:


#Start training strategy at night
if(int(now_hour) >= 8 and int(now_hour) <= 22):
    print('Begin to take pictures..........')
    takephoto()
    print('Finished !!')
if(int(now_hour) < 8 or int(now_hour) > 22):
    import ImageColorClusting
    import ImageTextureClusting
    import TrainColorClassify
    import TrainTextureClassify


# In[ ]:





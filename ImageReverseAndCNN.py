#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from glob import glob
from PIL import Image
import os as os

def get_file_path(root_path,file_list,dir_list):
    #Get all file names and directory names in the directory
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        #Get the path of the directory or file
        dir_file_path = os.path.join(root_path,dir_file)
        #Determine whether the path is a file or a path
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

#Root directory path
root_path = "D:\\Juypter\\darknet-master\\build\\darknet\\x64\\splitedImages"
#Used to store all file paths
#root_path = new_report(root_path)
print(root_path)
file_list = []
#Used to store all file paths
dir_list = []
get_file_path(root_path,file_list,dir_list)
print(file_list)


# In[ ]:


target_dir = 'D:\\result_img_224\\'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    
file_list1 = []

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


# In[ ]:


print(file_list1)


# In[ ]:


import Ipynb_importer
import vgg16cnn2 as vcn2


# In[ ]:


if not os.path.exists('Texture'):
        os.mkdir('Texture')
if not os.path.exists('Color'):
        os.mkdir('Color')


# In[ ]:


#Load the model and extract feature images
for filename in file_list1:
    print(filename)
    output_dir1_1 = 'Color\\' + filename.split("\\")[-1]
    if not os.path.exists(output_dir1_1):
        os.mkdir(output_dir1_1)
    style_img_path = filename
    vcn2.style_val = vcn2.read_img(style_img_path)
    vcn2.training_CNN(output_dir1_1)


# In[ ]:


vcn2.tf.reset_default_graph()
import vgg16cnn as vcn
for filename in file_list1:
    print(filename)
    output_dir2_2 = 'Texture\\' + filename.split("\\")[-1]
    if not os.path.exists(output_dir2_2):
        os.mkdir(output_dir2_2)
    style_img_path = filename
    vcn.style_val = vcn.read_img(style_img_path)
    vcn.training_CNN(output_dir2_2)


# In[ ]:





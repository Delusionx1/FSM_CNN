#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from PIL import Image
from PCV.clustering import hcluster
from matplotlib.pyplot import *
from numpy import *

# create a list of images

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
            if(dir_file == "result-00100.jpg"):
                file_list.append(dir_file_path)
        
#Root directory path
path = "D:\Juypter\Color"
#Used to store all file paths
imlist = []
#Used to store all the directory paths
dir_list = []
get_file_path(path,imlist,dir_list)
print(imlist)


# In[2]:


import shutil
# extract feature vector
features = zeros([len(imlist), 512])
for i, f in enumerate(imlist):
    im = array(Image.open(f))
    # multi-dimensional histogram
    h, edges = histogramdd(im.reshape(-1, 3), 8, normed=True, range=[(0, 255), (0, 255), (0, 255)])
    features[i] = h.flatten()
tree = hcluster.hcluster(features)

# visualize clusters with some (arbitrary) threshold
clusters = tree.extract_clusters(0.23 * tree.distance)

index1 = 0
# plot images for clusters with more than 3 elements
for c in clusters:

    elements = c.get_cluster_elements()
    nbr_elements = len(elements)
    if nbr_elements > 0:
        figure()
    target_dir1 = "D:\\Juypter\\ColorClustered\\"+str(index1)
    if not os.path.exists(target_dir1):
        os.makedirs(target_dir1)
        index2 = 0;
        for p in range(minimum(nbr_elements,20)):
            print(imlist[elements[p]])
            
            file_name,file_extend=os.path.splitext(imlist[elements[p]].split('\\')[-1])
            new_name=file_name+str(index2)+file_extend  
            newfile_path=os.path.join(target_dir1,new_name)
            print(newfile_path)
            shutil.copy(imlist[elements[p]], newfile_path)
            subplot(4, 5, p + 1)
            im = array(Image.open(imlist[elements[p]]))
            imshow(im)
            index2+=1
            axis('off')
    print("--------------")
    index1+=1
show()


# In[ ]:





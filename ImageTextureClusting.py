#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from PIL import Image
from PCV.clustering import hcluster
from matplotlib.pyplot import *
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import misc
import imageio


# create a list of images

def get_file_path(root_path,file_list,dir_list):
    #Get all file names and directory names in the directory
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path,dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            get_file_path(dir_file_path,file_list,dir_list)
        else:
            if(dir_file == "result-00100.jpg"):
                file_list.append(dir_file_path)
#Root directory path
path = "D:\Juypter\Texture"
#Used to store all file paths
imlist = []
#Used to store all the directory paths
dir_list = []
get_file_path(path,imlist,dir_list)
print(imlist)
    
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


# In[ ]:


target_dir = 'D:\\Juypter\\TextureGray\\'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
index = 0
for filename in imlist:
    lena = mpimg.imread(filename)
    gray = rgb2gray(lena) 
    plt.figure(figsize=(0.747, 0.747)) 
    plt.imshow(gray, cmap='Greys_r')
    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    print(target_dir+str(index)+'.jpg')
    plt.savefig(target_dir+str(index)+'.jpg',bbox_inches='tight',dpi=300,pad_inches=0.0)
    plt.show()
    index += 1;


# In[ ]:


def get_file_path2(root_path,file_list,dir_list):
    #Get all file names and directory names in the directory
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path,dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            get_file_path(dir_file_path,file_list,dir_list)
        else:
            file_list.append(dir_file_path)


# In[ ]:


#Used to store all file paths
imlist2 = []
#Used to store all the directory paths
dir_list2 = []
get_file_path2('D:\\Juypter\\TextureGray',imlist2,dir_list2)
print(imlist2)


# In[ ]:


import shutil
# extract feature vector (8 bins per color channel)
features = zeros([len(imlist2), 512])
for i, f in enumerate(imlist2):
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
    target_dir1 = "D:\\Juypter\\TextureClustered\\"+str(index1)
    if not os.path.exists(target_dir1):
        os.makedirs(target_dir1)
        index2 = 0;
        for p in range(minimum(nbr_elements,20)):
            print(imlist2[elements[p]])
            file_name,file_extend=os.path.splitext(imlist2[elements[p]].split('\\')[-1])
            #Add a suffix to the end of the file name
            new_name=file_name+str(index2)+file_extend   
            newfile_path=os.path.join(target_dir1,new_name)
            print(newfile_path)
            shutil.copy(imlist2[elements[p]], newfile_path)
            
            subplot(4, 5, p + 1)
            im = array(Image.open(imlist2[elements[p]]))
            imshow(im)
            index2+=1
            axis('off')
    print("--------------")
    index1+=1
show()


# In[ ]:





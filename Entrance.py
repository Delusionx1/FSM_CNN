#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import threading
from cv2 import cv2 as cv2
import datetime
import subprocess

def takephoto():
    cap = cv2.VideoCapture(0)
    index = 0
    ret, frame = cap.read()
    while ret:
        for index in range(100):
            #Save the file as the file name of date
            ISOTIMEFORMAT = '%Y-%m-%d-%H-%M-%S'
            theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
            print(str(theTime))
            #Set the resolution of image
            resize = cv2.resize(frame, (1600,1000), interpolation=cv2.INTER_NEAREST)
            #Set the image saved location
            os.chdir("D:\\Juypter\\darknet-master\\build\darknet\\x64\\collectedImages")
            os.makedirs(str(theTime))
            imgpath = str(theTime)+'/'+str(index)+'.jpg'
            cv2.imwrite(imgpath, resize)
            time.sleep(2)
            print(imgpath)
            os.chdir("D:\\Juypter\\darknet-master\\build\\darknet\\x64")
            #Use commond line to running the YOLOv3
            subprocess.Popen("darknet detect cfg/yolov3.cfg yolov3.weights "+"collectedImages/"+str(imgpath), shell = True)
            #print(d.read())
            
            time.sleep(8)
            ret, frame = cap.read()
            index += 1

    cap.release()
    cv2.destroyAllWindows()
    return 0 

if __name__=='__main__':
    print('Begin to take pictures..........')
    takephoto()
    print('Finished !!')


# In[ ]:





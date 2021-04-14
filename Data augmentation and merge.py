# -*- coding: utf-8 -*-

import os
from PIL import Image
import cv2
import numpy
import random, shutil
import math

#data augmentation
def Read(directory_name,SAVE_MARK_DIR):
    filelist=os.listdir(directory_name)
    filelist.sort(key=lambda x:int(x[:-4]))
    s=0
    for filename in filelist:
        img = Image.open(directory_name + "/" + filename)
        w = 80  
        h = 80         
        
       
        X=[0,7,14,20]
        Y=[0,7,14,20]
        
        for x in X:
            for y in Y:

                region = img.crop((x, y, x+w, y+h))     #裁剪区域
                region.save(SAVE_MARK_DIR+str(s)+".png")
                s=s+1

# merge
def merge(image_dir1,image_dir2,image_dir3,image_dir4,image_dir5,image_dir6,image_dir7,image_dir8,image_dir9,image_dir10,image_dir11,savepath):  
    array= []
    n=2088
    for w in range(0,n):
        w=str(w)
        array.append(w)
    print(array)
    
    for a in array:
        dir1=os.path.join(image_dir1,'%s.png'%(a))
        img1 = cv2.imread(dir1)
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    
        dir2=os.path.join(image_dir2,'%s.png'%(a))
        img2 = cv2.imread(dir2)
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
        dir3=os.path.join(image_dir3,'%s.png'%(a))
        img3 = cv2.imread(dir3)
        img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    
        dir4=os.path.join(image_dir4,'%s.png'%(a))
        img4= cv2.imread(dir4)
        img4 = cv2.cvtColor(img4,cv2.COLOR_BGR2GRAY)
    
        dir5=os.path.join(image_dir5,'%s.png'%(a))
        img5 = cv2.imread(dir5)
        img5 = cv2.cvtColor(img5,cv2.COLOR_BGR2GRAY)
    
        dir6=os.path.join(image_dir6,'%s.png'%(a))
        img6 = cv2.imread(dir6)
        img6 = cv2.cvtColor(img6,cv2.COLOR_BGR2GRAY)
    
        dir7=os.path.join(image_dir7,'%s.png'%(a))
        img7 = cv2.imread(dir7)
        img7 = cv2.cvtColor(img7,cv2.COLOR_BGR2GRAY)
        
        dir8=os.path.join(image_dir8,'%s.png'%(a))
        img8 = cv2.imread(dir8)
        img8 = cv2.cvtColor(img8,cv2.COLOR_BGR2GRAY)
        
        dir9=os.path.join(image_dir9,'%s.png'%(a))
        img9 = cv2.imread(dir9)
        img9 = cv2.cvtColor(img9,cv2.COLOR_BGR2GRAY)
        
        dir10=os.path.join(image_dir10,'%s.png'%(a))
        img10 = cv2.imread(dir10)
        img10 = cv2.cvtColor(img10,cv2.COLOR_BGR2GRAY)
        
        dir11=os.path.join(image_dir11,'%s.png'%(a))
        img11 = cv2.imread(dir11)
        img11 = cv2.cvtColor(img11,cv2.COLOR_BGR2GRAY)

        res = cv2.merge([img1, img2, img3,img4, img5, img6,img7,img8, img9, img10,img11])
        print(a)
        print(res.shape)
        numpy.save(os.path.join(savepath,'%s.npy'%(a)),res)   
              
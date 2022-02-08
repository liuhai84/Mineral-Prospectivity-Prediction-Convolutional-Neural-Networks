# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import random, shutil



#Before run, please copy the file folder of "sample" and renamed as "train" 

#80% for train, 20% for verify
def moveFile(fileDir,tarDir):
        pathDir = os.listdir(fileDir)  
        filenumber=len(pathDir)
        print(filenumber)
        rate=0.2   
        picknumber=int(filenumber*rate)
        print(picknumber)
        sample = random.sample(pathDir, picknumber) 
        print (sample)
        for name in sample:
                shutil.move(fileDir+name, tarDir+name)
        return


if __name__ == '__main__':
    fileDir = 'Data/train/0/'    
    tarDir = 'Data/verify/0/'    
    moveFile(fileDir,tarDir)
    
    fileDir = 'Data/train/1/'    
    tarDir = 'Data/verify/1/'    
    moveFile(fileDir,tarDir)


def data_augmentation(directory_name,SAVE_MARK_DIR):
    filelist=os.listdir(directory_name)
    filelist.sort(key=lambda x:int(x[:-4]))
    s=0
    for filename in filelist:
        image = np.load(directory_name + "/" +filename)
        rows, cols, chn = image.shape
       
        w = 80  
        h = 80         
        
        X=[0,7,14,20]
        Y=[0,7,14,20]
                
        for x in X:
              for y in Y:  
                   region1= image[:, :, 0][x:x+w,y:y+h] 
                   region2= image[:, :, 1][x:x+w,y:y+h]
                   region3= image[:, :, 2][x:x+w,y:y+h] 
                   region4= image[:, :, 3][x:x+w,y:y+h] 
                   region5= image[:, :, 4][x:x+w,y:y+h] 
                   region6= image[:, :, 5][x:x+w,y:y+h] 
                   region7= image[:, :, 6][x:x+w,y:y+h] 
                   region8= image[:, :, 7][x:x+w,y:y+h] 
                   region9= image[:, :, 8][x:x+w,y:y+h]  
                   region10= image[:, :, 9][x:x+w,y:y+h] 
                   region11= image[:, :, 10][x:x+w,y:y+h] 
                   res = cv2.merge([region1, region2, region3,region4, region5, region6,region7,region8, region9, region10,region11])
                   print(res.shape) 
                   np.save(os.path.join(SAVE_MARK_DIR,'%s.npy'%(s)),res)  
                   s=s+1
                   
directory_name= 'Data/train/0/'    
SAVE_MARK_DIR='Data/train_expand/0/'
data_augmentation(directory_name,SAVE_MARK_DIR)  

directory_name= 'Data/train/1/'    
SAVE_MARK_DIR='Data/train_expand/1/'
data_augmentation(directory_name,SAVE_MARK_DIR)              

directory_name= 'Data/verify/0/'    
SAVE_MARK_DIR='Data/verify_expand/0/'
data_augmentation(directory_name,SAVE_MARK_DIR) 

directory_name= 'Data/verify/1/'    
SAVE_MARK_DIR='Data/verify_expand/1/'
data_augmentation(directory_name,SAVE_MARK_DIR) 


if __name__ == '__main__':
    train_list = ''
    file_dir = 'Data/train_expand'
    labels = ['0','1']
    for label in labels:
        dir = file_dir + '/' + label
        for root, dirs, files in os.walk(dir):
            for file in files:
                train_list += dir + '/' + file + ' ' + label + '\n'
    with open('Data/train_expand/data_list.txt', 'w') as f:
        f.write(train_list[:-1])
        
    test_list = ''
    file_dir = 'Data/verify_expand'
    labels = ['0','1']
    for label in labels:
        dir = file_dir + '/' + label
        for root, dirs, files in os.walk(dir):
            for file in files:
                test_list += dir + '/' + file + ' ' + label + '\n'
    with open('Data/verify_expand/data_list.txt', 'w') as f:
        f.write(test_list[:-1])   
              
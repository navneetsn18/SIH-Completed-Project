# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:33:21 2020

@author: tarun aggarwal
"""

import cv2
import os
import pickle

dataPath=os.getcwd()+"/Data/"

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
def det(face_id,name):
    
#make sure 'haarcascade_frontalface_default.xml' is in the same folder as this code
    face_detector = cv2.CascadeClassifier(dataPath+'haarcascades/haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id (must enter number start from 1, this is the lable of person 1)
#face_id = input('\n enter user id end press <return> ==>  ')
#name=input('\n enter the name of person ')
    data_set =[str(face_id),name]

    with open(dataPath+'timestamps/id.txt', 'a') as filehandle:
        
        for listitem in data_set:
            for k in listitem:
                filehandle.write('%s' % k)
            filehandle.write('\t')     
        filehandle.write('\n')
    

     
#Initializing face capture. Look the camera and wait ...
# Initialize individual sampling face count
    count = 0

#start detect your face and take 30 pictures
    while(True):
        print(count)
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('camera',img)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1

        # Save the captured image into the datasets folder
            #print(img.shape)
            img=cv2.resize(img, (150,150),interpolation=cv2.INTER_AREA)
    

            cv2.imwrite(dataPath+"/dataset/" + str(face_id) + '.'  + str(count) + ".jpg", gray[y:y+h,x:x+w])
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if count == 1:
            cv2.imwrite(dataPath +  "photos/" + str(face_id)  + ".jpg", gray[y:y+h,x:x+w])
        
        if k == 27:
            break
        elif count >= 10: # Take 30 face sample and stop video
            break

# Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

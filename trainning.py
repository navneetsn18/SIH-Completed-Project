# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:33:41 2020

@author: tarun aggarwal
"""

import cv2
import numpy as np
from PIL import Image #pillow package
import os

# Path for face image database
dataPath=os.getcwd()+"/Data/"
path = dataPath+"dataset/"

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer1=cv2.face.EigenFaceRecognizer_create()
recognizer2= cv2.face.FisherFaceRecognizer_create()

detector = cv2.CascadeClassifier(dataPath+"haarcascades/haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[0])
        print(id)
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            #faceSamples.append(img_numpy[y:y+h,x:x+w])
            faceSamples.append(cv2.resize(img_numpy[y:y+h,x:x+w],(280,280)))
            ids.append(id)
            

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
recognizer1.train(faces,np.array(ids))
recognizer2.train(faces,np.array(ids))
# Save the model into trainer/trainer.yml
recognizer.write(dataPath+'trainer/trainer.yaml') # recognizer.save() worked on Mac, but not on Pi

recognizer1.write(dataPath+'trainer/trainer1.yaml') 
recognizer2.write(dataPath+'trainer/trainer2.yaml') 
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
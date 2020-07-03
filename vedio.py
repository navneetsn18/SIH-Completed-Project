# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:16:34 2020

@author: tarun aggarwal
"""

import cv2
import numpy as np
import os 
import time
import winsound

dataPath=os.getcwd()+"/Data/"


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer1=cv2.face.EigenFaceRecognizer_create()
recognizer2=cv2.face.FisherFaceRecognizer_create()
recognizer.read(dataPath+'trainer/trainer.yaml') 
recognizer1.read(dataPath+'trainer/trainer1.yaml')  #load trained model
recognizer2.read(dataPath+'trainer/trainer2.yaml') 
cascadePath = dataPath+"haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter, the number of persons you want to include
id = 2 #two persons (e.g. Jacob, Jack)



data_set = []

# open file and read the content in a list
with open(dataPath+"timestamps/"+'id.txt', 'r') as filehandle:
    for line in filehandle:
        temp=list(line.split('\t'))
        
        data_set.append(temp)
            

        # add item to the list

print(data_set)
user_id=[]
for i in data_set:
    user_id.append(i[0])
id=max(user_id)
user_name=[]
for i in data_set:
    user_name.append(i[1])
names=[" "]+user_name
 #key in names, start from the second place, leave first empty

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img =cam.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    i=0
    
    for(x,y,w,h) in faces:
    
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        id1, confidence1 = recognizer1.predict(cv2.resize(gray[y:y+h,x:x+w],(280,280)))
        id2, confidence2 = recognizer2.predict(cv2.resize(gray[y:y+h,x:x+w],(280,280)))
        
        
        # Check if confidence is less them 100 ==> "0" is perfect match 
        print(confidence1,confidence2)
      
        if (id==id1==id2):
            
            confidence = "  {0}%".format(round(100 - confidence))
            file = open(dataPath+'timestamps/'+str(id)+'.txt','a') 
            id = names[id]
            
            file.write(time.asctime( time.localtime(time.time()) )+'\n') 
            file.close() 
            
          
        else:
            winsound.Beep(1000,100)
            id = "unknown"
            i+=1
            confidence = "  {0}%".format(round(100 - confidence))
            cv2.imwrite(dataPath+"unknown/" + "unknown"+str(i)+".jpg", gray[y:y+h,x:x+w])
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

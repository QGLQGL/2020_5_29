import os
import cv2 as cv
import sys
from PIL import Image
import numpy as np

def getImageAndLabels(path):
    facesSamples = []
    ids = []
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    #print(imagePaths)
    
    face_engine = cv.CascadeClassifier('/home/pi/Downloads/opencv-3.4.0/data/haarcascades/haarcascade_frontalface_default.xml')

    
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        faces = face_engine.detectMultiScale(img_numpy,scaleFactor=1.1,minNeighbors=5)
        #print(os.path.split(imagePath))
        id = int(os.path.split(imagePath)[1].split('.')[0])
        #id = int(os.path.split(imagePath)[-1].split('.')[0])
        for x,y,w,h in faces:
            facesSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    
    return faceSamples,ids

if __name__ == '__main__':
    #tu pian lu jing
    path = './data/da/'
    faces,ids = getImageAndLabels(path)
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(faces,np.arry(ids))
    recognizer.write('tr/tr.yml/')
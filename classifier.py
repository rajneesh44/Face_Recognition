import numpy as np 
from PIL import Image
import os 
import cv2
data='D:/images'
def train_classifier(data):
    #creating a list that contains all the files (photos)
    path = [os.path.join(data,f) for f in os.listdir(data)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNp=np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)
    clf=cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("Classifier.yml")
train_classifier(data)

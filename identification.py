import cv2
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils
import sys


#CODE ONLY USED IF NEURALNET IS NOT IN SAME FOLDER AS CURRENT DIRECTORY
#sys.path.insert(1, '././githubFiles/DIP-Final/')

from NeuralNet import Mymodel
import numpy as np
import torch
from dataset_loader import dataset_loader
from PIL import Image

#MODEL PATH SHOULD BE LOCAL
#model_path='./githubFiles/DIP-Final/weights.pth'
model_path = "./weights.pth"

model=Mymodel()

#WAS NECESSARY TO ADD MAP_LOCATION COMMAND ARG TO TORCH.LOAD ON MY COMPUTER, LIKELY UNNNECESSARY IF YOU HAVE GPU
checkpoint=torch.load(model_path, map_location=torch.device('cpu'))


model.load_state_dict(checkpoint)
model.eval()



def maskIdentify(imgpath):

    frame = cv2.imread(imgpath)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    dnnFaceDetector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")

    #FACE DETECTOR TAKES GRAYSCALE CV IMAGE
    rects = dnnFaceDetector(gray, 1)


    for (i, rect) in enumerate(rects):
        x1 = rect.rect.left()
        y1 = rect.rect.top()
        x2 = rect.rect.right()
        y2 = rect.rect.bottom()

        #USE OFFSET VALUE TO alter Frame size, negative values result in tighter boxes while positive value correspond with wider boxes.
        offset = 5

        #RESIZE IMAGE TO 60 by 60
        face = cv2.resize(frame[y1-offset:y2+offset, x1-offset:x2+offset], (60,60))

        #pilIMG = Image.fromarray(face)

        #TURN ARRAY FROM 3D to 4D
        npInput = []
        npInput.append(face)

        #RUN PREDICTION AND PRINT
        x=np.moveaxis(npInput,-1,1)
        x_t=torch.tensor(x,dtype=torch.float32)
        y_pred=model.forward(x_t)
        print(y_pred)
        # plt.imshow(face)
        # plt.show()
        # print(x2-x1)
        # print(y2-y1)

        # Rectangle around the face

        #IMAGE FOR SHOWING FACE THAT IS BEING DETECTED AND ADDING RECTANGLE AROUND FACES IN FINAL IMAGE
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        plt.imshow(face)
        plt.show()
    plt.imshow(frame)
    plt.show()


#CHANGE FILENAME TO WHATEVER LOCATION YOU HAVE FOR IMAGES
filename = "./archive/images/maksssksksss"
for i in range(3):
    maskIdentify(filename+str(i)+".png")

# filename = sys.argv[1]
# print(filename)
# maskIdentify(filename)

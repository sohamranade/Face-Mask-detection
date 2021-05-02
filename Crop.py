import cv2
import numpy as np
import os

labels = []
newLabel = True
labelNum = -1

f = open(r'E:\Subjects\Columbia Subjects\DIP\DIP-Final\labels.txt')

for line in f:
    if len(line.split()) < 5:
        newLabel = True
        continue
    if newLabel:
        labels.append([])
        labelNum = labelNum + 1
    valuesInt = []
    values = line.split()
    for nums in values:
        n = int(nums)
        valuesInt.append(n)
    labels[labelNum].append(valuesInt)
    newLabel = False

f.close


pathImgBefore = r'E:\Subjects\Columbia Subjects\DIP\DIP-Final\dataset\train-images\images'
images = os.listdir(pathImgBefore)

pathImgAfter = r'E:\Subjects\Columbia Subjects\DIP\DIP-Final\dataset\imagesFace'
#os.mkdir(pathImgAfter)
#os.chdir(pathImgAfter)
pathCorrect = r'\MaskCorrect'
pathIncorrect = r'\MaskIncorrect'
#os.mkdir(pathCorrect)
#os.mkdir(pathIncorrect)
correct_count = 0
incorrect_count=0

for imageNum in range(len(labels)):
    img = cv2.imread(pathImgBefore + '\\' + images[imageNum])
    for face in labels[imageNum]:
        faceCrop = img[face[1]:face[1]+face[3]-1,face[0]:face[0]+face[2]-1,:]
        if face[13] == 3 and face[12] !=3:
            os.chdir(pathImgAfter + pathCorrect)
            faceCropRes = cv2.resize(faceCrop, (60, 60))
            cv2.imwrite(str(correct_count) + r'.jpg', faceCropRes)
            correct_count+=1
        else:
            os.chdir(pathImgAfter + pathIncorrect)
            faceCropRes = cv2.resize(faceCrop, (60, 60))
            cv2.imwrite(str(incorrect_count) + r'.jpg', faceCropRes)
            incorrect_count+=1 

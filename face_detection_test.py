import os
import cv2
import dlib
from imutils import face_utils
from NeuralNet import Mymodel
import numpy as np
import torch
from dataset_loader import dataset_loader
from PIL import Image
import pandas as pd
import csv

# Load the cascade
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade= cv2.CascadeClassifier('haarcascade_eye.xml')
# # Read the input image

def MaskIdentify(image,rects, model):
	masks=[]
	for i,rect in enumerate(rects):
		offset= 10
		width, height, _ = image.shape
		#print(width,height)
		x1= rect.rect.left()-offset
		x2=rect.rect.right()+offset
		y1=rect.rect.top()-offset
		y2=rect.rect.bottom()+offset
		if  x1 < 0:
			x1 = 0
		if y1 < 0:
		 	y1 = 0
		if x2> height:
		 	x2=height
		if y2> width:
		 	y2=width
		#print(x1,x2,y1,y2)
		face = cv2.resize(image[y1:y2, x1:x2], (60,60))
		face=np.expand_dims(face,0)
		x=np.moveaxis(face,-1,1)
		x_t=torch.tensor(x,dtype=torch.float32)
		model.eval()
		y_pred=model.forward(x_t)
		y_pred=y_pred.detach().numpy()
		y_pred[y_pred>.5]=1
		y_pred[y_pred<=.5]=0
		y_pred = int(y_pred)
		masks.append(y_pred)

	return masks




def read_txt(txt_path):
	data=open(txt_path, "r")
	lis=[]
	for rows in data:
		val=rows.rstrip("\n").split(',')
		val=[int(i) for i in val]
		#print(len(val))
		lis.append(val)
	#print(len(lis))
	return lis


def get_no_people(image,model):
	rects=model(image,1)

	if rects:
		num=len(rects)
	else:
		num=0
	return rects,num




#SECTION WITH TRANSFORMS
def eqHist(img):
    hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    i = hsi[:,:,2]
    i = cv2.equalizeHist(i)
    hsi[:,:,2] = i
    return cv2.cvtColor(hsi, cv2.COLOR_HSV2BGR)

def gaussian(img):
    return cv2.GaussianBlur(img,(7,7),0)

def median(img):
    return cv2.medianBlur(img,3)

def sharp(img):
    return cv2.addWeighted(img, 2, gaussian(img), -1, 0)



model_path='./weights.pth'
model=Mymodel()
checkpoint=torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()

dnn_path='./mmod_human_face_detector.dat'
dnnFaceDetector = dlib.cnn_face_detection_model_v1(dnn_path)

Images_path= './dataset/ImagesForTestResized1/'
labels_path= './dataset/ImagesForTest/labelsTest.txt'

#total_images=len(os.listdir(Images_path))
labels=read_txt(labels_path)
#print(len(labels))
error = np.zeros((5,3))
histError = np.zeros((5,3))
gaussError = np.zeros((5,3))
medianError = np.zeros((5,3))
sharpError = np.zeros((5,3))

#for i in range (1,501):
for i in range (1,500):
	if not i%100 == 1:
		image=cv2.imread(Images_path+str(i)+'.jpg')


		#IMAGE EDITING


		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image2=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

		people,num =get_no_people(gray,dnnFaceDetector)
		detected = MaskIdentify(image2, people, model)

		# print("Detected Values In Image")
		# print(detected)

		correct = labels[(i%100)-1]

		# print("Correct Values In Image")
		# print(correct)

		error[int((i-1)/100)][2] += len(correct)

		error[int((i-1)/100)][0] += (len(correct)-len(detected))
		print(i)
		for value in detected:
			try:
				correct.index(value)
				correct.remove(value)
			except ValueError:
				pass
		error[int((i-1)/100)][1] += len(correct)

for i in range (1,500):
	if not i%100 == 1:


		image=cv2.imread(Images_path+str(i)+'.jpg')

		#IMAGE EDITING
		image = eqHist(image)

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image2=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

		people,num =get_no_people(gray,dnnFaceDetector)
		detected = MaskIdentify(image2, people, model)

		# print("Detected Values In Image")
		# print(detected)

		correct = labels[(i%100)-1]

		# print("Correct Values In Image")
		# print(correct)

		histError[int((i-1)/100)][2] += len(correct)

		histError[int((i-1)/100)][0] += (len(correct)-len(detected))
		for value in detected:
			try:
				correct.index(value)
				correct.remove(value)
			except ValueError:
				pass
		histError[int((i-1)/100)][1] += len(correct)
		print(i+500)


for i in range (1,500):
	if not i%100 == 1:
		image=cv2.imread(Images_path+str(i)+'.jpg')
		image = gaussian(image)
		#IMAGE EDITING


		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image2=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

		people,num =get_no_people(gray,dnnFaceDetector)
		detected = MaskIdentify(image2, people, model)

		# print("Detected Values In Image")
		# print(detected)

		correct = labels[(i%100)-1]

		# print("Correct Values In Image")
		# print(correct)

		gaussError[int((i-1)/100)][2] += len(correct)

		gaussError[int((i-1)/100)][0] += (len(correct)-len(detected))

		for value in detected:
			try:
				correct.index(value)
				correct.remove(value)
			except ValueError:
				pass
		gaussError[int((i-1)/100)][1] += len(correct)
		print(i+1000)


for i in range (1,500):
	if not i%100 == 1:
		image=cv2.imread(Images_path+str(i)+'.jpg')
		image = median(image)
		#IMAGE EDITING


		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image2=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

		people,num =get_no_people(gray,dnnFaceDetector)
		detected = MaskIdentify(image2, people, model)

		# print("Detected Values In Image")
		# print(detected)

		correct = labels[(i%100)-1]

		# print("Correct Values In Image")
		# print(correct)

		medianError[int((i-1)/100)][2] += len(correct)

		medianError[int((i-1)/100)][0] += (len(correct)-len(detected))

		for value in detected:
			try:
				correct.index(value)
				correct.remove(value)
			except ValueError:
				pass
		medianError[int((i-1)/100)][1] += len(correct)
		print(i+1500)

for i in range (1,500):
	if not i%100 == 1:
		image=cv2.imread(Images_path+str(i)+'.jpg')
		image = sharp(image)
		#IMAGE EDITING


		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image2=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

		people,num =get_no_people(gray,dnnFaceDetector)
		detected = MaskIdentify(image2, people, model)

		# print("Detected Values In Image")
		# print(detected)

		correct = labels[(i%100)-1]

		# print("Correct Values In Image")
		# print(correct)

		sharpError[int((i-1)/100)][2] += len(correct)

		sharpError[int((i-1)/100)][0] += (len(correct)-len(detected))

		for value in detected:
			try:
				correct.index(value)
				correct.remove(value)
			except ValueError:
				pass
		sharpError[int((i-1)/100)][1] += len(correct)
		print(i+2000)

		# print("Error Values for Verification")
		# #writer.writerows(error)
		# print(error)
		#cv2.waitKey(1)

str_error = list(np.zeros((5), dtype=str))
str_error1 = list(np.zeros((5), dtype=str))
str_error2 = list(np.zeros((5), dtype=str))
str_error3 = list(np.zeros((5), dtype=str))
str_error4 = list(np.zeros((5), dtype=str))

for i, e in enumerate(error):
	str_error[i] = "   ".join([str(x) for x in e])
for i, e in enumerate(histError):
	str_error1[i] = "   ".join([str(x) for x in e])
for i, e in enumerate(gaussError):
	str_error2[i] = "   ".join([str(x) for x in e])
for i, e in enumerate(medianError):
	str_error3[i] = "   ".join([str(x) for x in e])
for i, e in enumerate(sharpError):
	str_error4[i] = "   ".join([str(x) for x in e])



with open('output.csv', 'w', newline='') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=',')
	spamwriter.writerow(["Filter Types", "No Noise", "Gaussian Blur", "Salt and Pepper", "Darkened", "Brightened"])
	spamwriter.writerow(["No Filter", str_error[0], str_error[1], str_error[2], str_error[3], str_error[4]])
	spamwriter.writerow(["Histogram Equalization", str_error1[0], str_error1[1], str_error1[2], str_error1[3], str_error1[4]])
	spamwriter.writerow(["Gaussian Filter", str_error2[0], str_error2[1], str_error2[2], str_error2[3], str_error2[4]])
	spamwriter.writerow(["Median Filter", str_error3[0], str_error3[1], str_error3[2], str_error3[3], str_error3[4]])
	spamwriter.writerow(["Sharpening Filter", str_error4[0], str_error4[1], str_error4[2], str_error4[3], str_error4[4]])


cv2.destroyAllWindows()

	#Subtract 1 as 100 will return wrong index



#cv2.destroyAllWindows()


#	rects = dnnFaceDetector(gray, 1)


# 	if rects:

# 		for i,rect in enumerate(rects):
# 			offset= 10
# 			x1,x2,y1,y2= rect.rect.left()-offset ,rect.rect.right()+offset ,rect.rect.top()-offset ,rect.rect.bottom()+offset

# 			cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

# 			face = cv2.resize(image2[y1:y2, x1:x2], (60,60))
# 			face=np.expand_dims(face,0)
# 			x=np.moveaxis(face,-1,1)
# 			x_t=torch.tensor(x,dtype=torch.float32)
# 			model.eval()
# 			y_pred=model.forward(x_t)
# 			y_pred=y_pred.detach().numpy()
# 			y_pred[y_pred>.5]=1
# 			y_pred[y_pred<=.5]=0
# 			print(y_pred)
# 	cv2.imshow('image',image)
# 	cv2.waitKey(0)

# cv2.destroyAllWindows()

#labels=read_txt(labels_path)

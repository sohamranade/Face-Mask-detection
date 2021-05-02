from  dataset_loader import dataset_loader
from PIL import Image
import numpy as np 

CMPath='E:/Subjects/Columbia Subjects/DIP/DIP-Final/dataset/ImagesFace/MaskCorrect'
IMPath='E:/Subjects/Columbia Subjects/DIP/DIP-Final/dataset/ImagesFace/MaskIncorrect'
total_images=10
x,y=dataset_loader(CMPath,IMPath,total_images)

for i in range(10):
	image=Image.fromarray(x[i],'RGB')
	image.show()
	print(y[i])


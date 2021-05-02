import os 
import numpy as np 
from PIL import Image

def dataset_loader(p0,p1,total_images):
	
	corr_image_list=[]
	corr_labels=[]
	incorr_image_list=[]
	incorr_labels=[]

	for imageName in range(total_images//2):
		imagePathC= p0 +'/'+str(imageName)+ '.jpg'
		imagePathI= p1 +'/'+str(imageName)+ '.jpg'
		imageC=Image.open(imagePathC)
		imageI=Image.open(imagePathI)
		image_arrC=np.array(imageC)
		image_arrI=np.array(imageI)

	#print(image_arr.shape)
		corr_image_list.append(image_arrC)
		corr_labels.append(0)
		incorr_image_list.append(image_arrI)
		incorr_labels.append(1)

	corr_image_list=np.array(corr_image_list)
	incorr_image_list=np.array(incorr_image_list)
	corr_labels=np.array(corr_labels)
	incorr_labels=np.array(incorr_labels)

	image_list=np.append(corr_image_list,incorr_image_list,axis=0)
	labels_list=np.append(corr_labels,incorr_labels,axis=0)
	random=np.random.choice(total_images,size=total_images,replace=False)
	image_list=image_list[random]
	labels_list=labels_list[random]
	labels_list=np.expand_dims(labels_list,1)

	return (image_list, labels_list)



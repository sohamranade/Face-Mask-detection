from NeuralNet import Mymodel
import numpy as np
import torch
from dataset_loader import dataset_loader
from PIL import Image
model_path='E:/Subjects/Columbia Subjects/DIP/DIP-Final/model/epoch_0030_loss_0.89861111/weights.pth'
model=Mymodel()
checkpoint=torch.load(model_path)
model.load_state_dict(checkpoint)
model.eval()

total_images=10
CMPath='E:/Subjects/Columbia Subjects/DIP/DIP-Final/dataset/ImagesFace/MaskCorrect'
IMPath='E:/Subjects/Columbia Subjects/DIP/DIP-Final/dataset/ImagesFace/MaskIncorrect'

x,y=dataset_loader(CMPath,IMPath,total_images)
for image in x:
	image=Image.fromarray(image)
	image.show()
x=np.moveaxis(x,-1,1)
x_t=torch.tensor(x,dtype=torch.float32)
y_pred=model.forward(x_t)
y_pred=y_pred.detach().numpy()
y_pred[y_pred>.5]=1
y_pred[y_pred<=.5]=0
print(y_pred)
print(y)

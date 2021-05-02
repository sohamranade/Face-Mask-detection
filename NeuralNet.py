import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,random_split
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
import os
from dataset_loader import dataset_loader

class Mymodel(nn.Module):
	def __init__(self,c=3,o_f=1):
		super(Mymodel,self).__init__()

		self.layer1=nn.Sequential(nn.Conv2d(c,10,kernel_size=5,stride=2,padding=0,bias=False),
			nn.ReLU(),
			nn.AvgPool2d(kernel_size=2,stride=2)
			)
		self.layer2=nn.Sequential(nn.Conv2d(10,40,kernel_size=5,stride=2,padding=0,bias=False),
			nn.ReLU(),
			nn.AvgPool2d(kernel_size=2,stride=2)
			)
		self.layer3=nn.Sequential(nn.Linear(2*2*40,100),
			nn.Dropout(p=.5),
			nn.ReLU(),
			nn.Linear(100,o_f),
			nn.Sigmoid()
			)
	
	def forward(self,Input):
		out=self.layer1(Input)
		out=self.layer2(out)
		out=torch.flatten(out,start_dim=1)
		out=self.layer3(out)
		return out

class DynamicDataset(Dataset):
	def __init__(self,x,y):
		self.X=x
		self.Y=y
	
	def __len__(self):
		return len(self.X)

	
	def __getitem__(self,idx):
		return self.X[idx],self.Y[idx]

# total_images=10
# corr_image_list=[]
# corr_labels=[]
# incorr_image_list=[]
# incorr_labels=[]

def train(model,train_loader, criterion, optimizer, train_size,device):
	model.train()
	train_loss=0
	correct=0
	for i,(features,labels) in enumerate(train_loader):
		features,labels=features.float().to(device),labels.float().to(device)
		y_pred=model.forward(features)
		loss=criterion(y_pred,labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_loss+=loss.item()/train_size
		for i,pred in enumerate(y_pred):
			pred=(0 if pred<=.5 else 1)
			if pred==labels[i,:]:
				correct+=1
	percent_train=correct/train_size
	return train_loss,percent_train

def test(model,test_loader,test_size,device):
	model.eval()
	num_correct=0
	#print(len(test_loader))
	with torch.no_grad():
		for i ,(feature,label) in enumerate(test_loader):
			#print(feature.shape,label.shape)
			feature,label=feature.float().to(device),label.float().to(device)
			y_pred=model.forward(feature)
			#print(y_pred.shape)
			y_pred=(0 if y_pred<=.5 else 1)
			if y_pred==label:
				num_correct+=1
		percent= num_correct/test_size

	return percent

# def test(model,test_loader, criterion, optimizer, test_size,device):
# 	model.eval()
# 	test_loss=0
# 	for i,(features,labels) in enumerate(train_loader):
# 		features,labels=features.float().to(device),labels.float().to(device)
# 		y_pred=model.forward(features)
# 		loss=criterion(y_pred,labels)
# 		#optimizer.zero_grad()
# 		#loss.backward()
# 		#optimizer.step()
# 		test_loss+=loss.item()/train_size
# 	return test_loss




def main():

	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")  
	CMPath='E:/Subjects/Columbia Subjects/DIP/DIP-Final/dataset/ImagesFace/MaskCorrect'
	IMPath='E:/Subjects/Columbia Subjects/DIP/DIP-Final/dataset/ImagesFace/MaskIncorrect'
	save_dir='E:/Subjects/Columbia Subjects/DIP/DIP-Final/model'
	total_images=9000
	x,y=dataset_loader(CMPath,IMPath,total_images)
	x=np.moveaxis(x,-1,1)
	dataset=DynamicDataset(x,y)
	dataset_size=dataset.__len__()
	split=0.4
	epochs=40
	batch_size=256
	test_size=int(np.floor(split*dataset_size))
	train_size=dataset_size - test_size

	train_set,test_set=random_split(dataset,[train_size,test_size])

	train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True)
	test_loader=DataLoader(test_set)
	
	model=Mymodel()
	model=model.to(device)
	criterion=nn.BCELoss(reduction='sum')
	optimizer= torch.optim.Adam(model.parameters(),lr=0.0005)

	for epoch in range(epochs):
		loss,train_accuracy=train(model,train_loader,criterion,optimizer,train_size,device)
		#loss=0
		accuracy=test(model,test_loader,test_size,device)
		#test_loss=train(model,test_loader,criterion,optimizer,test_size,device)
		print("Finished training {} epoch with loss:{} and train and test accuracy:{} & {}".format((epoch+1),loss,accuracy,train_accuracy))
		if epoch%10==0:
			model_folder_name = f'epoch_{epoch:04d}_loss_{accuracy:.8f}'
			if not os.path.exists(os.path.join(save_dir, model_folder_name)):
				os.makedirs(os.path.join(save_dir, model_folder_name))
			torch.save(model.state_dict(), os.path.join(save_dir, model_folder_name, 'weights.pth'))
			print(f'model saved to {os.path.join(save_dir, model_folder_name, "weights.pth")}\n')
from __future__ import print_function, division
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from complete_dataset import training_toolset
import argparse

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import torch.optim as optim
import math

from torchvision import datasets, models, transforms
from matplotlib import pyplot as plt
import os, time

from torchsummary import summary


#torch.set_default_tensor_type('torch.DoubleTensor')

def arg():
	parser = argparse.ArgumentParser(description='save data directory')

	parser.add_argument('--csv_file', dest = 'csv_file', type=str,
					default="/Users/evnw/Research/DeepFasion/attri_predict/landmarks_csv/landmarks.csv")

	parser.add_argument('--img_dir', dest = 'img_folder', type=str,
					default="/Users/evnw/Research/DeepFasion/attri_predict")

	args = parser.parse_args()
	return args

class VGG19(nn.Module):

	def __init__(self, num_classes, init_weights):
		super(VGG19, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),

			nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),

			nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),

			nn.Conv2d(256, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),

			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2)
			)
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, num_classes),
		)
		if init_weights:
			self._initialize_weights()

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

class VGG16(nn.Module):

	def __init__(self, num_classes, init_weights):
		super(VGG16, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),

			nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),

			nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),

			nn.Conv2d(256, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),

			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2)
			)
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, num_classes),
		)
		if init_weights:
			self._initialize_weights()

	def forward(self, x):
		x = self.features(x)
		print(x.size())
		x = x.view(x.size(0), -1)
		print(x.size())
		x = self.classifier(x)
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

class fasNet(nn.Module):

	def __init__(self, num_classes, init_weights):
		super(fasNet, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),

			nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),

			nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),

			nn.Conv2d(256, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			)
		self.landmarks_conv = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True)
		)
		self.landmarks_fc = nn.Sequential(
			nn.Linear(512 * 14 * 14, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 16)
		)
		self.global_feature = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2)
		)
		self.local_map_fc = nn.Sequential(
			nn.Linear(4096*7*7, 4096),
			nn.ReLU(True),
			nn.Dropout()
		)
		self.local_map_fc_sm = nn.Sequential(
			nn.Linear(8*7*7, 4096),
			nn.ReLU(True),
			nn.Dropout()
		)

		self.global_map_fc = nn.Sequential(
			nn.Linear(512 * 14 * 14, 4096),
			nn.ReLU(True),
			nn.Dropout()
		)

		self.final_fc = nn.Sequential(
			nn.Linear(8192, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 1050),
			nn.Dropout()
		)

		if init_weights:
			self._initialize_weights()

	def forward(self, x):
		tm = time.time()
		feature = self.features(x)
		print('1', time.time() - tm)
		tm = time.time()
		#print("feature: {}".format(feature.size()))

		#self.show_feature_map(feature)

		landmarks_conv = self.landmarks_conv(feature)
		print('1', time.time() - tm)
		tm = time.time()
		#print("landmarks_conv", landmarks_conv.size())

		landmarks_conv = landmarks_conv.view(landmarks_conv.size(0), -1)
		print('2', time.time() - tm)
		tm = time.time()
		#print("landmarks_conv_view", landmarks_conv.size())

		landmarks = self.landmarks_fc(landmarks_conv)
		print('3', time.time() - tm)
		tm = time.time()
		#print("landmarks", landmarks)

		
		local_feature = self.get_local_feature(landmarks, feature)
		print('4', time.time() - tm)
		tm = time.time()

		'''
		local_feature = local_feature.view(local_feature.size(0), -1)
		print('5', time.time() - tm)
		tm = time.time()
		#print("local_feature_view", local_feature.size())
		local_feature_fc = self.local_map_fc(local_feature)
		#print("local_feature_fc", local_feature_fc.size())
		print('5', time.time() - tm)
		tm = time.time()
		'''
		local_feature = self.get_local_feature_sm(local_feature)

		local_feature = local_feature.view(local_feature.size(0), -1)
		#print("local_feature_view", local_feature.size())
		local_feature_fc = self.local_map_fc_sm(local_feature)
		#print("local_feature_fc", local_feature_fc.size())
		print('5', time.time() - tm)
		tm = time.time()
		

		global_feature = feature.view(feature.size(0), -1)
		#print("global_feature_view", global_feature.size())
		global_feature_fc = self.global_map_fc(global_feature)
		#print("global_feature_fc", global_feature_fc.size())
		print('6', time.time() - tm)
		tm = time.time()

		combined_feature = torch.cat((local_feature_fc, global_feature_fc), 1)
		#print("combined_feature", combined_feature.size())
		print('7', time.time() - tm)
		tm = time.time()

		category_attributes = self.final_fc(combined_feature)
		#print("category_attributes", category_attributes.size())
		print('8', time.time() - tm)
		tm = time.time()

		sigmoid = nn.Sigmoid()
		category_attributes = sigmoid(category_attributes)
		landmarks = sigmoid(landmarks)

		return landmarks, feature, category_attributes

	def get_local_feature_sm(self, local_feature):
		feature_map = []
		for batch in range(local_feature.size(0)):
			current_feature = local_feature[batch]
			current = []
			for i in range(8):
				temp_map = current_feature[i*512, :, :]
				for j in range(1, 512):
					idx = i*512+j
					temp_map = torch.add(temp_map, current_feature[idx, :, :])
				current.append(temp_map)
			current_tensor = torch.stack((current[0], current[1], current[2], current[3], current[4], current[5], current[6], current[7]))
			feature_map.append(current_tensor)
		feature_map_tuple = tuple(feature_map)
		local_feature_map = torch.stack(feature_map_tuple)
		return local_feature_map



	def get_local_feature(self, landmarks, feature):

		local_list = []
		for batch in range(landmarks.size(0)):
			current = []
			for i in range(8):
				landmark_x = int(landmarks[batch][2*i]*14)
				landmark_y = int(landmarks[batch][2*i+1]*14)
				if landmark_x < 3:
					start_x = 0
				elif landmark_x + 3 >= 14:
					start_x = 7
				else:
					start_x = landmark_x - 3

				if landmark_y < 3:
					start_y = 0
				elif landmark_y + 3 >= 14:
					start_y = 7
				else:
					start_y = landmark_y - 3

				local_feature = feature[batch, :, start_x: start_x+7, start_y: start_y+7]
				current.append(local_feature)

			current_tensor = torch.cat((current[0], current[1], current[2], current[3], current[4], current[5], current[6], current[7]), 0)
			local_list.append(current_tensor)

		local_tuple = tuple(local_list)
		local_feature_map = torch.stack(local_tuple)
		
		return local_feature_map


	def show_feature_map(self, feature):
		num = feature.size(0)
		for i in range(num):
			current_feature = feature[i]
			current_feature_map = feature[i, 0, :, :]
			for j in range(1, feature[i].size(0)):
				current_feature_map = torch.add(current_feature_map, feature[i,j,:,:])
			ax = plt.subplot(1, num, i + 1)
			plt.tight_layout()
			ax.set_title('Sample #{}'.format(i))
			ax.axis('off')
			current_feature_map_np = current_feature_map.detach().numpy()
			plt.imshow(current_feature_map_np)
		plt.show()


	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()


if __name__ == '__main__':
	args = arg()
	training_tool = training_toolset()
	#dataset, dataset_arr = training_tool.initialize_dataset()
	dataset = training_tool.initialize_dataset()
	#training_tool.show_random_sample(dataset_arr, 4)

	batch_size = 2

	trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
											 shuffle=True, num_workers=4)

	net = fasNet(num_classes = 16, init_weights = False)
	#net = VGG16(num_classes = 16, init_weights = False)
	#net = net.float()
	#summary(net, (3, 224, 224))

	#criterion = nn.L1Loss()
	regression_loss = nn.MSELoss()
	#softmax_loss = nn.sof
	BCE_loss = nn.BCELoss()
	CE_loss = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.8)

	count = 0
	epochs = 1
	times = 10000

	for epoch in range(epochs):  # loop over the dataset multiple times
		running_loss = 0.0
		start_time = time.time()
		for i, data in enumerate(trainloader):
			#print('sample', time.time() - start_time)
			start_time = time.time()
			count+=1
			print(count)

			inputs, landmarks, visibility, attributes, category = data['image'], data['landmarks'], data['visibility'], data['attributes'], data['category']
			inputs = inputs.float()
			landmarks = landmarks.float()
			visibility = visibility.float()
			attributes = attributes.type(torch.FloatTensor)
			category = category.type(torch.LongTensor)

			#print('input',time.time() - start_time)
			start_time = time.time()

			optimizer.zero_grad()
			landmarks_out, feature, category_attributes = net(inputs)
			category_out = category_attributes[:, 0:50]
			attributes_out = category_attributes[:, 50:]
			#category_out = category_out.type(torch.LongTensor)
			#attributes_out = attributes_out.type(torch.LongTensor)

			print('forward',time.time() - start_time)
			start_time = time.time()
			'''
			for i in range(batch_size):
				for j in range(8):								# number of total landmarks
					if visibility[i][j] == 0:
						landmarks[i][j][0] = outputs[i][2*j]
						landmarks[i][j][1] = outputs[i][2*j+1]
			'''

			labels = torch.rand(batch_size, 16, requires_grad = False)
			for i in range(batch_size):
				for j in range(8):
					labels[i][2*j] = landmarks[i][j][0]
					labels[i][2*j+1] = landmarks[i][j][1]
			#print('label', time.time() - start_time)
			start_time = time.time()

			landmarks_loss = regression_loss(landmarks_out, labels)
			category_loss = CE_loss(category_out, torch.max(category,1)[1])
			attributes_loss = BCE_loss(attributes_out, attributes)

			if count%2 == 0:
				local_wght = 0.2
				global_wght = 0.8
			else:
				local_wght = 0.8
				global_wght = 0.2

			loss = landmarks_loss*local_wght + (category_loss+attributes_loss)*global_wght


			#print('loss',time.time() - start_time)
			start_time = time.time()

			loss.backward()

			print('back',time.time() - start_time)
			start_time = time.time()

			optimizer.step()

			#print('step',time.time() - start_time)
			start_time = time.time()

			running_loss += loss.item()
			#print('running_loss')
			if count % times == times-1:	# print every 2000 mini-batches
				print('[%d, %d] loss: %.3f' %
					  (epoch + 1, count + 1, running_loss / 200))
				running_loss = 0.0

			if count%100 == 0:
				torch.save(net.state_dict(), 'fasNet_1_{}.pt'.format(count))
				print('{}saved'.format(count))

			if count == times:
				break
		if count == times:
			break

from __future__ import print_function, division
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from landmarks_dataset import training_toolset
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

'''class VGG19(nn.Module):

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
				m.bias.data.zero_()'''

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


if __name__ == '__main__':
	args = arg()
	training_tool = training_toolset()
	dataset, dataset_arr = training_tool.initialize_dataset()
	#training_tool.show_random_sample(dataset_arr, 4)

	batch_size = 2

	trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
											 shuffle=True, num_workers=4)

	net = VGG19(num_classes = 16, init_weights = False)
	#net = net.float()
	#summary(net, (3, 224, 224))

	#criterion = nn.L1Loss()
	criterion = nn.MSELoss()
	#criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.8)

	count = 0
	epochs = 1
	times = 10000

	for epoch in range(epochs):  # loop over the dataset multiple times
		running_loss = 0.0
		start_time = time.time()
		for i, data in enumerate(trainloader):
			print('sample', time.time() - start_time)
			start_time = time.time()
			count+=1
			print(count)

			inputs, landmarks, visibility = data['image'], data['landmarks'], data['visibility']
			inputs = inputs.float()
			landmarks = landmarks.float()
			visibility = visibility.float()

			print('input',time.time() - start_time)
			start_time = time.time()

			#inputs = inputs.double()
			#landmarks = landmarks.double()
			#visibility = visibility.double()

			optimizer.zero_grad()
			outputs = net(inputs)

			print('forward',time.time() - start_time)
			start_time = time.time()
			
			for i in range(batch_size):
				for j in range(8):                                # number of total landmarks
					if visibility[i][j] == 0:
						landmarks[i][j][0] = outputs[i][2*j]
						landmarks[i][j][1] = outputs[i][2*j+1]

			'''
			print(outputs[0])
			print(landmarks[0])
			print(outputs[0].type())
			print(landmarks[0].type())
			'''

			labels = torch.rand(batch_size, 16, requires_grad = False)
			for i in range(batch_size):
				for j in range(8):
					labels[i][2*j] = landmarks[i][j][0]
					labels[i][2*j+1] = landmarks[i][j][1]

			print('label', time.time() - start_time)
			start_time = time.time()

			loss = criterion(outputs, labels)

			print('loss',time.time() - start_time)
			start_time = time.time()

			loss.backward()

			print('back',time.time() - start_time)
			start_time = time.time()

			optimizer.step()

			print('step',time.time() - start_time)
			start_time = time.time()

			running_loss += loss.item()
			print('running_loss')
			if count % times == times-1:	# print every 2000 mini-batches
				print('[%d, %d] loss: %.3f' %
					  (epoch + 1, count + 1, running_loss / 200))
				running_loss = 0.0
			print('print')

			if count == times:
				break
		if count == times:
			break

	torch.save(net.state_dict(), 'vgg19_lanmarks_1_10000.pt')
	print('saved')

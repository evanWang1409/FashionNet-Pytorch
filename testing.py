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

torch.set_default_tensor_type('torch.DoubleTensor')

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

def imshow(img, landmark):
    #img = img / 2 + 0.5     # unnormalize
    #img = img 
    landmark = landmark.detach().numpy()
    landmarks = []
    for i in range(0, len(landmark), 2):
    	landmarks.append([landmark[i], landmark[i+1]])
    landmarks = np.array(landmarks)
    landmarks = landmarks*100
    print(landmarks)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    for i in range(len(landmarks)):
    	plt.scatter(landmarks[i][0], landmarks[i][1], s=10, marker='.', c='r')


if __name__ == '__main__':
	args = arg()
	training_tool = training_toolset()
	dataset, dataset_arr = training_tool.initialize_dataset()
	#training_tool.show_random_sample(dataset_arr, 4)

	batch_size = 1

	testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
											 shuffle=True, num_workers=batch_size)

	net = VGG16(num_classes = 16, init_weights = True)
	net.load_state_dict(torch.load('vgg19_lanmarks_1_2000.pt'))

	dataiter = iter(testloader)
	sample = dataiter.next()
	image, landmark, visibility = sample['image'], sample['landmarks'], sample['visibility']
	output = net(image)
	imshow(image[0], output[0])
	plt.show()

















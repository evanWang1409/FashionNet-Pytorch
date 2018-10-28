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

def arg():
	parser = argparse.ArgumentParser(description='save data directory')

	parser.add_argument('--csv_file', dest = 'csv_file', type=str,
                    default="/Users/evnw/Research/DeepFasion/attri_predict/landmarks_csv/landmarks.csv")

	parser.add_argument('--img_dir', dest = 'img_folder', type=str,
                    default="/Users/evnw/Research/DeepFasion/attri_predict")

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = arg()
	training_tool = training_toolset()
	dataset, dataset_arr = training_tool.initialize_dataset()
	training_tool.show_random_sample(dataset_arr, 4)
import cv2, re, csv, os
from matplotlib import pyplot as plt
import numpy as np
import argparse
txt_path = '/Users/evnw/Research/DeepFasion/attri_predict/Anno/list_category_cloth.txt'
csv_path = '/Users/evnw/Documents/Github/FashionNet-Pytorch/anno/list_category_cloth.csv'

if __name__ == '__main__':
	attr = []
	file = open(txt_path)
	csv_file = open(csv_path, 'w')
	writer = csv.writer(csv_file, delimiter = ',')

	num = int(file.readline())            # read total image number
	file.readline()
	for time in range(num):
		new_line = file.readline()
		new_line = new_line.split()
		writer.writerow(new_line)
		print(new_line[0])

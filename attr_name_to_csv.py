import cv2, re, csv, os
from matplotlib import pyplot as plt
import numpy as np
import argparse
txt_path = '/Users/evnw/Research/DeepFasion/attri_predict/Anno/list_attr_cloth.txt'
csv_path = '/Users/evnw/Documents/Github/FashionNet-Pytorch/anno/list_attr_cloth.csv'

if __name__ == '__main__':
	attr = []
	file = open(txt_path)
	csv_file = open(csv_path, 'w')
	writer = csv.writer(csv_file, delimiter = ',')

	num = int(file.readline())            # read total image number
	file.readline()
	for time in range(num):
		temp = []
		new_line = file.readline()
		new_line_arr = new_line.split()
		new_line = re.sub(r'[0-9]|\n', '', new_line)
		temp.append(new_line)
		print(new_line)
		temp.append(new_line_arr[-1])
		writer.writerow(temp)

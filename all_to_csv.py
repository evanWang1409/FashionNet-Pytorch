import cv2, re, csv, os
from matplotlib import pyplot as plt
import numpy as np
import argparse




def read_attr(file_dir):
	landmarks_path = os.path.join(file_dir, 'list_landmarks.txt')
	attr_path = os.path.join(file_dir, 'list_attr_img.txt')
	cat_path = os.path.join(file_dir, 'list_category_img.txt')

	landmarks_file = open(landmarks_path)
	num = int(landmarks_file.readline())            # read total image number
	landmarks_file.readline()                       # read attributes instructions

	attr_file = open(attr_path)
	cat_file = open(cat_path)

	attr_file.readline();attr_file.readline()
	cat_file.readline();cat_file.readline()

	img_landmarks = []
	img_cat = []
	img_attr = []

	for time in range(num):
		landmarks_line = landmarks_file.readline()
		landmarks_line = landmarks_line.split()
		cat_line = cat_file.readline()
		cat_line = cat_line.split()
		attr_line = attr_file.readline()
		attr_line = attr_line.split()

		img_name = landmarks_line[0]
		cloth_type = landmarks_line[1]
		cat = int(cat_line[-1])
		attr = np.array(attr_line[1:], dtype = np.float32)

		landmarks = []

		if (len(landmarks_line)-2)%3 != 0:
			raise Exception("number of attributes not satisfied")

		for i in range(int((len(landmarks_line)-2)/3)):
			landmarks.append([landmarks_line[3*i+2],landmarks_line[3*i+3],landmarks_line[3*i+4]])

		landmarks = np.array(landmarks, dtype = np.float32)

		img_landmarks.append([img_name, cloth_type, landmarks])
		img_cat.append(cat)
		img_attr.append(attr)

	return img_landmarks, img_cat, img_attr



def save_train_csv(img_landmarks, img_cat, img_attr, csv_dir):
	#sorted(img_landmarks, key=lambda img_landmark: len(img_landmark[2]))

	landmarks_csv_path = csv_dir + '/' + "landmarks.csv"
	cat_csv_path = os.path.join(csv_dir, 'cat.csv')
	attr_csv_path = os.path.join(csv_dir, 'attr.csv')

	landmarks_file = open(landmarks_csv_path, 'w', newline='')
	landmarks_writer = csv.writer(landmarks_file, delimiter=',')

	cat_file = open(cat_csv_path, 'w', newline='')
	cat_writer = csv.writer(cat_file, delimiter=',')

	attr_file = open(attr_csv_path, 'w', newline='')
	attr_writer = csv.writer(attr_file, delimiter=',')


	for i in range(len(img_landmarks)):
		print(i)
		land_row = img_landmarks[i]
		attr_row = img_attr[i]
		cat = img_cat[i]

		img_path = land_row[0]

		land_temp = []
		attr_temp = []
		cat_temp = []

		land_temp.append(img_path)
		attr_temp.append(img_path)
		cat_temp.append(img_path)

		for landmark in land_row[2]:
			land_temp.append(-landmark[0]+1)
			land_temp.append(landmark[1])
			land_temp.append(landmark[2])

		while(len(land_temp) < 25):
			land_temp.append(0)

		for val in attr_row:
			val = int((val+1)/2)
			attr_temp.append(val)

		cat_temp.append(cat)
		
		landmarks_writer.writerow(land_temp)
		cat_writer.writerow(cat_temp)
		attr_writer.writerow(attr_temp)


def arg():
	parser = argparse.ArgumentParser(description='save data directory')
	parser.add_argument('--file_dir', dest = 'file_dir', type=str,
                    default='/Users/evnw/Research/DeepFasion/attri_predict/Anno')
	parser.add_argument('--img_folder', dest = 'img_folder', type=str,
                    default='/Users/evnw/Research/DeepFasion/attri_predict')
	parser.add_argument('--csv_dir', dest = 'csv_dir', type=str,
                    default='/Users/evnw/Documents/Github/FashionNet-Pytorch/anno')
	args = parser.parse_args()
	return args



if __name__ == '__main__':
	args = arg()
	img_landmarks, img_cat, img_attr = read_attr(args.file_dir)
	img_folder = args.img_folder
	#display_landmarks(img_landmarks, img_folder)
	csv_dir = args.csv_dir
	save_train_csv(img_landmarks, img_cat, img_attr, csv_dir)

import cv2, re, csv
from matplotlib import pyplot as plt
import numpy as np




def read_landmarks(file):
	num = int(file.readline())            # read total image number
	file.readline()                       # read attributes instructions

	img_landmarks = []

	for time in range(num):
		new_line = file.readline()
		new_line = new_line.split()
		img_name = new_line[0]
		cloth_type = new_line[1]
		landmarks = []

		if (len(new_line)-2)%3 != 0:
			raise Exception("number of attributes not satisfied")

		for i in range(int((len(new_line)-2)/3)):
			landmarks.append([new_line[3*i+2],new_line[3*i+3],new_line[3*i+4]])

		landmarks = np.array(landmarks, dtype = np.float32)

		img_landmarks.append([img_name, cloth_type, landmarks])

	return img_landmarks



def display_landmarks(img_landmarks, img_folder):
	for i in range(len(img_landmarks)):
		img_path = img_folder + '/' + img_landmarks[i][0]
		img = cv2.imread(img_path)
		print(img.shape)
		plt.imshow(img)
		plt.scatter(img_landmarks[i][2][:, 1], img_landmarks[i][2][:, 2])
		plt.show()
	return



def save_train_csv(img_landmarks, csv_dir):
	sorted(img_landmarks, key=lambda img_landmark: len(img_landmark[2]))
	img_landmarks_flat = []
	csv_path = csv_dir + '/' + "landmarks.csv"
	with open(csv_path, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		for row in img_landmarks:
			temp = []
			temp.append(row[0])
			for landmark in row[2]:
				temp.append(landmark[1])
				temp.append(landmark[2])

			while(len(temp) < 17):
				temp.append(0)
			writer.writerow(temp)



if __name__ == '__main__':
	file_path = '/Users/evnw/Research/DeepFasion/attri_predict/Anno/list_landmarks.txt'
	file = open(file_path)
	img_landmarks = read_landmarks(file)
	img_folder = '/Users/evnw/Research/DeepFasion/attri_predict'
	#display_landmarks(img_landmarks, img_folder)
	csv_dir = '/Users/evnw/Research/DeepFasion/attri_predict/landmarks_csv'
	save_train_csv(img_landmarks, csv_dir)

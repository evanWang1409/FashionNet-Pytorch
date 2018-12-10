import os, csv, shutil

men_cat = {3:'jackets-coats',11:'jackets-coats', 4:'jackets-coats',
			26:'jeans', 
			34:'pants', 22:'pants',
			7:'shirts', 
			32:'shorts', 
			2:'suits-blazers',
			16:'sweaters', 10:'sweaters'}

women_cat = {49:'dresses', 50:'dresses', 45:'dresses', 41:'dresses', 
			3:'jackets-coats',11:'jackets-coats', 
			26:'jeans', 
			42:'jumpsuits',
			34:'pants', 22:'pants',
			32:'shorts', 
			33:'skirts', 
			16:'sweaters', 10:'sweaters', 
			19:'tops'}

csv_path = '/users/evnw/documents/github/fashionnet-pytorch/anno/cat.csv'
clothes_root = '/users/evnw/research/deepfasion/attri_predict'
target_root = '/users/evnw/documents/github/fashionnet-pytorch/resnet/dataset_resnet/train'

if __name__ == '__main__':
	csv_file = open(csv_path, 'r')
	reader = csv.reader(csv_file, delimiter = ',')
	count = 10000
	for row in reader:
		img = row[0]
		cat = int(row[1])
		old_path = os.path.join(clothes_root, img)
		count+=1
		new_img_name = '{}.jpg'.format(count)
		if cat in men_cat.keys():
			men_category = men_cat[cat]
			men_dir = os.path.join(target_root, 'men')
			men_dir = os.path.join(men_dir, men_category)
			new_men_img_path = os.path.join(men_dir, new_img_name)
			shutil.copy2(old_path, new_men_img_path)
			print(img, new_men_img_path)

		if cat in women_cat.keys():
			women_category = women_cat[cat]
			women_dir = os.path.join(target_root, 'women')
			women_dir = os.path.join(women_dir, women_category)
			new_women_img_path = os.path.join(women_dir, new_img_name)
			shutil.copy2(old_path, new_women_img_path)
			print(img, new_women_img_path)

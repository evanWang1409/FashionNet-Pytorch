from fas_resnet_pred import men_pred, women_pred
import numpy as np
from matplotlib import pyplot as plt
import csv, os

if __name__ == '__main__':
	'''
	clothes_path = '/Users/evnw/documents/github/fashionnet-pytorch/resnet/test_imgs/test2.jpg'
	women_pred(clothes_path)
	
	clothes_path = '/Users/evnw/documents/github/fashionnet-pytorch/resnet/test_imgs/test6.jpg'
	men_pred(clothes_path)
	'''
	men_conf = [[79.52105698,  0.90834021,  0.16515277,  0.16515277,  2.89017341,  5.78034682, 10.56977704],
				[ 6.51340996, 82.75862069,  3.44827586,  0,  1.91570881,  1.14942529, 4.21455939],
				[ 3.04878049,  7.31707317, 78.04878049,  0,  6.09756098,  0, 5.48780488],
				[25.80645161,  0,  0, 64.51612903,  0,  0, 9.67741935],
				[ 9.3373494,   1.95783133,  1.95783133,  0, 80.87349398,  0.90361446, 4.96987952],
				[32.01581028,  0.79051383,  0,  0,  1.58102767, 60.07905138, 5.53359684],
				[14.38016529,  0.66115702,  0,  0,  1.48760331,  1.81818182, 81.65289256]]

	women_conf = [[93.47377117, 3.80008261, 0, 0.33044197, 0,  0.16522098, 1.28046262, 0.61957869, 0.33044197],
	[ 5.91666667, 82.5, 0.91666667, 0.16666667, 0.5, 1.91666667, 2., 3.83333333, 2.25],
	[ 0.3875969,   5.81395349, 86.04651163, 0.7751938, 2.71317829,  1.9379845, 0.7751938,   1.1627907,   0.3875969 ],
	[30.73170732,  3.41463415,  0.48780488, 62.43902439,  0,  2.92682927, 0,  0,  0],
	[ 0.67567568,  4.72972973, 18.24324324,  1.35135135, 63.51351351,  8.10810811, 0,  2.7027027,   0.67567568],
	[ 1.20845921,  8.6102719,   2.41691843,  0.3021148,   0.90634441, 77.34138973, 5.13595166,  1.35951662,  2.71903323],
	[13.04347826,  6.91699605,  0.19762846,  0,  0.19762846,  3.55731225, 72.92490119,  1.77865613,  1.38339921],
	[ 3.63636364, 24.46280992,  0.66115702,  0,  0,  1.15702479, 0.99173554, 66.11570248,  2.97520661],
	[ 9.48509485, 44.44444444,  1.89701897,  0.27100271,  0.27100271,  5.4200542, 5.69105691,  8.1300813,  24.3902439 ]]

	men_classes = ['jackets-coats', 'jeans', 'pants', 'shirts', 'shorts', 'suits-blazers', 'sweaters']
	women_classes = ['dresses', 'jackets-coats', 'jeans', 'jumpsuits', 'pants', 'shorts', 'skirts', 'sweaters', 'tops']

	root = '/Users/evnw/documents/github/fashionnet-pytorch/resnet'
	men_loss_path = os.path.join(root, 'fas_resnet_men_101.csv')
	women_loss_path = os.path.join(root, 'fas_resnet_women_101.csv')

	csv_file = open(men_loss_path, 'r')
	reader = csv.reader(csv_file, delimiter = ',')
	men_loss_data = []
	iters = []
	for row in reader:
		if len(row) < 2:
			break
		men_loss_data.append(float(row[1]))
		iters.append(int(row[0]))
	csv_file = open(women_loss_path, 'r')
	reader = csv.reader(csv_file, delimiter = ',')
	women_loss_data = []
	for row in reader:
		if len(row) < 2:
			break
		women_loss_data.append(float(row[1]))
	plt_men, = plt.plot(iters, men_loss_data, color = 'blue')
	plt_women, = plt.plot(iters, women_loss_data, color = 'green')
	plt.legend([plt_men, plt_women], ['Men Clothes', 'Women Clothes'])
	plt.title('Running loss along training')
	plt.xlabel('Iters')
	plt.ylabel('loss')
	plt.show()

	'''
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(men_conf, interpolation='nearest')
	fig.colorbar(cax)
	ax.set_xticklabels(['']+men_classes)
	ax.set_yticklabels(['']+men_classes)
	fig.suptitle('Confusion matrix for men clothes (%)', fontsize=20)
	plt.xlabel('Predicted', fontsize=18)
	plt.ylabel('Ground Truth', fontsize=18)
	plt.show()


	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(women_conf, interpolation='nearest')
	fig.colorbar(cax)
	ax.set_xticklabels(['']+women_classes, fontsize=8)
	ax.set_yticklabels(['']+women_classes, fontsize=8)
	fig.suptitle('Confusion matrix for women clothes (%)', fontsize=20)
	plt.xlabel('Predicted', fontsize=18)
	plt.ylabel('Ground Truth', fontsize=18)
	plt.show()
	'''
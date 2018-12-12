import os

train_dir = '/home/zw119/floordog/dataset_resnet/train/men'

sum = 0
for subdir in os.listdir(train_dir):
	cat_dir = os.path.join(train_dir, subdir)
	if not os.path.isdir(cat_dir):
		continue
	print('{}: {}'.format(subdir, len(os.listdir(cat_dir))))
	sum+=len(os.listdir(cat_dir))
print(sum)
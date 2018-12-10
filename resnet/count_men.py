import os

train_dir = '/Users/evnw/research/deepfasion/inshop/img/men'

sum = 0
for subdir in os.listdir(train_dir):
	cat_dir = os.path.join(train_dir, subdir)
	if not os.path.isdir(cat_dir):
		continue
	print(cat_dir)
	print(len(os.listdir(cat_dir)))
	sum+=len(os.listdir(cat_dir))
print(sum)
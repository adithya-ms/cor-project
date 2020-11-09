import pandas as pd
import csv

good_df = pd.read_csv('good_representations.csv')

dataset_X = []
dataset_Y = []

for rep in good_df.values :
	print (len(rep))
	dataset_X.append(rep[0:675])


	object_name = rep[-1].split('//')[1].split('_')[0]
	dataset_Y.append(object_name)


print (len(dataset_X), len(dataset_Y))


with open('good_dataset_X.csv', 'w') as file:
	wr = csv.writer(file)
	wr.writerows(dataset_X)

with open('good_dataset_Y.csv', 'w') as file :
	wr = csv.writer(file)

	for y in dataset_Y :
		wr.writerow([y])

good_df_x = pd.read_csv('good_dataset_X.csv', header=None)
print (good_df_x)

good_df_y = pd.read_csv('good_dataset_Y.csv', header=None)
print (good_df_y)



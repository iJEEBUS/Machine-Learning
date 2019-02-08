import pandas as pd 

with open('./data/training.txt', 'r') as data_file:
	data = data_file.readlines()

df = pd.DataFrame(data, columns=['A'])

class_dict = {}

for index, row in df.iterrows():
	classification = row['A'].split()[0].lower()
	if classification not in class_dict.keys():
		class_dict[classification] = 1
	else:
		class_dict[classification] += 1
	n = index
	
n = 11293
atheism_probability = 480 / n
graphics_probability = 584 / n
MSwindows_probability = 572 / n
pc_probability = 590 / n
mac_probability = 578 / n
xwindows_probability = 593 / n
forsale_probability = 585 / n
autos_probability = 594 / n
motorcycles_probability = 598 / n
baseball_probability = 597 / n
hockey_probability = 600 / n
cryptology_probability = 595 / n
electronics_probability = 591 / n
medicine_probability = 594 / n
space_probability = 593 / n
christianity_probability = 598 / n
guns_probability = 545 / n
mideastpolitics_probability = 564 / n
politics_probability = 465 / n
religion_probability = 377 / n

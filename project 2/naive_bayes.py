import pandas as pd 

def add_post(key, post):
	if key not in text.keys():
		text[key] = post
	else:
		text[key] += post


# probability estimates of each class
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

P = {
	atheism_probability,
	graphics_probability,
	MSwindows_probability,
	pc_probability,
	mac_probability,
	xwindows_probability,
	forsale_probability,
	autos_probability,
	motorcycles_probability,
	baseball_probability,
	hockey_probability,
	cryptology_probability,
	electronics_probability,
	medicine_probability,
	space_probability,
	guns_probability,
	mideastpolitics_probability,
	politics_probability,
	religion_probability,
	}

with open('./data/training.txt', 'r') as data_file:
	data = data_file.readlines()

print(1)
docs = pd.DataFrame(data, columns=['A']) # training docs

# Learn 
# 1. Collect all words occurring in the Sample documents
vocabulary = set() # set of all distinct words
whole_posts = []
post = []
text = {} # single document per class (concate all docs for each class)

# creation of vocabulary set
# creation of all the posts as strings
for index, row in docs.iterrows():
	
	post = row['A'].split()[1:]
	post = " " + " ".join(post) # add leading space -> easier use when adding to text doc
	words = row['A'].split()
	whole_posts.append(post)

	for word in words:
		vocabulary.add(word) # do not need to check for dups -> it's a set

# populate the text dictionary like below
# text = {
#		  'atheism': 'all of the posts concated...',
#  		  'graphics': 'all of the posts...', 
#		 }
for i in range(n):
	if i < 480:
		add_post('atheism', whole_posts[i])
	
	elif i < 1064:
		add_post('graphics', whole_posts[i])
	
	elif i < 1636:
		add_post('MSwindows', whole_posts[i])
	
	elif i < 2226:
		add_post('pc', whole_posts[i])
		
	elif i < 2804:
		add_post('mac', whole_posts[i])
	
	elif i < 3397:
		add_post('xwindows', whole_posts[i])
		
	elif i < 3982:
		add_post('forsale', whole_posts[i])
	
	elif i < 4576:
		add_post('autos', whole_posts[i])
	
	elif i < 5174:
		add_post('motorcycles', whole_posts[i])
	
	elif i < 5771:
		add_post('baseball', whole_posts[i])
	
	elif i < 6371:
		add_post('hockey', whole_posts[i])
	
	elif i < 6966:
		add_post('cryptology', whole_posts[i])
	
	elif i < 7557:
		add_post('electronics', whole_posts[i])
	
	elif i < 8151:
		add_post('medicine', whole_posts[i])

	elif i < 8744:
		add_post('space', whole_posts[i])

	elif i < 9342:
		add_post('christianity', whole_posts[i])
	
	elif i < 9887:
		add_post('guns', whole_posts[i])

	elif i < 10451:
		add_post('mideast_politics', whole_posts[i])

	elif i < 10916:
		add_post('politics', whole_posts[i])

	elif i < 11293:
		add_post('religion', whole_posts[i])

word_counts = {}
# get total number of words in each text
for key in text.keys():
	word_counts[key] = len(text[key].split())

print(2)
# create a dict for the number of times a word occurs in a piece of text
# text = { 
#			'word' : { 'politics' : num 
#						'space' : num},
#
#		    'word2' : {'politics': num ,
#						'space': num }
#		 }
num_occurences = {}
count = 0
for word in vocabulary:
	for key in text.keys():
		num_occurences[word] = {}
		count += 1
		print(count)
		num_occurences[word][key] = {text[key].count(word)}
import pandas as pd
import csv

# loading training data into data frame
all_data = pd.read_csv("./data/training.txt", sep="\n")
#all_data.head()

'''
1.0 Collect all words occurring in the sample documents
'''
# collect all distinct words
vocab = set()
num_rows = len(all_data.index)

for row in range(num_rows):
    for post in all_data.iloc[row]:
        split_post = post.split()[1:]
        for word in split_post:
            vocab.add(word)
            
'''
2.0 Create a document dictionary where key is a class and values are all posts in that class
'''
docs = {}

for row in range(num_rows):
    for post in all_data.iloc[row]:
        classification = post.split()[0]
        only_content = ' '.join(post.split()[1:])
        
        if classification in docs.keys():
            docs[classification].append(only_content) 
        else:
            docs[classification] = [only_content]
            
'''
2.1 Generate the probability estimate of each class
'''
prob_estimates = {}

for classification in docs.keys():
    num_class_posts = len(docs[classification])
    prob_estimates[classification] = (num_class_posts / num_rows)
    
'''
2.2 Create a single document per class.
    Stored as tuple (n, text) where n is the number of word positions in text
'''
text = {}

for classification in docs.keys():
    combined_posts = ' '.join(docs[classification])
    num_word_positions = len(combined_posts.split())
    text[classification] = (num_word_positions, combined_posts)
    
'''
2.3 determine the number of times each unique word appears in each Text 

    #TODO - currently hardcoded in words and classes for testing
'''
word_occurrence_estimate = {}

for word in ['and']:
    num_occurences = 0
    for classification in ['religion']: #text.keys():
        num_occurences = text[classification][1].count(word)
        estimate = (num_occurences + 1) / (text[classification][0] + len(vocab))
        try:
            word_occurrence_estimate[word][classification] = estimate
        except:
            word_occurrence_estimate[word] = {}
            word_occurrence_estimate[word][classification] = estimate
            
#print(word_occurrence_estimate['and']['religion'])


'''
3.0 Classification - classify new data
'''
class_answer = ''


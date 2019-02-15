import pandas as pd
import csv
from math import log, exp

# loading training data into data frame
all_data = pd.read_csv("./data/training.txt", sep="\n")
all_data = all_data.sample(100)

'''
1.0 Collect all words occurring in the sample documents
'''
print("Collecting all distinct words...")
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
print("Creating document dictionary...")
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
print("Generating probability estimates for each classification...")
prob_estimates = {}

for classification in docs.keys():
    num_class_posts = len(docs[classification])
    prob_estimates[classification] = (num_class_posts / num_rows)
    
'''
2.2 Create a single document per class.
    Stored as tuple (n, text) where n is the number of word positions in text
'''
print("Merging all like documents into one...")
text = {}

for classification in docs.keys():
    combined_posts = ' '.join(docs[classification])
    num_word_positions = len(combined_posts.split())
    text[classification] = (num_word_positions, combined_posts)
    
'''
2.3 determine the number of times each unique word appears in each Text 
'''
print("Calculating word occurrence probabilities...")
word_occurrence_estimate = {}

count = 0
tot_words = len(vocab)

for word in vocab:
    count += 1
    num_occurences = 0
    for classification in text.keys():
        num_occurences = text[classification][1].count(word)
        estimate = (num_occurences + 1) / (text[classification][0] + len(vocab))
        try:
            word_occurrence_estimate[word][classification] = estimate
        except:
            word_occurrence_estimate[word] = {}
            word_occurrence_estimate[word][classification] = estimate
    print("%s / %s words processed" % (count, tot_words))


'''
3.0 Classification - classify new data
# TODO need to get this working
'''
print("Classifying new data...")
total_correct_classifications = 0

#new_data = pd.read_csv('./data/training.txt', sep='\n')
#num_rows = len(new_data.index) // 10

for row in range(num_rows):

    for post in all_data.iloc[row]:
        class_probabilities = {}
        max_probability = ('', None) # (class, probability)
        probability = 0.0
        correct_classification = post.split()[0]
        only_content = ' '.join(post.split()[1:])
        print(len(class_probabilities.keys()))
    
    for classification in text.keys():
        for word in only_content.split():
            if word in word_occurrence_estimate.keys():
                probability += log(word_occurrence_estimate[word][classification]) # use log to account for underflow
            else:
                prob = log(1) - log((len(text[classification]) + len(vocab)))
                probability += prob
        class_probabilities[classification] = probability
    print('\n')
    # set the max probability of each posts classification
    for classification in class_probabilities.keys():
        print(classification, class_probabilities[classification])
        if (max_probability[1] == None):
            max_probability = (classification, probability)
        elif (class_probabilities[classification] < max_probability[1]):
            #print("New highest prob")
            max_probability = (classification, probability)
            
    # counting correct assignments        
    if correct_classification == max_probability[0]:
        total_correct_classifications += 1
        
    print("%s  ::  %s" % (correct_classification, max_probability[0]))

accuracy = (total_correct_classifications / num_rows)
print("Accuracy: %s" % (accuracy))


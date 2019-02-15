import pandas as pd
import csv
from math import log, exp

stop_words = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

# loading training data into data frame
all_data = pd.read_csv("./data/training.txt", sep="\n")

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
            if word in stop_words:
                pass
            else:
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
'''
print("Classifying new data...")
total_correct_classifications = 0
lowest_num = 0
max_probability = None
new_data = pd.read_csv('./data/testing.txt', sep='\n')
num_rows = len(new_data.index)

for row in range(num_rows):

    for post in new_data.iloc[row]:
        class_probabilities = {}
        max_probability = ['', None] # (class, probability)
        probability = 0.0
        correct_classification = post.split()[0]
        only_content = ' '.join(post.split()[1:])
    
        # calculate probability that a word belongs to a certain class of post
        for classification in text.keys():
            max_probability = ['', None] # (class, probability)
            probability = 0.0
            for word in only_content.split():
                if word in word_occurrence_estimate.keys():
                    probability += log(word_occurrence_estimate[word][classification]) # use log to account for underflow
                else:
                    prob = log(1) - log((len(text[classification]) + len(vocab)))
                    probability += prob
            class_probabilities[classification] = probability
    
        # find the post with the highest probability
        for classification in text.keys():
            if (max_probability[1] == None):
                max_probability[0] = classification
                max_probability[1] = class_probabilities[classification]
            elif ((abs(class_probabilities[classification]) < abs(max_probability[1])) ):
                max_probability[0] = classification
                max_probability[1] = class_probabilities[classification]
                lowest_num = max_probability[1]
                
        # counting correct assignments        
        correct = (correct_classification == max_probability[0])
        if correct:
            total_correct_classifications += 1
            print(correct)
        else:
            print("%-10s :: %-10s :: %-10s" % (correct, correct_classification, max_probability[0]))
        

accuracy = (total_correct_classifications / num_rows)
print("Accuracy: %s" % (accuracy))
print(num_rows)
import pandas as pd
import csv
from math import log, exp
import datetime

# loading training data into data frame
all_data = pd.read_csv("./data/training.txt", sep="\n")
#all_data = all_data.head(1300)
#all_data = all_data.sample(1300)

'''
1.0 Collect all words occurring in the sample documents
'''
print("Loading and processing data...")
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
print(".")
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
print("..")
prob_estimates = {}

for classification in docs.keys():
    num_class_posts = len(docs[classification])
    prob_estimates[classification] = (num_class_posts / num_rows)
    
'''
2.2 Create a single document per class.
    Stored as tuple (n, text) where n is the number of word positions in text
'''
print("...")
text = {}

for classification in docs.keys():
    combined_posts = ' '.join(docs[classification])
    num_word_positions = len(combined_posts.split())
    text[classification] = (num_word_positions, combined_posts)
    
'''
2.3 determine the number of times each unique word appears in each Text 
'''
print("....")
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


'''
3.0 Classification - classify new data
'''
print("Classifying new data...")
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0
accuracy = 0


# Used to make confusion matrix
correct_answers = []
predicted_answers = []

lowest_num = 0
max_probability = None
new_data = pd.read_csv('./data/testing.txt', sep='\n')
#new_data = new_data.head(1000)
#new_data = new_data.sample(200)
num_rows = len(new_data.index)

for row in range(num_rows):

    for post in new_data.iloc[row]:
        class_probabilities = {}
        max_probability = ['', None] # (class, probability)
        probability = 0.0
        correct_classification = post.split()[0]
        correct_answers.append(correct_classification) # add correct answer to ans list
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
                
      
        class_guess = max_probability[0]
        predicted_answers.append(class_guess)
        
        # counting correct assignments  
        correct = (correct_classification == class_guess)
        if correct:
            true_positive += 1
           

'''
3.1 Scoring output with precision, recall, accuracy, F!, and misclassification
'''
# Accuracy
accuracy = (true_positive / num_rows)

# Misclassification
misclassification = (num_rows - true_positive) / (num_rows)
 
# Precision -  the fraction of events where we correctly declared C out of all instances where the algorithm declared C
# Recall - the fraction of events where we correctly declared C out of all of the cases where the true of state of the world is C
# F1 - combine precision and recall into one metric

# Create the confusion matrix
precision = 0.0
recall = 0.0

precision_list = []
recall_list = []

actual = pd.Series(correct_answers, name='Actual')
pred = pd.Series(predicted_answers, name='Predicted')
confusion_matrix = pd.crosstab(actual, pred)
normalized_confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)

for index, row in confusion_matrix.iterrows():
    tp = confusion_matrix[index][index]
    tp_and_fp = sum(row)
    tp_and_fn = sum(confusion_matrix[index])
    precision = tp / tp_and_fp
    recall = tp / tp_and_fn
    precision_list.append(precision)
    recall_list.append(recall)
precision = sum(precision_list) / len(precision_list)
recall = sum(recall_list) / len(recall_list)

F1 = 2 * ((precision * recall) / (precision + recall))
    
    
'''
3.2 Report statistics / Write the statistics to the output file
'''
with open('output.txt', 'a') as fout:
    fout.write("Method: %s\n" % ('Base Algorithm'))
    fout.write("Run at: %s\n" % (datetime.datetime.now()))
    fout.write("Accuracy: %s\n" % (accuracy))
    fout.write("Precision: %s\n" % (precision))
    fout.write("Recall: %s\n" % (recall))
    fout.write("F1: %s\n" % (F1))
    fout.write("Misclassification %s\n" % (misclassification))
    fout.write("---------------------------------------------\n")
    
print("Accuracy: %s" % (accuracy))
print("Misclassification: %s" % (misclassification))
print("Precision: %s" % (precision))
print("Recall %s" % (recall))
print("F1: %s" % (F1))
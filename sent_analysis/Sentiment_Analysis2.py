
# coding: utf-8

# In[1]:


import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import re
import ipywidgets as widgets
from IPython.display import display


# In[2]:


files_pos = open("positive.txt", "r", encoding="ISO-8859-1").read()
files_neg = open("negative.txt", "r",encoding="ISO-8859-1").read()


# In[3]:


len(files_neg)


# ### Preprocessing

# In[7]:


from nltk.corpus import stopwords
all_words = []
documents = []

for r in files_pos.split('\n'):
    documents.append((r, "pos"))
    
for r in files_neg.split('\n'):
    documents.append((r, "neg"))
    
short_pos_words = word_tokenize(files_pos)
short_neg_words = word_tokenize(files_neg)

for w in short_pos_words:
    all_words.append(w.lower())
for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
stop_words = list(set(stopwords.words('english')))



# In[12]:


len(all_words)


# In[15]:


# pickling the list documents to save future recalculations 

save_documents = open("pickled_algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()


# In[16]:


# creating a frequency distribution of each adjectives. 
BOW = nltk.FreqDist(all_words)
BOW


# In[17]:


# listing the 5000 most frequent words
word_features = list(BOW.keys())[:5000]
word_features[0], word_features[-1]


# In[18]:


save_word_features = open("pickled_algos/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


# In[19]:


# function to create a dictionary of features for each review in the list document.
# The keys are the words in word_features 
# The values of each key are either true or false for wether that feature appears in the review or not
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# Creating features for each review
featuresets = [(find_features(rev), category) for (rev, category) in documents]

# Shuffling the documents 
random.shuffle(featuresets)
print(len(featuresets))


# In[22]:


training_set = featuresets[:9000]
testing_set = featuresets[9000:]
print( 'training_set :', len(training_set), '\ntesting_set :', len(testing_set))


# In[23]:


classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(15)


# In[24]:


# Printing the most important features 

mif = classifier.most_informative_features()

mif = [a for a,b in mif]
print(mif)


# In[25]:


# getting predictions for the testing set by looping over each reviews featureset tuple
# The first element of the tuple is the feature set and the second element is the label 
ground_truth = [r[1] for r in testing_set]

preds = [classifier.classify(r[0]) for r in testing_set]


# In[26]:


from sklearn.metrics import f1_score
f1_score(ground_truth, preds, labels = ['neg', 'pos'], average = 'micro')


# In[27]:


import numpy as np

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

y_test = ground_truth
y_pred = preds
class_names = ['neg', 'pos']



def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# In[28]:


from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.metrics import accuracy_score


# Classifiers for an ensemble model: 
# Naive Bayes (NB)
# Multinomial NB
# Bernoulli NB
# Logistic Regression
# Stochastic Gradient Descent Classifier - SGD
# Support Vector Classification - SVC
# 

# In[1]:


print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_clf = SklearnClassifier(MultinomialNB())
MNB_clf.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_clf, testing_set))*100)

BNB_clf = SklearnClassifier(BernoulliNB())
BNB_clf.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BNB_clf, testing_set))*100)

LogReg_clf = SklearnClassifier(LogisticRegression())
LogReg_clf.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogReg_clf, testing_set))*100)

SGD_clf = SklearnClassifier(SGDClassifier())
SGD_clf.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGD_clf, testing_set))*100)

print("Finished training models")
# ### Storing all models using pickle 

# In[ ]:


def create_pickle(c, file_name): 
    save_classifier = open(file_name, 'wb')
    pickle.dump(c, save_classifier)
    save_classifier.close()
print("Finished pickling")
classifiers_dict = {'ONB': [classifier, 'pickled_algos/ONB_clf.pickle'],
                    'MNB': [MNB_clf, 'pickled_algos/MNB_clf.pickle'],
                    'BNB': [BNB_clf, 'pickled_algos/BNB_clf.pickle'],
                    'LogReg': [LogReg_clf, 'pickled_algos/LogReg_clf.pickle'],
                    'SGD': [SGD_clf, 'pickled_algos/SGD_clf.pickle']}




for clf, listy in classifiers_dict.items(): 
    create_pickle(listy[0], listy[1])


# In[ ]:


#
predictions = {}
# acc_scores = {}
# for clf, listy in classifiers_dict.items():
#     # getting predictions for the testing set by looping over each reviews featureset tuple
#     # The first elemnt of the tuple is the feature set and the second element is the label
#     acc_scores[clf] = accuracy_score(ground_truth, predictions[clf])
#     print(f'Accuracy_score {clf}: {acc_scores[clf]}')
#
#
# # In[ ]:
#
#
# from sklearn.metrics import f1_score
# ground_truth = [r[1] for r in testing_set]
#
# f1_scores = {}
# for clf, listy in classifiers_dict.items():
#     # getting predictions for the testing set by looping over each reviews featureset tuple
#     # The first elemnt of the tuple is the feature set and the second element is the label
#     predictions[clf] = [listy[0].classify(r[0]) for r in testing_set]
#     f1_scores[clf] = f1_score(ground_truth, predictions[clf], labels = ['neg', 'pos'], average = 'micro')
#     print(f'f1_score {clf}: {f1_scores[clf]}')

# # Ensemble Model

# In[ ]:

print("Before Classifier")
from nltk.classify import ClassifierI

# Defininig the ensemble model class 

class EnsembleClassifier(ClassifierI):
    
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    # returns the classification based on majority of votes
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    # a simple measurement the degree of confidence in the classification 
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


# #### Loading all models using pickle

# In[ ]:


# Load all classifiers from the pickled files 

# function to load models given filepath
def load_model(file_path): 
    classifier_f = open(file_path, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier


# Original Naive Bayes Classifier
ONB_Clf = load_model('pickled_algos/ONB_clf.pickle')

# Multinomial Naive Bayes Classifier 
MNB_Clf = load_model('pickled_algos/MNB_clf.pickle')


# Bernoulli  Naive Bayes Classifier 
BNB_Clf = load_model('pickled_algos/BNB_clf.pickle')

# Logistic Regression Classifier 
LogReg_Clf = load_model('pickled_algos/LogReg_clf.pickle')

# Stochastic Gradient Descent Classifier
SGD_Clf = load_model('pickled_algos/SGD_clf.pickle')
print("Finished loading models")


# Initializing the ensemble classifier 
ensemble_clf = EnsembleClassifier(ONB_Clf, MNB_Clf, BNB_Clf, LogReg_Clf, SGD_Clf)

# List of only feature dictionary from the featureset list of tuples 
feature_list = [f[0] for f in testing_set]

# Looping over each to classify each review
ensemble_preds = [ensemble_clf.classify(features) for features in feature_list]


# In[ ]:


f1_score(ground_truth, ensemble_preds, average = 'micro')


# # Live Sentiment Analysis
# 
# Using the sentiment function we can classify individual reviews. 

# In[ ]:


# Function to do classification a given review and return the label a
# and the amount of confidence in the classifications
def sentiment(text):
    feats = find_features(text)
    return ensemble_clf.classify(feats), ensemble_clf.confidence(feats)


# In[ ]:

print("Before text")
# sentiment analysis of reviews of captain marvel found on rotten tomatoes
text_a = '''He is a near perfect striker, imagine without him Dortmund would be losing. The next Lewandowski but possibly better'''
text_b = '''He's the greatest thing in football after Messi-Ronaldo. Friendship with Mbappé has ended.'''
text_c = '''Ighalo looked fantastic. He was holding off defenders and pulling down balls at will the last few minutes when he came on.'''
text_d = '''It is dumb against somebody who can consistently put the ball within 6 inches of the crossbar. Almost impossible for defenders to keep it out with their heads. But against an average kicker, this would probably be pretty successful.'''
text_e = '''If at the start of the season someone told me that United fans would be praying Liverpool wouldn’t advance in the CL because they’ve already accepted Liverpool have won the title, at a canter, I would have grasped at that with both hands. Icing on the cake that United have been quite this bad though.'''

print(sentiment(text_a), sentiment(text_b), sentiment(text_c), sentiment(text_d), sentiment(text_e))


# ## Random Forest

# In[ ]:

print("Before converting to Pandas.")
# converting the training set  into a pandas data frame
#
# from tqdm import tqdm_notebook as tqdm
# import time
# import pandas as pd
# df = pd.DataFrame([training_set[0][0]])
# for f in tqdm(training_set[1:]):
#     df = df.append([f[0]], ignore_index=True)
#
#
# # In[ ]:
#
#
# # converting the testing set  into a pandas data frame
# df_test = pd.DataFrame([testing_set[0][0]])
# for f in tqdm(testing_set[1:]):
#     df_test = df_test.append([f[0]], ignore_index=True)
# print("After converting to Pandas.")
#
# # In[ ]:
#
#
# df.tail()
#
#
# # In[ ]:
#
#
# from sklearn.ensemble import RandomForestClassifier
#
# clf = RandomForestClassifier(n_estimators=100, min_samples_split = 100, random_state=0)
# X = df
# y = [x[1] for x in training_set]
#
#
# # In[ ]:
#
#
# clf.fit(X, y )
#
#
#
# # In[ ]:
#
#
# X_test = df_test
# y_test = [x[1] for x in testing_set]
# clf.score(X, y)
#
#
# # In[ ]:
#
#
# df.columns
#
#
# # In[ ]:
#
#
# from sklearn.datasets import load_iris
# iris = load_iris()
#
# # Model (can also use single decision tree)
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators=10)
#
# # Train
# model.fit(iris.data, iris.target)
# # Extract single tree
# estimator = clf.estimators_[5]
#
# from sklearn.tree import export_graphviz
# # Export as dot file
# export_graphviz(estimator, out_file='tree.dot',
#                 feature_names = df.columns,
#                 class_names = ['neg', 'pos'],
#                 rounded = True, proportion = False,
#                 precision = 2, filled = True)
#
# # Convert to png using system command (requires Graphviz)
# from subprocess import call
# call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
#
# # Display in jupyter notebook
# from IPython.display import Image
# Image(filename = 'tree.png')
#
#
# # In[ ]:
#
#
# iris.target_names
#
#
# # In[ ]:
#
#
# clf.decision_path(X)


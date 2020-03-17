# import os
# files_pos = os.listdir('train/pos')
# files_pos = [open('train/pos/' + f, 'r').read() for f in files_pos]
# files_neg = os.listdir('train/neg')
# files_neg = [open('train/neg/' + f, 'r').read() for f in files_neg]

files_pos = open("positive.txt", "r").read()
files_neg = open("negative.txt", "r").read()


# Preprocessing

all_words = []
documents = []

import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import matplotlib as plt
import random

stop_words = list(set(stopwords.words('english')))

#  j is adject, r is adverb, and v is verb
# allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for p in files_pos:

    # create a list of tuples where the first element of each tuple is a review
    # the second element is the label
    documents.append((p, "pos"))

    # remove punctuations
    cleaned = re.sub(r'[^(a-zA-Z)\s]', '', p)

    # tokenize
    tokenized = word_tokenize(cleaned)

    # remove stopwords
    stopped = [w for w in tokenized if not w in stop_words]

    # parts of speech tagging for each word
    pos = nltk.pos_tag(stopped)

    # make a list of  all adjectives identified by the allowed word types list above
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in files_neg:
    # create a list of tuples where the first element of each tuple is a review
    # the second element is the label
    documents.append((p, "neg"))

    # remove punctuations
    cleaned = re.sub(r'[^(a-zA-Z)\s]', '', p)

    # tokenize
    tokenized = word_tokenize(cleaned)

    # remove stopwords
    stopped = [w for w in tokenized if not w in stop_words]

    # parts of speech tagging for each word
    neg = nltk.pos_tag(stopped)

    # make a list of  all adjectives identified by the allowed word types list above
    for w in neg:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

pos_A = []
for w in pos:
    if w[1][0] in allowed_word_types:
        pos_A.append(w[0].lower())
pos_N = []
for w in neg:
    if w[1][0] in allowed_word_types:
        pos_N.append(w[0].lower())

from wordcloud import WordCloud

text = ' '.join(pos_A)
wordcloud = WordCloud().generate(text)

plt.figure(figsize=(15, 9))
# Display the generated image:
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# pickling the list documents to save future recalculations

save_documents = open("pickled_algos/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

# creating a frequency distribution of each adjectives.
BOW = nltk.FreqDist(all_words)

# listing the 5000 most frequent words
word_features = list(BOW.keys())[:5000]
word_features[0], word_features[-1]

save_word_features = open("pickled_algos/word_features5k.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

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

training_set = featuresets[:20000]
testing_set = featuresets[20000:]
print('training_set :', len(training_set), '\ntesting_set :', len(testing_set))

classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)

classifier.show_most_informative_features(15)

# Printing the most important features

mif = classifier.most_informative_features()

mif = [a for a, b in mif]
print(mif)


# getting predictions for the testing set by looping over each reviews featureset tuple
# The first elemnt of the tuple is the feature set and the second element is the label
ground_truth = [r[1] for r in testing_set]

preds = [classifier.classify(r[0]) for r in testing_set]

from sklearn.metrics import f1_score

f1_score(ground_truth, preds, labels=['neg', 'pos'], average='micro')

import numpy as np
import matplotlib.pyplot as plt

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

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC

print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)

MNB_clf = SklearnClassifier(MultinomialNB())
MNB_clf.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_clf, testing_set)) * 100)

BNB_clf = SklearnClassifier(BernoulliNB())
BNB_clf.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BNB_clf, testing_set)) * 100)

LogReg_clf = SklearnClassifier(LogisticRegression())
LogReg_clf.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogReg_clf, testing_set)) * 100)

SGD_clf = SklearnClassifier(SGDClassifier())
SGD_clf.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGD_clf, testing_set)) * 100)

SVC_clf = SklearnClassifier(SVC())
SVC_clf.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_clf, testing_set)) * 100)

### Storing all models using pickle

def create_pickle(c, file_name):
    save_classifier = open(file_name, 'wb')
    pickle.dump(c, save_classifier)
    save_classifier.close()

classifiers_dict = {'ONB': [classifier, 'pickled_algos/ONB_clf.pickle'],
                    'MNB': [MNB_clf, 'pickled_algos/MNB_clf.pickle'],
                    'BNB': [BNB_clf, 'pickled_algos/BNB_clf.pickle'],
                    'LogReg': [LogReg_clf, 'pickled_algos/LogReg_clf.pickle'],
                    'SGD': [SGD_clf, 'pickled_algos/SGD_clf.pickle'],
                    'SVC': [SVC_clf, 'pickled_algos/SVC_clf.pickle']}

for clf, listy in classifiers_dict.items():
    create_pickle(listy[0], listy[1])

acc_scores = {}
for clf, listy in classifiers_dict.items():
    # getting predictions for the testing set by looping over each reviews featureset tuple
    # The first elemnt of the tuple is the feature set and the second element is the label
    acc_scores[clf] = accuracy_score(ground_truth, predictions[clf])
    print(f'Accuracy_score {clf}: {acc_scores[clf]}')

from sklearn.metrics import f1_score, accuracy_score

ground_truth = [r[1] for r in testing_set]
predictions = {}
f1_scores = {}
for clf, listy in classifiers_dict.items():
    # getting predictions for the testing set by looping over each reviews featureset tuple
    # The first elemnt of the tuple is the feature set and the second element is the label
    predictions[clf] = [listy[0].classify(r[0]) for r in testing_set]
    f1_scores[clf] = f1_score(ground_truth, predictions[clf], labels=['neg', 'pos'], average='micro')
    print(f'f1_score {clf}: {f1_scores[clf]}')

# Ensemble Model

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

#### Loading all models using pickle

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

# Initializing the ensemble classifier
ensemble_clf = EnsembleClassifier(ONB_Clf, MNB_Clf, BNB_Clf, LogReg_Clf, SGD_Clf)

# List of only feature dictionary from the featureset list of tuples
feature_list = [f[0] for f in testing_set]

# Looping over each to classify each review
ensemble_preds = [ensemble_clf.classify(features) for features in feature_list]

# %%

f1_score(ground_truth, ensemble_preds, average='micro')

# Function to do classification a given review and return the label a
# and the amount of confidence in the classifications
def sentiment(text):
    feats = find_features(text)
    return ensemble_clf.classify(feats), ensemble_clf.confidence(feats)

# sentiment analysis of reviews of captain marvel found on rotten tomatoes
text_a = '''The problem is with the corporate anticulture that controls these productions-and 
            the fandom-targeted demagogy that they're made to fulfill-which responsible casting 
                can't overcome alone.'''
text_b = '''Does it work? The short answer is: yes. There's enough to keep both diehard 
                Marvel fans and newcomers engaged.'''
text_c = '''It was lacking, a bit flat, and I'm honestly concerned about how she will enter
            the Marvel Cinematic Universe...it's so concerned with being a feminist film that 
            it forgets how to be a superhero movie.'''
text_d = '''The film may be about women breaking their shackles, but the lead actress feels kept 
            in check for much of the picture. Humor winds up being provided by Samuel Jackson's Nick 
            Fury, heart by Lashana Lynch's Maria Rambeau, and pathos by...well, it ain't Larson'''
text_e = '''"Everything was beautiful and nothing hurt"'''

sentiment(text_a), sentiment(text_b), sentiment(text_c), sentiment(text_d), sentiment(text_e)

## Random Forest

# converting the training set  into a pandas data frame

from tqdm import tqdm_notebook as tqdm
import time
import pandas as pd

df = pd.DataFrame([training_set[0][0]])
for f in tqdm(training_set[1:]):
    df = df.append([f[0]], ignore_index=True)

# converting the testing set  into a pandas data frame
df_test = pd.DataFrame([testing_set[0][0]])
for f in tqdm(testing_set[1:]):
    df_test = df_test.append([f[0]], ignore_index=True)

df.tail()

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=0)
X = df
y = [x[1] for x in training_set]

clf.fit(X, y)

X_test = df_test
y_test = [x[1] for x in testing_set]
clf.score(X, y)

from sklearn.datasets import load_iris

iris = load_iris()

# Model (can also use single decision tree)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10)

# Train
model.fit(iris.data, iris.target)
# Extract single tree
estimator = clf.estimators_[5]

from sklearn.tree import export_graphviz

# Export as dot file
export_graphviz(estimator, out_file='tree.dot',
                feature_names=df.columns,
                class_names=['neg', 'pos'],
                rounded=True, proportion=False,
                precision=2, filled=True)

# Convert to png using system command (requires Graphviz)
from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image

Image(filename='tree.png')

clf.decision_path(X)
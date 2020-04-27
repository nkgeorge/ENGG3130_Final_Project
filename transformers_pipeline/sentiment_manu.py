import os
from transformers import pipeline
import seaborn as sns


# This pipeline uses a fine tuned DistilBert model by default
def find_sent(file_name, directory):
    score_list = []
    positive = 0
    negative = 0

    file_name = directory + file_name

    fp = open(file_name, "r")
    data = fp.read()
    text = data.splitlines()
    fp.close()

    # Use default (and most accurate model) for sentiment analysis

    nlp = pipeline('sentiment-analysis', device=0)

    for x in text:
        comment_sent = nlp(x)
        score_list.append(comment_sent[0]['score'])
        if comment_sent[0]['label'] == 'POSITIVE':
            positive += 1
        else:
            negative += 1

    # sns.distplot(score_list)
    # average_sent goes from 0-1 with 0 being negative and 1 being positive
    return positive / negative


def sent_each_file(filelist, sentlist):
    for i in range(len(filelist)):
        print('Average sentiment of', ''.join(filelist[i]), ':', sentlist[i])


def sent_over_folder(directory):
    game_sent = []
    file_names = []

    for file in os.listdir(directory):
        game_sent.append(find_sent(file, directory))
        file_names.append(file)
        print('Done:', file)

    sent_each_file(file_names, game_sent)

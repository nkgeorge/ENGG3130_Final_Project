from transformers import pipeline
import os
import matplotlib.pyplot as plt

# This pipeline uses a fine tuned DistilBert model by default
def average(lst):
    return sum(lst) / len(lst)

def find_sent(file_name):
    score_list = []
    positive = 0
    negative = 0

    # file_name = 'data/' + file_name

    fp = open(file_name, "r")
    data = fp.read()
    text = data.splitlines()
    fp.close()

    # Use default (and most accurate model) for sentiment analysis
    nlp = pipeline('sentiment-analysis')

    for x in text:
        comment_sent = nlp(x)
        print(comment_sent)
        score_list.append(comment_sent[0]['score'])
        if comment_sent[0]['label'] == 'POSITIVE':
            positive += 1
        else:
            negative += 1

    # average_sent goes from 0-1 with 0 being negative and 1 being positive
    average_score = average(score_list)
    return positive / negative


# game_sent = []
# for file in os.listdir('data/'):
#     game_sent.append(find_sent(file))
#     print('Done:', file)
print(find_sent('Man U vs Everton.txt'))

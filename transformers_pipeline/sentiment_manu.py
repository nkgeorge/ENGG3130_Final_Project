from transformers import pipeline
import os

# This pipeline uses a fine tuned DistilBert model by default
def average(lst):
    return sum(lst) / len(lst)

def find_sent(file_name):
    score_list = []
    positive = 0
    negative = 0

    file_name = 'data/' + file_name

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

    # average_sent goes from 0-1 with 0 being negative and 1 being positive
    average_score = average(score_list)
    return positive / negative


game_sent = []
# Enter in path to data folder
for file in os.listdir('C:/Users/nkgeo/PycharmProjects/test2/data'):
    game_sent.append(find_sent(file))
    print('Done:', file)

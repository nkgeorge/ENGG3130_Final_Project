from transformers import pipeline

# This pipeline uses a fine tuned DistilBert model by default
# Models we can use: DistilBertConfig, AlbertConfig, CamembertConfig, XLMRobertaConfig, BartConfig, RobertaConfig, BertConfig, XLNetConfig, FlaubertConfig, XLMConfig
# Allocate a pipeline for sentiment-analysis
def Average(lst):
    return sum(lst) / len(lst)

fp = open("data2.txt", "r")
data = fp.read()

text = data.splitlines()
print(text)
fp.close()



# Use default (and most accurate model) for sentiment analysis
nlp = pipeline('sentiment-analysis')
average_score = []
positive = 0
negative = 0
for x in text:
    comment_sent = nlp(x)
    print(comment_sent)
    average_score.append(comment_sent[0]['score'])
    if comment_sent[0]['label'] == 'POSITIVE':
        positive += 1
    else:
        negative += 1

#average_sent goes from 0-1 with 0 being negative and 1 being positive
average_sent = (positive/negative)
print(Average(average_score))


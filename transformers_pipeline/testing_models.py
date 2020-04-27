from transformers import pipeline
# Models we can use: DistilBertConfig, AlbertConfig, CamembertConfig, XLMRobertaConfig, BartConfig, RobertaConfig, BertConfig, XLNetConfig, FlaubertConfig, XLMConfig
models = ['xlm-roberta-large',
          'xlm-roberta-base',
          'camembert-base',
          'albert-base-v2',
          'distilbert-base-cased-distilled-squad']

text = ['Hello this is a bad sentiment']

# nlp = pipeline('sentiment-analysis', model='XLMRobertaForSequenceClassification', tokenizer='XLMRobertaTokenizer')
# for x in text:
#     print('XLMRoberta:', nlp(x))


# Perform Sentiment Analysis on text in 'text' with models in 'models
for x in models:

    nlp = pipeline('sentiment-analysis', model=x, device=0)   # Allocate a pipeline for sentiment-analysis
    for y in text:
        print(x, ":", nlp(y))   #Perform Sentiment analysis and display results

from transformers import pipeline
# This pipeline uses a fine tuned DistilBert model
# Models we can use: DistilBertConfig, AlbertConfig, CamembertConfig, XLMRobertaConfig, BartConfig, RobertaConfig, BertConfig, XLNetConfig, FlaubertConfig, XLMConfig
# Allocate a pipeline for sentiment-analysis
models = ['distilroberta-base',
          'distilbert-base-cased-distilled-squad',
          'camembert-base',
          'albert-base-v2',
          'xlm-roberta-base']

text = ['A monumental achievement in film and the rare film that goes from "very good" to "historically great" in its final act.',
        'Its symptomatic of an awkward, unwieldy movie that has lots of material to show you and lots of surface distractions, but nothing at all to say.',
        'Having Oblak compared to Adrian tonight was the difference and shows why Oblak is probably the best keeper in the world. Insane performance and an incredible game. Doesnt help that Adrian was shockingly bad']
nlp = pipeline('sentiment-analysis')
for x in text:
    print('Default fine tuned DistilBert model:', nlp(x))


# Perform Sentiment Analysis on text in 'text' with models in 'models
for x in models:
    nlp = pipeline('sentiment-analysis', model=x)   # Allocate a pipeline for sentiment-analysis
    for y in text:
        print(x, ":", nlp(y))   #Perform Sentiment analysis and display results

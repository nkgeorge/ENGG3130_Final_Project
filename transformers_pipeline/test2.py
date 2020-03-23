from transformers import pipeline
# This pipeline uses a fine tuned DistilBert model
# Models we can use: DistilBertConfig, AlbertConfig, CamembertConfig, XLMRobertaConfig, BartConfig, RobertaConfig, BertConfig, XLNetConfig, FlaubertConfig, XLMConfig
# Allocate a pipeline for sentiment-analysis
nlp = pipeline('sentiment-analysis', model='openai-gpt')

text = ['A monumental achievement in film and the rare film that goes from "very good" to "historically great" in its final act.',
        'Its symptomatic of an awkward, unwieldy movie that has lots of material to show you and lots of surface distractions, but nothing at all to say.',
        'Having Oblak compared to Adrian tonight was the difference and shows why Oblak is probably the best keeper in the world. Insane performance and an incredible game. Doesnt help that Adrian was shockingly bad']

# Perform Sentiment Analysis on various types of text
for x in text:
    print(nlp(x))

# Model:
# [{'label': 'POSITIVE', 'score': 0.9998766}]
# [{'label': 'NEGATIVE', 'score': 0.9979703}]
# [{'label': 'POSITIVE', 'score': 0.99807125}]

# Model: bert-base-uncased
# [{'label': 'LABEL_1', 'score': 0.6228135}]
# [{'label': 'LABEL_1', 'score': 0.60044444}]
# [{'label': 'LABEL_1', 'score': 0.60955757}]
from transformers import pipeline

text = ['It\u2019s the hope that kills you.',
        'Probably repeat for the next 60',
        'Might as well have kept LVG.',
        'Bullshit, deserved draw against liverpool who had won about 7 million games in a row, and then a comfortable away win against norwich, aswell as that away win at stamford bridge, all of this with the only world class outfield player united has out injured. This was a very poor perfomance but that doesent mean the previous games were not much better']
nlp = pipeline('sentiment-analysis')

for x in text:
    comment_sent = nlp(x)
    print(comment_sent)
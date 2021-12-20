# A program that substitues words for their sysnonyms in a news article

import os
import sys
import json
import random
import nltk

from nltk.corpus import wordnet

# Get articles
input_path = sys.argv[1]
output_path = sys.argv[2]

input = open(input_path, "r");

articles = []

for line in input:
    item = json.loads(line)
    articles.append(item)

input.close()

# Generate articles
output_file = open(output_path, "w+")

for i in range(0, len(articles)):
    article = articles[i]["text"]
    words = article.split()

    idxs = list(range(0, len(words)))
    random.shuffle(idxs)
    
    num = 1;
    
    for j in idxs:
        syn = wordnet.synsets(words[j])[0].lemmas()[0].name()
        
        if (wordnet.synset(words[j]).wup_similarity(wordnet.synset(syn)) > 0.55):
            words[j] = syn
            num = num + 1

        new_article = ' '.join(words)
        articles[i]["text"] = new_article

        articles[i]["authors"] = "[]"
        articles[i]["split"] = "test"
        articles[i]["label"] = ""
        articles[i]["numSyns"] = num

        output_file.write(json.dumps(unmodified[i]) + '\n')

output_file.close()

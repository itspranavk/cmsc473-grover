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
        synset = wordnet.synsets(words[j])
        
        try:
            if (len(synset) > 0):
                words[j] = synset[0].lemmas()[0].name()
                num  = num + 1
        except ValueError:
             print("Something went wrong for word: " + synset[0].lemmas()[0].name() + "\n");

        new_article = ' '.join(words)
        articles[i]["text"] = new_article

        articles[i]["authors"] = "[]"
        articles[i]["split"] = "test"
        articles[i]["label"] = ""
        articles[i]["numSyns"] = str(num)

        output_file.write(json.dumps(articles[i]))
	output_file.write("\n")
		
    print(num)

output_file.close()

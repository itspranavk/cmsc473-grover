# A program that tests the output of Grover when words are changed to their synonyms

import os
import sys
import json
import random

# Get articles
input_unmodified_path = sys.argv[1]
input_modified_path = sys.argv[2]
output_path = sys.argv[3]

unmodified_articles = open(input_unmodified_path, "r");
modified_articles = open(input_modified_path, "r");

unmodified = []
modified = []

for line in unmodified_articles:
    item = json.loads(line)
    unmodified.append(item)

for line in modified_articles:
    item = json.loads(line)
    modified.append(item)

unmodified_articles.close()
modified_articles.close()

# Generate articles
output_file = open(output_path, "w+")

for i in range(0, len(unmodified)):
    article = unmodified[i]["text"]
    syn_article = modified[i]["text"]

    words = article.split()
    syns = syn_article.split()

    idxs = list(range(0, len(words)))
    random.shuffle(idxs)

    for j in idxs:
        words[j] = syns[j]

        new_article = ' '.join(words)
        unmodified[i]["text"] = new_article

        unmodified[i]["authors"] = "[]"
        unmodified[i]["split"] = "test"
        unmodified[i]["label"] = ""

        output_file.write(json.dumps(unmodified[i]) + '\n')

output_file.close()

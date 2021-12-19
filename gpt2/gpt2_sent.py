import json
import requests
from nltk import tokenize

def getGPT2(og):
    r = requests.post(
        "https://api.deepai.org/api/text-generator",
        data={
            'text': og,
        },
        headers={'api-key': '8afebcf5-1c8a-4c02-9ba2-dbffb65295d3'}
    )
    if 'output' not in r.json():
        print("ERROR WITH API")
        exit(1)
    newList = tokenize.sent_tokenize(r.json()['output'])
    while len(newList) < 2:
        print("RETRYING")
        r = requests.post(
            "https://api.deepai.org/api/text-generator",
            data={
                'text': og,
            },
            headers={'api-key': '8afebcf5-1c8a-4c02-9ba2-dbffb65295d3'}
        )
        if 'output' not in r.json():
            print("ERROR WITH API")
            exit(1)
        newList = tokenize.sent_tokenize(r.json()['output'])
    
    newPiece = replaceQuotes(newList[1])
    return newPiece

def replaceQuotes(text):
    replacing = text.replace("\'", "\u2019")
    start = True
    while "\"" in replacing:
        if start:
            replacing = replacing.replace("\"", "\u201c", 1)
            start = False
        else:
            replacing = replacing.replace("\"", "\u201d", 1)
            start = True
    return replacing

def joinSentences(plens, sentences):
    text = ""
    total = 0
    i = 0
    for plen in plens:
        while i < total + plen:
            text += sentences[i] + " "
            i += 1
        text = text[:-1]
        text += "\n"
        total += plen
    return text

with open('realnews/supertiny.jsonl', 'r') as fr:
    lines  = fr.readlines()

toWrite = []
for line in lines:
    ogArticle = json.loads(line)
    toWrite.append(json.dumps(ogArticle) + "\n")
    ogArticle['label'] = "machine"
    text = ogArticle['text']
    ogParagraphs = text.split("\n")
    for i in range(len(ogParagraphs)):
        ogParagraphs[i] = tokenize.sent_tokenize(ogParagraphs[i])
    ogSentences = []
    for p in ogParagraphs:
        ogSentences.extend(p)
    plens = [len(p) for p in ogParagraphs]
    newSentences = ogSentences.copy()
    for i in range(1, len(ogSentences)):
        old = ogSentences[i - 1]
        newSentences[i] = getGPT2(old)
        ogArticle['text'] = joinSentences(plens, newSentences)
        toWrite.append(json.dumps(ogArticle) + "\n")

with open('gpt2_sent.jsonl', 'w') as fw:
    fw.writelines(toWrite)


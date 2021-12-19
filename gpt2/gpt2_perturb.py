import json
import requests

def getGPT2(og):
    r = requests.post(
        "https://api.deepai.org/api/text-generator",
        data={
            'text': og,
        },
        headers={'api-key': '8afebcf5-1c8a-4c02-9ba2-dbffb65295d3'}
    )
    new = r.json()['output']
    newList = new.split("\n\n")
    newPiece = ""
    i = 1
    while(i < 3  and i < len(newList)):
        newPiece += " " + newList[i]
        i += 1
    newPiece = replaceQuotes(newPiece)
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

with open('realnews/supertiny.jsonl', 'r') as fr:
    lines  = fr.readlines()

toWrite = []
for line in lines:
    ogArticle = json.loads(line)
    toWrite.append(json.dumps(ogArticle) + "\n")
    ogArticle['label'] = "machine"
    text = ogArticle['text']
    ogSplit = text.split('\n')
    newSplit = ogSplit.copy()
    for i in range(1, len(ogSplit)):
        old = ogSplit[i - 1]
        newSplit[i] = getGPT2(old)
        ogArticle['text'] = "\n".join(newSplit)
        toWrite.append(json.dumps(ogArticle) + "\n")

with open('gpt2_set.jsonl', 'w') as fw:
    fw.writelines(toWrite)


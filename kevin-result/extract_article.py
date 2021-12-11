import json


with open('syn_output.jsonl', 'r') as json_file:
    json_list = list(json_file)

with open('discrimination-data_large_p=0.96.jsonl', 'r') as new_json_file:
    new_json_list = list(new_json_file)

print(len(json_list))
print(len(new_json_list))

kevin_keys = ['url', 'url_used', 'title', 'text', 'summary', 'authors', 'publish_date', 'domain', 'warc_date', 'status', 'split', 'inst_index']
grover_keys = ['article', 'domain', 'title', 'date', 'authors', 'ind30k', 'url', 'label', 'orig_split', 'split', 'random_score']

# Ordered_Dict = {k : Not_Ordered_Dict[k] for k in key_order}


new_list = []

for json_str in json_list:
    result = json.loads(json_str)
    result['article'] = result['text']
    result.pop('text')
    result.pop('summary', None)
    result.pop('gens_article', None)
    result.pop('url_used', None)
    result['authors'] = ""
    result["label"] = "machine"
    result['date'] = result['publish_date']
    result.pop('publish_date', None)
    result.pop('warc_date', None)
    result.pop('inst_index', None)
    result.pop('status', None)
    result = {k : result.get(k, None) for k in grover_keys}
    result['article'] = result['article']+"\n"
    new_list.append(result)

with open('output.jsonl', 'w') as outfile:
    for entry in new_list:
        json.dump(entry, outfile)
        outfile.write('\n')



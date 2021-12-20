The script generate_sysnonyms.py takes as input a set of articles and outputs a new set of articles, in which words have been changed
to their sysonyms (the first article in the new set has one word changed, the next article has two words changed, etc.). The new articles
have a new JSON element, 'NumSyns'. which is simly the number of sysnonyms present in the article. Sysonyms are generated using the
[NLTK wordnet](https://www.nltk.org/).

To install NLTK Wordnet:
1. in the anaconda prompt, run `pip install nltk`
2. Run the python shell and enter two commands, `import nltk` and `nltk.download()`. This will open the nltk downloader
3. In the downloader, select "Corpora" and scroll down to "Wordnet". Click download.

Then, call `python generate_synonyms.py <input directory> <output directory>` to generate the synonyms.

Lastly, run 

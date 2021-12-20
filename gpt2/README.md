# GPT-2 Transformer Language Model Perturbations
- Paragraph replacement
- Sentence replacement
- Done using DeepAI's GPT-2 based text generator: https://deepai.org/machine-learning-model/text-generator
- Includes unittests for helper functions within sentence (similar to paragraph) replacement

The gpt2_sent and gpt2_perturb scripts take in a set of real-news articles, and write articles with perturbations out to a necessary file. The scripts call the DeepAI API to get paragraph/sentence replacements, then applies them to the original article and for each perturbation, creates a new JSONL entry that corresponds to a partially machine-generated article. The perturbation set for the experiment can be found in gpt2_sent.jsonl (sentence replacement) and gpt2_data.jsonl (paragraph replacement). The probabilities are found in marko_test-probs.npy (paragraph replacement) and marko17_test-probs.npy (sentence replacement)

## Running the scripts
Make sure that the output files specified in the scripts do not exist or are not located in the same directory.
Run for sentence replacement:
'''
python gpt2_sent.py
'''
Run for paragraph replacement:
'''
python gpt2_perturb.py
'''

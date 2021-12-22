import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
import os
import numpy as np
import tensorflow as tf
import nltk
from tensorflow.python.ops.session_ops import encode_resource_handle
from sample.encoder import get_encoder

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "input", None,
    "Path to .jsonl file containing articles.")

flags.DEFINE_string(
    "output", None,
    "Path to output .jsonl file")

flags.DEFINE_integer(
    "index", 0,
    "Index of base article")

flags.DEFINE_float(
    "R", 0.5,
    "Proportion of base article retained")

flags.DEFINE_bool(
    "use_gens", False, 
    "Whether to run generated articles")

flags.DEFINE_bool(
    "spectrum", False,
    "Whether to generate a spectrum of results. Will ignore R."
)

flags.DEFINE_bool(
    "continuous", False,
    "Instead of randomizing at each step, continue with previous iteration."
)

flags.DEFINE_bool(
    "position_test", False,
    "Tests effect on position on confidence score."
)

flags.DEFINE_bool(
    "insert", False,
    "Insert sentences instead of substituting them."
)

def main(_):
    # We'll store the output jsonl here
    out = []
    # Grover-friendly jsonl format
    base = {'domain': '', 'publish_date': '', 'authors': '', 'title': '', 'text': '', 'summary': None, 'split': 'test', 'label': 'human'}
    # Store sentences from target and source 
    target = np.array([])
    source = np.array([])
    # Read all input files
    with open(FLAGS.input, "r") as f:
        for i, l in enumerate(f):
            item = json.loads(l)
            # Target
            if i == FLAGS.index:
                base['domain'] = item['domain']
                base['publish_date'] = item['publish_date']
                # Grover issue with authors as lists
                if type(item['authors']) == list:
                    base['authors'] = item['authors'][0]
                else:
                    base['authors'] = item['authors']
                base['title'] = item['title']
                # Use nltk to split sentences
                target = np.array(nltk.tokenize.sent_tokenize(item['text']))
                # target = np.array(nltk.tokenize.sent_tokenize(item['gens_article'][0]))
            # Source
            else:
                # Decide whether to use human-written or machine-written articles as source
                if not FLAGS.use_gens:
                    # Use nltk to split sentences
                    tokens = np.array(nltk.tokenize.sent_tokenize(item['text']))
                    source = np.concatenate((source, tokens))
                else:
                    # Use nltk to split sentences
                    tokens = np.array(nltk.tokenize.sent_tokenize(item['gens_article'][0]))
                    source = np.concatenate((source, tokens))
    # Intiialize random number generator
    rng = np.random.default_rng()
    # Blending
    if not FLAGS.position_test:
        # Not a spectrum, use R
        if not FLAGS.spectrum:
            # Retain R% of target article
            n = int((1-FLAGS.R)*target.shape[0])
            target_index = rng.choice(target.shape[0], n, replace=False)
            source_index = rng.choice(source.shape[0], n, replace=False)
            # Substitution
            if not FLAGS.insert:
                target[target_index] = source[source_index]
            # Insertion
            else:
                target = np.insert(target, target_index, source[source_index])
            # Store resultant article
            base['text'] = " ".join(target)
            out.append(json.dumps(base))
        # Generate spectrum, ignore R
        else:
            base['text'] = " ".join(target)
            out.append(json.dumps(base))
            # Randomize indices at each step
            if not FLAGS.continuous:
                for r in np.arange(1, target.shape[0]):
                    temp = target.copy()
                    target_index = rng.choice(temp.shape[0], r, replace=False)
                    source_index = rng.choice(source.shape[0], r, replace=False)
                    # Substitution
                    if not FLAGS.insert:
                        temp[target_index] = source[source_index]
                    # Insertion
                    else:
                        temp = np.insert(temp, target_index, source[source_index])
                    # Store resultant article
                    base['text'] = " ".join(temp)
                    out.append(json.dumps(base))
            # Generate indices only at the start
            else:
                n = target.shape[0]
                target_index = rng.choice(target.shape[0], n, replace=False)
                source_index = rng.choice(source.shape[0], n, replace=False)
                temp = target.copy()
                for i in np.arange(n):
                    # Substitution
                    if not FLAGS.insert:
                        temp[target_index] = source[source_index]
                    # Insertion
                    else:
                        temp = np.insert(temp, target_index, source[source_index])
                    # Store resultant article
                    base['text'] = " ".join(temp)
                    out.append(json.dumps(base))
    # Effect of position
    else:
        # Choose one sentence at random
        source_index = rng.choice(source.shape[0], 1, replace=False)
        # Generate spectrum by changing location of placement
        for i in np.arange(target.shape[0]):
            temp = target.copy()
            # Substitution
            if not FLAGS.insert:
                temp[i] = source[source_index]
            # Insertion
            else:
                temp = np.insert(temp, i, source[source_index])
            # Store resultant article
            base['text'] = " ".join(temp)
            out.append(json.dumps(base))
    # Write .jsonl output
    with open(FLAGS.output, "w") as f:
        f.writelines("%s\n" % l for l in out)

if __name__ == "__main__":
    flags.mark_flag_as_required("input")
    flags.mark_flag_as_required("output")
    tf.app.run()
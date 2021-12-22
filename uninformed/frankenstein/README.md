# Frankenstein Articles

Under each child directory, we have included the data generated using `frankenstein.py` script. The goal of the given script is to randomize the blending process and output articles in a Grover-friendly format for discrimination purposes.

To run `frankenstein.py`, you can specify these arguments:
* `input` - Path to .jsonl file containing articles
* `output` - Path to output .jsonl file
* `index` - Index of base article
* `R` - Proportion of base article retained
* `use_gens` - Whether to run generated articles
* `spectrum` - Whether to generate a spectrum of results. **Will ignore R**
* `continuous` - Instead of randomizing at each step, continue with previous iteration
* `position_test` - Tests effect on position on confidence score
* `insert` - Insert sentences instead of substituting them

Sample usage:

```text
python frankenstein.py -input data.jsonl -output frankenstein.jsonl -index 5 -R 0.7 -use_gens
```

## Blending Articles with Different Classification
### i.e Blending Human-written and Machine-written articles

Under `./diff_class`, we have included all runs of Grover discrimination, including data files, probabilities, and generated graphs.

## Blending Articles with Same Classification

Under `./same_class`, we have included all runs of Grover discrimination, including data files, probabilities, and generated graphs.

## Insertion instead of Substitution

Under `./insertion`, we have included all runs of Grover discrimination, including data files, probabilities, and generated graphs.

## Effect of Position

Under `./position`, we have included all runs of Grover discrimination, including data files, probabilities, and generated graphs.
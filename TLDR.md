# A place to keep track of what we're doing!

## Model Selection

We are using the medium generator, medium discriminator, `p=0.96` model for testing purposes. It is the right balance between accuracy and computational requirements. I've uplaoded the model on google drive for easy access. Downloading from cloud storage is a bit finicky and inconsistent.

Once you download the model, place it under `models/medium`. We will also use the `lm/config/large.json` configuration so our code is consistent with the number of tensors in each hidden layer (768 in base vs 1024 in medium)

There's an inconsistency in naming convention since the repository was published. Since 2019, it appears large models were renamed to medium.

**To make it clear: when we say large, it is equivalent to saying medium.**

## Generating Articles

We can prepare data such that it gives the generator context. The April 2019 sample is a good example containing 8 articles.

We can generate fake articles using the command:

```
PYTHONPATH=$(pwd) python sample/contextual_generate.py -model_config_fn lm/configs/large.json -model_ckpt models/medium/model.ckpt-1562 -metadata_fn sample/april2019_set_mini.jsonl -out_fn april2019_set_mini_out.jsonl -top_p 0.96
```

This will use the chosen model with the correct configuration file for the network structure.

This does take a while to run because the medium model is pretty good. It creates weird articles but it's surprisingly coherent!

**TO-DO Maybe we can try to play around the vocab size (under `lm/config/large.json`) for more contextual articles?**

## Training

### Preparing Data

A few things to keep in mind:
- Apparently it doesn't support multiple authors **TO-DO**
- Make sure all training articles have flag `"split": "train"` in the `jsonl` file.
- Make sure all labels are set correctly `"label": "human"` or `"label": "machine"`

I've created two additional datasets from the `realnews_tiny.jsonl` dataset: one for training and one more testing!

### Training the Model

This command will start the training process.

```
PYTHONPATH=$(pwd) python discrimination/run_discrimination.py -config_file lm/configs/large.json -input_data realnews/realnews_tiny-train.jsonl -output_dir models/test -do_train True -batch_size 5 -num_train_epochs 1.0
```

The trained model will be stored under `models/test`. Make sure to delete this directory if you are retraining the model for the same output directory. This will create a small scale generator and create checkpoints for the discriminator.

## Testing

### Preparing Data

A few things to keep in mind:
- Apparently it doesn't support multiple authors **TO-DO**
- Make sure all testing articles have flag `"split": "test"` in the `jsonl` file.
- Make sure all labels are set correctly `"label": "human"` or `"label": "machine"`

### Testing the Model

This command will start the testing process.

```
PYTHONPATH=$(pwd) python discrimination/run_discrimination.py -config_file lm/configs/large.json -input_data realnews/realnews_tiny-test.jsonl -output_dir models/medium -batch_size 5 -predict_test
```

Make sure to include the `-predict_test` flag! Otherwise, the model will not actually test anything. After running the discrimination, the model will generate a few files: `test.tf_record` and `test-probs.npy`. 

We only care about `test-probs.npy`. This contains the internal probabilities of each of our two labels before clasification, stored as a numpy file. You can load the file in python using:

```python
import numpy as np
probs = np.load('models/medium/test-probs.npy')
```

This will assign the variable `probs` an array of shape `(n,2)` where `n` is the number of testing articles.

I've also modifed the `run_discrimination.py` file to output the predicted and actual labels before calculating the accuracy for sanity's sake. Keep in mind, for the model, the labels are assigned numerical classes. The label `machine` is class `0` and label `human` is class `1`.

## Future

Can we add more labels? Currently the model only supports machine and human labels. What happens if we train a model with three labels: `human-real`, `human-fake`, and `machine-fake`?

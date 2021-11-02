# Installing Anaconda, Cuda, and Grover

#### Installing Anaconda
  If you do not already have anaconda installed, use <a href="https://www.anaconda.com/products/individual">this link</a> to download the installer. 
Once installed, you may proceed to the next step.

#### Installing Cuda
  The github page for Grover claims that installing Nvidia Cuda v10.0 is required for using `tensorflow_gpu`. I am not sure if this step is necessary since we are using 
our own GPU cluster. in either case, if you wish to install Cuda, use <a href="https://developer.nvidia.com/cuda-10.0-download-archive">this link</a>.

#### Downloading Grover
  You can either clone Grover or dowload a .zip archive from <a href="https://github.com/rowanz/grover">this repository</a>. 
  
  
# Setting up the Environent
First, open the Anaconda Prompt and navigate to the `grover-master` folder. Then:
  
  1. Use `conda create -n grover python="3.6"` to create a new conda environment named `grover`.
  2. Use `conda activate grover` to enter the new environment. If you wish to exit the environment, use `conda deactivate grover`.

Next, you will need to install the required packages. To do this:

  1. Enter `pip install tensorflow==1.13.1`. This will simply install tensorflow.
  2. Use `pip install requirements-gpu.txt`. This will install the remaining packages.

Finally, you will have to change some environment variables, namely `PYTHONPATH` and `LD_LIBRARY_PATH`

  1. Use `set PYTHONPATH=<current directory>` to set python's `sys` path to the current directory (replace `<current directory>` with the actual path to your current 
  directory, which should be the `grover-master` folder).
  2. Enter `set LD_LIBRARY_PATH=<cuda library directory>` to tell tensorflow where the Cuda library files are. For me, the directory was 
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64`.
  
Assuming all of this worked, you should be ready to test Grover out!


# Using Grover for Context-Based Generation
The following steps act as a sort of quickstart to see if Grover is working:

  1. `python download_model.py base`- this will download a pre-trained model good for generating small datasets. I will discuss training a custom model later.
  2. `python sample/contextual_generate.py -model_config_fn lm/configs/base.json -model_ckpt models/base/model.ckpt -metadata_fn sample/april2019_set_mini.jsonl -out_fn       
  april2019_set_mini_out.jsonl`- this program generates machine-written articles given a set of human-written articles. The output of this program can be viewed in  `april2019_set_mini_out.jsonl`, under `"gens_article"`. Below is an example paragraph from when I ran the program:
  
>Malware researcher Marcus Hutchins, former chief scientist at Microsoft and now a senior vice president at the chipmaker Atomico, pleaded guilty to a specific crime and was sentenced today to a year of probation, with all of that time back spent on the security module his criminal charge caused: malware, an illegal alternative to used code in Internet Explorer.

Once you have verified that `contextual_generate.py` is working, you can use it to generate all sorts of articles! You can use the following arguments to affect the output of the program:

* `-metadata_fn` (required): the jsonl file containing the context articles' metadata. Look at `sample\april2019_set_mini.jsonl` for an example.
* `-out_fn` (required): the jsonl file for the program's output
* `-model_config_fn` (required): the config file for the model to be used
* `-model_ckpt` (required): the checkpoints for the model being used. These should correspond with the configuration above.
* `-target`: what to generate- could be article, title, etc.
* `-batch_size`: how many generations to output per article in the metadata.
* `-num_folds`: the number of folds- usefull for splitting one big file up into multiple jobs.
* `-fold`: the current fold
* `-max_batch_size`: the max batch size. If none is specified, it is infered based on the number of hidden layers in the model (best not to change it).
* `-top_p`: p used for top p sampling



# Using Grover for Discrimination
To setup Grover for discrimination, you will frst have to train it on a realnews dataset. A small example dataset is provided in `grover-master`, but we will likely have to obtain the full dataset from Zellers.

### Training Grover using the example realnews set
First, you will have to locate and modify the example realnews set so the discrimination program knows how to process it.

1. find `grover-master\realnews\realnews_tiny.jsonl` and open it with any text editor.
2. Add to each object `"label":"human"`. This is simply the training label for each example article.
3. find the `"authors"` section for each object. Currently, these are in the form of string arrays. Due to a bug in the code, they will need to be changed to string literals. Take each string array (ex. `"authors":["Kevin Nolan", "Thomas Goldstein"]`) and change it to just the first author (ex. `"authors":"Kevin Nolan"`). <i>Note: this is a temporary fix for a bug in sample\encoder.py that we will need to address later on. However, for the purposes of testing the discriminator, this will do.</i>

Now you are ready to train the discriminator on the example realnews dataset:

1. In the anaconda prompt, enter `python discrimination\run_discrimination.py -input_data realnews\realnews_tiny.jsonl -output_dir test_discrimination_output -do_train True -batch_size 5 -num_train_epochs 1.0`. This will train Grover based on the tiny realnews dataset given. You can see the checkpoints, as well as other output items in the `test_discrimination_output` directory.
2. It is important to note that for training, `run_discrimination.py` expects that no output directory with the specified name exists. This means that if you want to train the model again, you will need to run `rmdir -r <output directory name>`, or specify a different name.

The model trained using this dataset is obviously terrible, and should not be used for real discrimination. However, it serves as an easy way to make sure everything is up and running.

### Using Grover for Discrimination
Using Grover in discrimination mode is easy, assuming you have a pre-trained model to work with.

1. The the anaconda prompt, use `python discrimintion\run_discrimination.py -input_data realnews\realnews_tiny.jsonl -output_dir test_discrimination_output`. Here we are using the same articles from the previous section to determine if they're written by people or not. Additionally, we are using the output directory from the previous section. It is important to use an output directory from a pre-trained model in this step because that's where `run_discrimination.py` is expecting the checkpoints to be.

As before, it is sort of pointless to test the model using the same data that was used to train it, but this really only serves to make sure it is working properly. Once we have the correct realnews dataset, we can use `run_discrimination.py` to train and test grover in a more realistic sense. In order to do that, the following arguments are specified:

* `-config_file`: the onfiguration file for the model (set to `lm\configs\base.json` by default).
* `-input_data` (required): the input file(s) for training or using the model
* `-output_dir` (required): the output directory. If you are training the model, this directory should not already exist.
* `-additional_data`: any other data you would like to use when training or using a model
* `-init_checkpoint`: the initial checkpoint from the pre-trained model. if the output dir is specified correctly, this is not required.
* `-max_seq_length`: total input sequence length after tokenization
* `-iterations_per_loop`: how many steps to make in each estimator call
* `-batch_size`: the batch size
* `-max_training_examples`: the maximum number of training examples, if you want to limit the number.
* `-do_train` (required if training): set to true if you would like to train the model. Otherwise, do not specify.
* `-predict_val`: whether to run eval on the dev set
* `-predict_test`: whether to run the model in inference mode on the test set
* `-num_train_epochs`: the number of training epochs
* `-warmup_proportion`: proportion of training to perform linear learning rate warmup for
* `-adafactor`: whether to run adafactor
* `-learning_rate`: initial learning rate

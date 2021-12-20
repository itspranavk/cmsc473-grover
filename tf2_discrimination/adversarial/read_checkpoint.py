from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import numpy as np
#print_tensors_in_checkpoint_file(file_name='gs://grover-models/discrimination/generator=base~discriminator=grover~discsize=base~dataset=p=0.96/model.ckpt-1562',tensor_name='newslm/embeddings/word_embed',all_tensors=False)

from tensorflow.python import pywrap_tensorflow
import os

checkpoint_path = 'gs://grover-models/discrimination/generator=base~discriminator=grover~discsize=base~dataset=p=0.96/model.ckpt-1562'
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

for key in var_to_shape_map:
    if key == 'newslm/embeddings/word_embed':
        with open('embedding.npy', 'wb') as f:
            np.save(f, reader.get_tensor(key))
    #pass
    print("tensor_name: ", key, reader.get_tensor(key).shape) # Remove this is you want to print only variable names

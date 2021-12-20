import tensorflow as tf

def convert_tf1_to_tf2(checkpoint_path, output_prefix):
    """Converts a TF1 checkpoint to TF2.

    To load the converted checkpoint, you must build a dictionary that maps
    variable names to variable objects.
    ```
    ckpt = tf.train.Checkpoint(vars={name: variable})
    ckpt.restore(converted_ckpt_path)

      ```

      Args:
        checkpoint_path: Path to the TF1 checkpoint.
        output_prefix: Path prefix to the converted checkpoint.

      Returns:
        Path to the converted checkpoint.
      """
    vars = {}
    reader = tf.train.load_checkpoint(checkpoint_path)
    dtypes = reader.get_variable_to_dtype_map()
    for key in dtypes.keys():
        vars[key] = tf.Variable(reader.get_tensor(key))
    return tf.train.Checkpoint(vars=vars).save(output_prefix)

check_point_path = 'gs://grover-models/discrimination/generator=base~discriminator=grover~discsize=base~dataset=p=0.96/model.ckpt-1562'
converted_path = convert_tf1_to_tf2(check_point_path,
                                    'converted-tf1-to-tf2')

# Try loading the converted checkpoint.
a = tf.Variable(0.)
b = tf.Variable(0.)
c = tf.Variable(0.)
ckpt = tf.train.Checkpoint(vars={'a': a, 'b': b, 'scoped/c': c})
ckpt.restore(converted_path).assert_consumed()
print("\nRestored [a, b, c]: ", [a.numpy(), b.numpy(), c.numpy()])
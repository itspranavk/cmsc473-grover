# Original work Copyright 2018 The Google AI Language Team Authors.
# Modified work Copyright 2019 Rowan Zellers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
For discrimination finetuning (e.g. saying whether or not the generation is human/grover)
"""
import json
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

from lm.dataloader import classification_convert_examples_to_features, classification_input_fn_builder, classification_input_dataset, classification_convert_examples_to_features_new
from lm.modeling import classification_model_fn_builder, GroverConfig, GroverModelTF2
from lm.utils import _save_np
from lm.optimization_adafactor import CustomSchedule, loss_function
from sample.encoder import get_encoder

flags = tf.compat.v1.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "config_file", '../lm/configs/base.json',
    "The config json file corresponding to the pre-trained news model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
        "input_data", 'gs://yanxu98/tinydata/realnews_tiny.jsonl',
    "The input data dir. Should contain the .tsv files (or other data files) for the task.")

flags.DEFINE_string(
        "additional_data", None,
    "Should we provide additional input data? maybe.")

flags.DEFINE_string(
    "output_dir", 'gs://yanxu98/tinydata_out',
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
        "init_checkpoint", 'gs://grover-models/discrimination/generator=base~discriminator=grover~discsize=base~dataset=p=0.96/model.ckpt-1562',
    "Initial checkpoint (usually from a pre-trained model).")

flags.DEFINE_integer(
    "max_seq_length", 1024,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer("iterations_per_loop", 32,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("batch_size", 32, "Batch size used")

flags.DEFINE_integer("max_training_examples", -1, "if you wanna limit the number")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("predict_val", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "predict_test", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool(
    "require_labels", True,
    "Whether require labels when running eval/test"
)

flags.DEFINE_float("num_train_epochs", 100.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_bool("adafactor", False, "Whether to run adafactor")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_bool("use_tpu", True, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", "grover",
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


def _flatten_and_tokenize_metadata(encoder, item):
    """
    Turn the article into tokens
    :param item: Contains things that need to be tokenized

    fields are ['domain', 'date', 'authors', 'title', 'article', 'summary']
    :return: dict
    """
    metadata = []
    for key in ['domain', 'date', 'authors', 'title', 'article']:
        val = item.get(key, None)
        if val is not None:
            metadata.append(encoder.__dict__[f'begin_{key}'])
            metadata.extend(encoder.encode(val))
            metadata.append(encoder.__dict__[f'end_{key}'])
    return metadata


def main(_):
    LABEL_LIST = ['machine', 'human']
    LABEL_INV_MAP = {label: i for i, label in enumerate(LABEL_LIST)}

    # These lines of code are just to check if we've already saved something into the directory
    if tf.io.gfile.exists(FLAGS.output_dir):
        print(f"The output directory {FLAGS.output_dir} exists!")
        #if FLAGS.do_train:
        #    print("EXITING BECAUSE DO_TRAIN is true", flush=True)
        #    return
        #for split in ['val', 'test']:
        #    if tf.gfile.Exists(os.path.join(FLAGS.output_dir, f'{split}-probs.npy')) and getattr(FLAGS,
        #                                                                                         f'predict_{split}'):
        #        print(f"EXITING BECAUSE {split}-probs.npy exists", flush=True)
        #        return
        # Double check to see if it has trained!
        #if not tf.gfile.Exists(os.path.join(FLAGS.output_dir, 'checkpoint')):
        #    print("EXITING BECAUSE NO CHECKPOINT.", flush=True)
        #    return
        #stuff = {}
        #with tf.gfile.Open(os.path.join(FLAGS.output_dir, 'checkpoint'), 'r') as f:
        #    # model_checkpoint_path: "model.ckpt-0"
        #    # all_model_checkpoint_paths: "model.ckpt-0"
        #    for l in f:
        #        key, val = l.strip().split(': ', 1)
        #        stuff[key] = val.strip('"')
        #if stuff['model_checkpoint_path'] == 'model.ckpt-0':
        #    print("EXITING BECAUSE IT LOOKS LIKE NOTHING TRAINED", flush=True)
        #    return


    #elif not FLAGS.do_train:
    #    print("EXITING BECAUSE DO_TRAIN IS FALSE AND PATH DOESNT EXIST")
    #    return
    else:
        tf.io.gfile.makedirs(FLAGS.output_dir)

    news_config = GroverConfig.from_json_file(FLAGS.config_file)

    # TODO might have to change this
    encoder = get_encoder()
    examples = {'train': [], 'val': [], 'test': []}
    np.random.seed(123456)
    tf.print("*** Parsing files ***", output_stream=sys.stdout)

    if True:
        with tf.io.gfile.GFile(FLAGS.input_data, "r") as f:
            for l in f:
                item = json.loads(l)
                # This little hack is because we don't want to tokenize the article twice
                context_ids = _flatten_and_tokenize_metadata(encoder=encoder, item=item)
                examples[item['split']].append({
                    'info': item,
                    'ids': context_ids,
                    'label': item['label'],
                })
                assert item['label'] in LABEL_INV_MAP

        additional_data = {'machine': [], 'human': []}
        if FLAGS.additional_data is not None:
            print("NOW WERE LOOKING AT ADDITIONAL INPUT DATA", flush=True)
            with tf.io.gfile.GFile(FLAGS.additional_data, "r") as f:
                for l in f:
                    item = json.loads(l)
                    # This little hack is because we don't want to tokenize the article twice
                    context_ids = _flatten_and_tokenize_metadata(encoder=encoder, item=item)
                    additional_data[item['label']].append({
                        'info': item,
                        'ids': context_ids,
                        'label': item['label'],
                    })
    tf.print("*** Done parsing files ***", output_stream=sys.stdout)
    print("LETS GO", flush=True)
    if FLAGS.max_training_examples > 0:

        examples_by_label = {'human': [], 'machine': []}
        for x in examples['train']:
            examples_by_label[x['label']].append(x)

        new_examples = []
        print("Unique machine examples: {} -> {}".format(len(examples_by_label['machine']),
                                                         FLAGS.max_training_examples), flush=True)
        machine_ex_to_keep = examples_by_label['machine'][:FLAGS.max_training_examples]

        # So we just cut down on the TRUE machine examples. now lets try adding in additional examples
        # examples_by_label['human'].extend(additional_data['human'])

        if len(additional_data['machine']) > 0:
            amount_to_add = len(examples_by_label['human']) - len(machine_ex_to_keep)
            if amount_to_add > 0:
                machine_ex_to_keep.extend(additional_data['machine'][:amount_to_add])

        for i, human_ex in enumerate(examples_by_label['human']):
            new_examples.append(human_ex)
            new_examples.append(machine_ex_to_keep[i % len(machine_ex_to_keep)])

        print("Length of examples: {} -> {}".format(len(examples['train']), len(new_examples)), flush=True)
        examples['train'] = new_examples

    # Training
    if FLAGS.do_train:
        num_train_steps = int(len(examples['train']) * FLAGS.num_train_epochs / FLAGS.batch_size)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
        assert num_train_steps > 0
    else:
        num_train_steps = None
        num_warmup_steps = None

    # Boilerplate
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
    strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)
    with strategy.scope():
        model = GroverModelTF2(config=news_config,
                               batch_size=FLAGS.batch_size,
                               seq_length=FLAGS.max_seq_length,
                               num_labels=len(LABEL_LIST),
                               pool_token_id=encoder.begin_summary,
                               pad_token_id=news_config.pad_token_id,
                               chop_off_last_token=False,
                               )
        learning_rate = CustomSchedule(news_config.hidden_size)

        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                             epsilon=1e-9)


        model.compile(optimizer='adam',
                      # Anything between 2 and `steps_per_epoch` could help here.
                      #steps_per_execution = 2,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['SparseCategoricalCrossentropy'])
    #weights = [model.embedder.full_position_embeddings, model.embedder.embedding_table, model.embedder.ln]
    #for weight in weights:
    #    print(weight.get_weights())
    #for layer in layers:
    #    print(layer.get_config(), layer.get_weights())
    #print(weights)
    #saver = tf1.train.Saver(var_list=model.layers)
    #saver.restore(sess=None, save_path=FLAGS.init_checkpoint)
    #ckpt = tf.train.Checkpoint(variables=weights)
    #ckpt.restore(FLAGS.init_checkpoint).assert_consumed()
    model.load_weights(FLAGS.init_checkpoint)
    print(model.embedder.embedding_table.get_weights())

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")

        tf.print(f"***** Recreating training file at {train_file} *****")
        classification_convert_examples_to_features_new(examples['train'], batch_size=FLAGS.batch_size,
                                                    max_seq_length=FLAGS.max_seq_length,
                                                    encoder=encoder, output_file=train_file,
                                                    labels=LABEL_LIST,
                                                    chop_from_front_if_needed=False)
        tf.print("***** Running training *****")
        tf.print("  Num examples = %d", len(examples['train']))
        tf.print("  Num epochs = %d", FLAGS.num_train_epochs)
        tf.print("  Batch size = %d", FLAGS.batch_size)
        tf.print("  Num steps = %d", num_train_steps)
        train_input_dataset = classification_input_dataset(input_file=train_file, seq_length=FLAGS.max_seq_length,
                                                           is_training=True, drop_remainder=True,batch_size=FLAGS.batch_size)
        for input_ids,label in train_input_dataset:
            with tf.GradientTape() as tape:
                #tape.watch(model.embedder.embedding_table)
                pred = model(input_ids)
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(label, pred)
            gradient = tape.gradient(loss, model.trainable_variables)
            #print(model.trainable_variables[1])
            #print(model.trainable_variables[1].numpy())
            print(gradient[1])
            indices = tf.make_ndarray(tf.make_tensor_proto(gradient[1].indices))
            values  = tf.make_ndarray(tf.make_tensor_proto(gradient[1].values))
            with open('embed+grad.npy', 'wb') as f:
                np.save(f, model.trainable_variables[1].numpy())
                grad = np.zeros_like(model.trainable_variables[1].numpy())
                cnt = 0
                for idx in indices:
                    grad[idx,:] = values[cnt,:]
                    cnt+=1
                    print(idx, cnt)
                np.save(f, grad)
            assert 0
            #print(gradient)
           

        model.fit(train_input_dataset, epochs=5,
                  steps_per_epoch=num_train_steps,
                  validation_data=train_input_dataset,
                  validation_steps=num_train_steps)

    splits_to_predict = [x for x in ['val', 'test'] if getattr(FLAGS, f'predict_{x}')]
    for split in splits_to_predict:
        num_actual_examples = len(examples[split])
        if num_actual_examples == 0:
            continue
        print(num_actual_examples)
        predict_file = os.path.join(FLAGS.output_dir, f'{split}.tf_record')
        tf.print(f"***** Recreating {split} file {predict_file} *****")
        classification_convert_examples_to_features_new(examples[split], batch_size=FLAGS.batch_size,
                                                    max_seq_length=FLAGS.max_seq_length,
                                                    encoder=encoder, output_file=predict_file,
                                                    labels=LABEL_LIST, pad_extra_examples=True,
                                                    chop_from_front_if_needed=False)
        val_input_dataset = classification_input_dataset(input_file=predict_file, seq_length=FLAGS.max_seq_length,
                                                         is_training=False, drop_remainder=True,batch_size=FLAGS.batch_size)

        probs = np.zeros((num_actual_examples, 2), dtype=np.float32)
        result = model.predict(val_input_dataset)
        for i, res in enumerate(result):
            if i < num_actual_examples:
                probs[i] = res['probs']

        _save_np(os.path.join(FLAGS.output_dir, f'{split}-probs.npy'), probs)

        preds = np.argmax(probs, 1)
        if FLAGS.require_labels:
            labels = np.array([LABEL_INV_MAP[x['label']] for x in examples[split][:num_actual_examples]])
            print('{} ACCURACY IS {:.3f}'.format(split, np.mean(labels == preds)), flush=True)


if __name__ == "__main__":
    #flags.mark_flag_as_required("input_data")
    #flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()


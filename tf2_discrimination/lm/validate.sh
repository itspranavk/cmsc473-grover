#!/usr/bin/env bash

#export PYTHONPATH=/home/rowanz/code/fakenewslm

max_seq_length=1024
num_tpu_cores=8
batch_size_per_core=2
model_type="base"
input_file="gs://yxu98_grover/data_1024/val0000.tfrecord"
OUTPUT_DIR="gs://yxu98_grover/validate"
init_checkpoint="gs://yxu98_grover/model.ckpt-4000"
#model_type=""

let batch_size="$batch_size_per_core * $num_tpu_cores"


python validate.py \
    --config_file=configs/${model_type}.json \
    --input_file=${input_file} \
    --output_dir=${OUTPUT_DIR} \
    --max_seq_length=${max_seq_length} \
    --batch_size=${batch_size} \
    --use_tpu=True \
    --tpu_name=$(hostname) \
    --num_tpu_cores=$num_tpu_cores \
    --init_checkpoint=${init_checkpoint} \
    --validation_name="preds.h5"

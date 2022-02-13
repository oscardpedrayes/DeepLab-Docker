#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================
#
# This script is used to run local test on PASCAL VOC 2012. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   bash ./local_test.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
#export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim



# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Run model_test first to make sure the PYTHONPATH is correctly set.
#python3.7 "${WORK_DIR}"/model_test.py

# Go to datasets folder and download PASCAL VOC 2012 segmentation dataset.
DATASET_DIR="/media/oscar/New Volume/DATASETS/PNOA"


# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
EXP_NAME='E007-06'
PASCAL_FOLDER="20200902"
EXP_FOLDER="exp/train_on_trainval_set_mobilenetv3"
INIT_FOLDER="${DATASET_DIR}/${PASCAL_FOLDER}/init_models"
TRAIN_LOGDIR="${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/train/${EXP_NAME}"
EVAL_LOGDIR="${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/eval/${EXP_NAME}"
VIS_LOGDIR="${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/vis"
METRICS_LOGDIR="${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/metrics/${EXP_NAME}"
EXPORT_DIR="${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/export/${EXP_NAME}"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${METRICS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="http://download.tensorflow.org/models"
CKPT_NAME="deeplab_mnv3_small_cityscapes_trainfine"
#CKPT_NAME="deeplab_mnv3_large_cityscapes_trainfine"

TF_INIT_CKPT="${CKPT_NAME}_2019_11_15.tar.gz"

cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

PASCAL_DATASET="${DATASET_DIR}/${PASCAL_FOLDER}/tfrecord"

# Train
# 1 epoch for 12BD = 25274/12 = 2106-7
# 1 epoch for 64BS = 25247/64 = 394-5 #23640 60ep, 11820 30ep, 126360 320ep 


#Start logging cpu/gpu data
# Una vez se comienza a entrenar se mide el rendimiento de la cpu y de la gpu cada 10 segundos
mpstat -P ALL 10 > "${METRICS_LOGDIR}"/temp_cpu.csv &
nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 10 > "${METRICS_LOGDIR}"/gpu_rendimiento.csv &


NUM_ITERATIONS=126360 #23640 #126360 #60epochs   #63180 #30 epochs 
python3.7 "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="mobilenet_v3_small_seg" \
  --optimizer='momentum' \
  --adam_learning_rate=0.00005 \
  --base_learning_rate=0.0005 \
  --end_learning_rate=0.0005 \
  --image_pooling_crop_size=257,257 \
  --image_pooling_stride=4,5 \
  --add_image_level_feature=1 \
  --aspp_convs_filters=128 \
  --aspp_with_concat_projection=0 \
  --aspp_with_squeeze_and_excitation=1 \
  --decoder_use_sum_merge=1 \
  --decoder_filters=12 \
  --decoder_output_is_logits=1 \
  --image_se_uses_qsigmoid=1 \
  --decoder_output_stride=8 \
  --output_stride=16 \
  --train_crop_size="257,257" \
  --train_batch_size=12 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=false \
  --tf_initial_checkpoint="${INIT_FOLDER}/deeplab_mnv3_small_cityscapes_trainfine/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${PASCAL_DATASET}" \
  --dataset="pnoa" 

#Stop logging CPU/GPU data
pkill mpstat
pkill nvidia-smi
# Modificar los csvs de rendimiento para que sea más sencillos tratarlos con Excel
python3.7 "${WORK_DIR}"/cpu_rend.py "${METRICS_LOGDIR}"/temp_cpu.csv "${METRICS_LOGDIR}"/cpu_rendimiento.csv # Adaptar el formato de lso datos de cpu para que sea más cómodo de tratar con excel
sed -i -- 's/ %//g' "${METRICS_LOGDIR}"/gpu_rendimiento.csv # Quitar los % del fichero de la gpu que molestan, y vale con la cabecera
sed -i -- 's/ MiB//g' "${METRICS_LOGDIR}"/gpu_rendimiento.csv # Quitar las unidades que luego para hacer medias y demás molesta

# Run evaluation. This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=82.20%.

python3.7 "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="train" \
  --model_variant="mobilenet_v3_small_seg" \
  --output_stride=16 \
  --decoder_output_stride=8 \
  --image_se_uses_qsigmoid=1 \
  --decoder_output_is_logits=1 \
  --decoder_filters=12 \
  --decoder_use_sum_merge=1 \
  --aspp_with_squeeze_and_excitation=1 \
  --aspp_with_concat_projection=0 \
  --aspp_convs_filters=128 \
  --add_image_level_feature=1 \
  --image_pooling_stride=4,5 \
  --image_pooling_crop_size=257,257 \
  --eval_crop_size="257,257" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${PASCAL_DATASET}" \
  --max_number_of_evaluations=1 \
  --dataset="pnoa"

python3.7 "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="mobilenet_v3_small_seg" \
  --output_stride=16 \
  --decoder_output_stride=8 \
  --image_se_uses_qsigmoid=1 \
  --decoder_output_is_logits=1 \
  --decoder_filters=12 \
  --decoder_use_sum_merge=1 \
  --aspp_with_squeeze_and_excitation=1 \
  --aspp_with_concat_projection=0 \
  --aspp_convs_filters=128 \
  --add_image_level_feature=1 \
  --image_pooling_stride=4,5 \
  --image_pooling_crop_size=257,257 \
  --eval_crop_size="257,257" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${PASCAL_DATASET}" \
  --max_number_of_evaluations=1 \
  --dataset="pnoa"
 # --expName="${METRICS_LOGDIR}/expPrueba.csv"

 python3.7 "${WORK_DIR}"/getMetrics.py "${EVAL_LOGDIR}/" "${METRICS_LOGDIR}/" expName

# Visualize the results.
python3.7 "${WORK_DIR}"/vis.py \
  --logtostderr \
  --vis_split="vis" \
  --model_variant="mobilenet_v3_small_seg" \
  --output_stride=16 \
  --decoder_output_stride=8 \
  --image_se_uses_qsigmoid=1 \
  --decoder_output_is_logits=1 \
  --decoder_filters=12 \
  --decoder_use_sum_merge=1 \
  --aspp_with_squeeze_and_excitation=1 \
  --aspp_with_concat_projection=0 \
  --aspp_convs_filters=128 \
  --add_image_level_feature=1 \
  --image_pooling_stride=4,5 \
  --image_pooling_crop_size=257,257 \
  --vis_crop_size="257,257" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${PASCAL_DATASET}" \
  --max_number_of_iterations=1 \
  --dataset="pnoa" 

# Export the trained checkpoint.
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

python3.7 "${WORK_DIR}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="mobilenet_v3_small_seg" \
  --output_stride=16 \
  --decoder_output_stride=8 \
  --image_se_uses_qsigmoid=1 \
  --decoder_output_is_logits=1 \
  --decoder_filters=12 \
  --decoder_use_sum_merge=1 \
  --aspp_with_squeeze_and_excitation=1 \
  --aspp_with_concat_projection=0 \
  --aspp_convs_filters=128 \
  --add_image_level_feature=1 \
  --image_pooling_stride=4,5 \
  --image_pooling_crop_size=257,257 \
  --num_classes=12 \
  --crop_size=257 \
  --crop_size=257 \
  --inference_scales=1.0

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.

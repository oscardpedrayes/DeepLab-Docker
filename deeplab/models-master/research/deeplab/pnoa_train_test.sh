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
PASCAL_FOLDER="20200902"
EXP_FOLDER="exp/train_on_trainval_set_xception65"
INIT_FOLDER="${DATASET_DIR}/${PASCAL_FOLDER}/init_models"
TRAIN_LOGDIR="${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/vis"
METRICS_LOGDIR="${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/metrics"
EXPORT_DIR="${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${METRICS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="http://download.tensorflow.org/models"
TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"
cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

PASCAL_DATASET="${DATASET_DIR}/${PASCAL_FOLDER}/tfrecord"

# Train 10 iterations.
# 1 epoch = 25274/12 = 2106-7


# Run evaluation. This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=82.20%.

python3.7 "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="train" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=28 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size="257,257" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${PASCAL_DATASET}" \
  --max_number_of_evaluations=1 \
  --dataset="pnoa" \
  --expName="${METRICS_LOGDIR}/expPrueba.csv"

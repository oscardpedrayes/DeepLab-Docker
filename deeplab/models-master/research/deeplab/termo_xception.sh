#!/bin/bash

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



# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
EXP_NAME='E001-11-bs4-1000ep-lr5e4-65w' #'YE002-01v2'
DATASET_DIR="/media/oscar/New Volume/DATASETS/TERMO/"

EXP_FOLDER="exp"
INIT_FOLDER="${DATASET_DIR}/init_models"
TRAIN_LOGDIR="${DATASET_DIR}/${EXP_FOLDER}/train/${EXP_NAME}"
EVAL_LOGDIR="${DATASET_DIR}/${EXP_FOLDER}/eval/${EXP_NAME}"
VIS_LOGDIR="${DATASET_DIR}/${EXP_FOLDER}/vis/${EXP_NAME}"
METRICS_LOGDIR="${DATASET_DIR}/${EXP_FOLDER}/metrics/${EXP_NAME}"
EXPORT_DIR="${DATASET_DIR}/${EXP_FOLDER}/export/${EXP_NAME}"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${METRICS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="http://download.tensorflow.org/models"

#TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz" #xception65
TF_INIT_CKPT="deeplabv3_xception_2018_01_04.tar.gz" #xception65 best?

#TF_INIT_CKPT="deeplab_cityscapes_xception71_trainfine_2018_09_08.tar.gz" #X71 worse 
#TF_INIT_CKPT="xception_71_2018_05_09.tar.gz" # X71 best

#TF_INIT_CKPT="resnet_v1_50_2018_05_04.tar.gz"
#TF_INIT_CKPT="xception_41_2018_05_09.tar.gz"

MODEL_NAME='xception_65'
MODEL_DIR="xception/model.ckpt" #"train_fine/model.ckpt" #

cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

INITIAL="${INIT_FOLDER}/${MODEL_DIR}" 
#If you want to reuse an old experiment
#EXP_NAME_CP='E012-02'
#INITIAL="${DATASET_DIR}/${EXP_FOLDER}/train/${EXP_NAME_CP}/model.ckpt-8454"


TF_DATASET="${DATASET_DIR}/tfrecord/"


DATASET='termo'
LEARNING_RATE=0.0001 #0.0005 
BATCH_SIZE=4
IMAGES_TRAINING=30 #2250 #1500 #1125 #634 #1125
EPOCHS=1000
#NUM_ITERATIONS=$((IMAGES_TRAINING * EPOCHS / BATCH_SIZE))
NUM_ITERATIONS=6668

### TRAIN ###
# python3.7 "${WORK_DIR}"/train.py \
#   --logtostderr \
#   --train_split="Train1" \
#   --model_variant="${MODEL_NAME}" \
#   --optimizer='adam' \
#   --adam_learning_rate="${LEARNING_RATE}" \
#   --base_learning_rate="${LEARNING_RATE}"  \
#   --end_learning_rate="${LEARNING_RATE}"  \
#   --atrous_rates=6 \
#   --atrous_rates=12 \
#   --atrous_rates=18 \
#   --output_stride=16 \
#   --decoder_output_stride=4 \
#   --train_crop_size="481,641"  \
#   --train_batch_size="${BATCH_SIZE}" \
#   --training_number_of_steps="${NUM_ITERATIONS}" \
#   --fine_tune_batch_norm=false \
#   --tf_initial_checkpoint="${INITIAL}"  \
#   --train_logdir="${TRAIN_LOGDIR}" \
#   --dataset_dir="${TF_DATASET}" \
#   --dataset="${DATASET}"
#
#### TESTING ###

#python3.7 "${WORK_DIR}"/runEval.py "${EXP_NAME}" "${MODEL_NAME}" "Train1" "${DATASET}" "${DATASET_DIR}/" "${TF_DATASET}" 
#python3.7 "${WORK_DIR}"/getMetrics.py "${EVAL_LOGDIR}/Train1/" "${METRICS_LOGDIR}/Train1/" 

#python3.7 "${WORK_DIR}"/runEval.py "${EXP_NAME}" "${MODEL_NAME}" "Test" "${DATASET}" "${DATASET_DIR}/" "${TF_DATASET}" 
#python3.7 "${WORK_DIR}"/getMetrics.py "${EVAL_LOGDIR}/Test1/" "${METRICS_LOGDIR}/Test/" 
#

#### Save visualization examples
# python3.7 "${WORK_DIR}"/vis.py \
#   --logtostderr \
#   --vis_split="Test1" \
#   --model_variant="${MODEL_NAME}" \
#   --atrous_rates=6 \
#   --atrous_rates=12 \
#   --atrous_rates=18 \
#   --output_stride=16 \
#   --decoder_output_stride=4 \
#   --vis_crop_size="481,641" \
#   --checkpoint_dir="${TRAIN_LOGDIR}" \
#   --vis_logdir="${VIS_LOGDIR}" \
#   --dataset_dir="${TF_DATASET}" \
#   --max_number_of_iterations=1 \
#   --dataset="${DATASET}"

#
### Export the trained checkpoint.
# CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
# EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"
#
# python3.7 "${WORK_DIR}"/export_model.py \
#   --logtostderr \
#   --checkpoint_path="${CKPT_PATH}" \
#   --export_path="${EXPORT_PATH}" \
#   --model_variant="${MODEL_NAME}" \
#   --atrous_rates=6 \
#   --atrous_rates=12 \
#   --atrous_rates=18 \
#   --output_stride=16 \
#   --decoder_output_stride=4 \
#   --num_classes=3 \
#   --crop_size=481 \
#   --crop_size=641 \
#   --inference_scales=1.0

### SAVE PREDICTED TEST
python3.7 "${WORK_DIR}"/detect_termo.py "${EXP_NAME}" "${EXPORT_DIR}" '/media/oscar/New Volume/DATASETS/TERMO/other_samples/images/' "${VIS_LOGDIR}/detections/" 
#
## GET CONFUSION MATRIX
python3.7 "${WORK_DIR}"/getConfusionMatrix_termo.py "${METRICS_LOGDIR}" '/media/oscar/New Volume/DATASETS/TERMO/other_samples/gt/' "${VIS_LOGDIR}/detections/" 

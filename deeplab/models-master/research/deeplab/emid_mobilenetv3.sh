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

# Go to datasets folder and download PASCAL VOC 2012 segmentation dataset.
DATASET_DIR="/media/oscar/New Volume/DATASETS/EMID"


# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
EXP_NAME='E003-02'
PASCAL_FOLDER="20210510/GIJ/All"
EXP_FOLDER="exp_OLD"
INIT_FOLDER="${DATASET_DIR}/${PASCAL_FOLDER}/init_models"
TRAIN_LOGDIR="${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/train/${EXP_NAME}"
EVAL_LOGDIR="${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/eval/${EXP_NAME}"
VIS_LOGDIR="${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/vis/${EXP_NAME}"
METRICS_LOGDIR="${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/metrics/${EXP_NAME}"
EXPORT_DIR="${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/export/${EXP_NAME}"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}/Train1"
mkdir -p "${EVAL_LOGDIR}/Test1"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${METRICS_LOGDIR}"
mkdir -p "${METRICS_LOGDIR}/Train1"
mkdir -p "${METRICS_LOGDIR}/Test1"
mkdir -p "${EXPORT_DIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="http://download.tensorflow.org/models"
#CKPT_NAME="deeplab_mnv3_small_cityscapes_trainfine"
CKPT_NAME="deeplab_mnv3_large_cityscapes_trainfine"

TF_INIT_CKPT="${CKPT_NAME}_2019_11_15.tar.gz"

MODEL_NAME='mobilenet_v3_large_seg'
MODEL_DIR="deeplab_mnv3_large_cityscapes_trainfine/model.ckpt" #"train_fine/model.ckpt" #

cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

PASCAL_DATASET="${DATASET_DIR}/${PASCAL_FOLDER}/tfrecord512x384"

# Train 10 iterations.
# 1 epoch = 25274/12 = 2106-7

#Start logging cpu/gpu data
# Una vez se comienza a entrenar se mide el rendimiento de la cpu y de la gpu cada 10 segundos
#mpstat -P ALL 10 > "${METRICS_LOGDIR}"/temp_cpu.csv &
#nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 10 > "${METRICS_LOGDIR}"/gpu_rendimiento.csv &

LEARNING_RATE=0.00005 
BATCH_SIZE=16
IMAGES_TRAINING=1125 #634
EPOCHS=300
NUM_ITERATIONS=$((IMAGES_TRAINING * EPOCHS / BATCH_SIZE)) #8428 #4214 #6320 #3160 #1580-20ep for 8batch
#
#####
# python3.7 "${WORK_DIR}"/train.py \
#   --logtostderr \
#   --train_split="Train1" \
#   --model_variant="${MODEL_NAME}" \
#   --optimizer='adam' \
#   --adam_learning_rate="${LEARNING_RATE}" \
#   --base_learning_rate="${LEARNING_RATE}"  \
#   --end_learning_rate="${LEARNING_RATE}"  \
#   --image_pooling_crop_size=385,513 \
#   --image_pooling_stride=4,5 \
#   --add_image_level_feature=1 \
#   --aspp_convs_filters=128 \
#   --aspp_with_concat_projection=0 \
#   --aspp_with_squeeze_and_excitation=1 \
#   --decoder_use_sum_merge=1 \
#   --decoder_filters=6 \
#   --decoder_output_is_logits=1 \
#   --image_se_uses_qsigmoid=1 \
#   --decoder_output_stride=8 \
#   --output_stride=8 \
#   --train_crop_size="385,513" \
#   --train_batch_size="${BATCH_SIZE}" \
#   --training_number_of_steps="${NUM_ITERATIONS}" \
#   --fine_tune_batch_norm=false \
#   --tf_initial_checkpoint="${INIT_FOLDER}/${MODEL_DIR}" \
#   --train_logdir="${TRAIN_LOGDIR}" \
#   --dataset_dir="${PASCAL_DATASET}" \
#   --dataset="emid-gij"
###### 

##Stop logging CPU/GPU data
##pkill mpstat
##pkill nvidia-smi
## Modificar los csvs de rendimiento para que sea más sencillos tratarlos con Excel
##python3.7 "${WORK_DIR}"/cpu_rend.py "${METRICS_LOGDIR}"/temp_cpu.csv "${METRICS_LOGDIR}"/cpu_rendimiento.csv # Adaptar el formato de lso datos de cpu para que sea más cómodo de tratar con excel
##sed -i -- 's/ %//g' "${METRICS_LOGDIR}"/gpu_rendimiento.csv # Quitar los % del fichero de la gpu que molestan, y vale con la cabecera
##sed -i -- 's/ MiB//g' "${METRICS_LOGDIR}"/gpu_rendimiento.csv # Quitar las unidades que luego para hacer medias y demás molesta
#
#
#
#### Evaluation on Train and Test datasets
 #python3.7 "${WORK_DIR}"/eval.py \
 #  --logtostderr \
 #  --eval_split="Train1" \
 #  --model_variant="${MODEL_NAME}"\
 #  --atrous_rates=12 \
 #  --atrous_rates=24 \
 #  --atrous_rates=36 \
 #  --output_stride=8 \
 #  --decoder_output_stride=4 \
 #  --eval_crop_size="385,513" \
 #  --checkpoint_dir="${TRAIN_LOGDIR}" \
 #  --eval_logdir="${EVAL_LOGDIR}" \
 #  --dataset_dir="${PASCAL_DATASET}" \
 #  --max_number_of_evaluations=1 \
 #  --dataset="emid"
##   

 #python3.7 "${WORK_DIR}"/eval.py \
 #  --logtostderr \
 #  --eval_split="Test1" \
 #  --model_variant="${MODEL_NAME}"\
 #  --atrous_rates=12 \
 #  --atrous_rates=24 \
 #  --atrous_rates=36 \
 #  --output_stride=8 \
 #  --decoder_output_stride=4 \
 #  --eval_crop_size="385,513" \
 #  --checkpoint_dir="${TRAIN_LOGDIR}" \
 #  --eval_logdir="${EVAL_LOGDIR}" \
 #  --dataset_dir="${PASCAL_DATASET}" \
 #  --max_number_of_evaluations=1 \
 #  --dataset="emid"
####

## python3.7 "${WORK_DIR}"/getMetrics.py "${EVAL_LOGDIR}/" "${METRICS_LOGDIR}/" expName
## #python3.7 "${WORK_DIR}"/getLoss.py "${TRAIN_LOGDIR}/" "${METRICS_LOGDIR}/" expName
##
#
#python3.7 "${WORK_DIR}"/runEval.py "${EXP_NAME}" "${MODEL_NAME}" "Test1"
#python3.7 "${WORK_DIR}"/runEval.py "${EXP_NAME}" "${MODEL_NAME}" "Train1"
#
#python3.7 "${WORK_DIR}"/getMetrics.py "${EVAL_LOGDIR}/Test1/" "${METRICS_LOGDIR}/Test1/" expName
#python3.7 "${WORK_DIR}"/getMetrics.py "${EVAL_LOGDIR}/Train1/" "${METRICS_LOGDIR}/Train1/" expName
#
#
### Visualize the results.
 #python3.7 "${WORK_DIR}"/vis.py \
 #  --logtostderr \
 #  --vis_split="Vis" \
 #  --model_variant="${MODEL_NAME}" \
 #  --output_stride=8 \
 #  --decoder_output_stride=8 \
 #  --image_se_uses_qsigmoid=1 \
 #  --decoder_output_is_logits=1 \
 #  --decoder_filters=6 \
 #  --decoder_use_sum_merge=1 \
 #  --aspp_with_squeeze_and_excitation=1 \
 #  --aspp_with_concat_projection=0 \
 #  --aspp_convs_filters=128 \
 #  --add_image_level_feature=1 \
 #  --image_pooling_stride=4,5 \
 #  --image_pooling_crop_size=385,513 \
 #  --vis_crop_size="385,513" \
 #  --checkpoint_dir="${TRAIN_LOGDIR}" \
 #  --vis_logdir="${VIS_LOGDIR}" \
 #  --dataset_dir="${PASCAL_DATASET}" \
 #  --max_number_of_iterations=1 \
 #  --dataset="emid-gij" 
#
### # Export the trained checkpoint.
 CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
 EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"
#
 python3.7 "${WORK_DIR}"/export_model.py \
   --logtostderr \
   --checkpoint_path="${CKPT_PATH}" \
   --export_path="${EXPORT_PATH}" \
   --model_variant="${MODEL_NAME}" \
   --output_stride=8 \
   --decoder_output_stride=8 \
   --image_se_uses_qsigmoid=1 \
   --decoder_output_is_logits=1 \
   --decoder_filters=6 \
   --decoder_use_sum_merge=1 \
   --aspp_with_squeeze_and_excitation=1 \
   --aspp_with_concat_projection=0 \
   --aspp_convs_filters=128 \
   --add_image_level_feature=1 \
   --image_pooling_stride=4,5 \
   --image_pooling_crop_size=385,513 \
   --num_classes=7 \
   --crop_size=385 \
   --crop_size=513 \
   --inference_scales=1.0
#
#
#
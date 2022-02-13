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
EXP_NAME='E003-03'
PASCAL_FOLDER="20210510/GIJ/All"
EXP_FOLDER="exp"
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

TF_INIT_CKPT="resnet_v1_101_2018_05_04.tar.gz"


#TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz" #xception65
#TF_INIT_CKPT="deeplabv3_xception_2018_01_04.tar.gz" #xception65 best?

#TF_INIT_CKPT="deeplab_cityscapes_xception71_trainfine_2018_09_08.tar.gz" #X71 worse 
#TF_INIT_CKPT="xception_71_2018_05_09.tar.gz" # X71 best

#TF_INIT_CKPT="resnet_v1_50_2018_05_04.tar.gz"
#TF_INIT_CKPT="xception_41_2018_05_09.tar.gz"

MODEL_NAME='resnet_v1_101_beta'
MODEL_DIR="resnet_v1_101/model.ckpt" #"train_fine/model.ckpt" #

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
BATCH_SIZE=8
IMAGES_TRAINING=1125 #634
EPOCHS=160
NUM_ITERATIONS=$((IMAGES_TRAINING * EPOCHS / BATCH_SIZE)) #8428 #4214 #6320 #3160 #1580-20ep for 8batch

####
 python3.7 "${WORK_DIR}"/train.py \
   --logtostderr \
   --train_split="Train1" \
   --model_variant="${MODEL_NAME}" \
   --optimizer='adam' \
   --adam_learning_rate="${LEARNING_RATE}" \
   --base_learning_rate="${LEARNING_RATE}"  \
   --end_learning_rate="${LEARNING_RATE}"  \
   --atrous_rates=6 \
   --atrous_rates=12 \
   --atrous_rates=18 \
   --output_stride=16 \
   --decoder_output_stride=4 \
   --train_crop_size="385,513" \
   --train_batch_size="${BATCH_SIZE}" \
   --training_number_of_steps="${NUM_ITERATIONS}" \
   --fine_tune_batch_norm=false \
   --tf_initial_checkpoint="${INIT_FOLDER}/${MODEL_DIR}" \
   --train_logdir="${TRAIN_LOGDIR}" \
   --dataset_dir="${PASCAL_DATASET}" \
   --dataset="emid-gij"
###### 
#
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

# python3.7 "${WORK_DIR}"/getMetrics.py "${EVAL_LOGDIR}/" "${METRICS_LOGDIR}/" expName
# #python3.7 "${WORK_DIR}"/getLoss.py "${TRAIN_LOGDIR}/" "${METRICS_LOGDIR}/" expName
#

python3.7 "${WORK_DIR}"/runEval.py "${EXP_NAME}" "${MODEL_NAME}" "Test1"
python3.7 "${WORK_DIR}"/runEval.py "${EXP_NAME}" "${MODEL_NAME}" "Train1"

python3.7 "${WORK_DIR}"/getMetrics.py "${EVAL_LOGDIR}/Test1/" "${METRICS_LOGDIR}/Test1/" expName
python3.7 "${WORK_DIR}"/getMetrics.py "${EVAL_LOGDIR}/Train1/" "${METRICS_LOGDIR}/Train1/" expName



## Visualize the results.
 python3.7 "${WORK_DIR}"/vis.py \
   --logtostderr \
   --vis_split="Vis" \
   --model_variant="${MODEL_NAME}" \
   --atrous_rates=6 \
   --atrous_rates=12 \
   --atrous_rates=18 \
   --output_stride=16 \
   --decoder_output_stride=4 \
   --vis_crop_size="385,513" \
   --checkpoint_dir="${TRAIN_LOGDIR}" \
   --vis_logdir="${VIS_LOGDIR}" \
   --dataset_dir="${PASCAL_DATASET}" \
   --max_number_of_iterations=1 \
   --dataset="emid-gij" 
#
### # Export the trained checkpoint.
# CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
# EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"
##
 python3.7 "${WORK_DIR}"/export_model.py \
   --logtostderr \
   --checkpoint_path="${CKPT_PATH}" \
   --export_path="${EXPORT_PATH}" \
   --model_variant="${MODEL_NAME}" \
   --atrous_rates=6 \
   --atrous_rates=12 \
   --atrous_rates=18 \
   --output_stride=16 \
   --decoder_output_stride=4 \
   --num_classes=7 \
   --crop_size=513 \
   --crop_size=385 \
   --inference_scales=1.0
#
#

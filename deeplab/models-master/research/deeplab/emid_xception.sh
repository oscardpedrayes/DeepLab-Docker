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
EXP_NAME='E006-03' #'YE002-01v2'
DATASET_DIR="/datasets/EMID/20220121/AVI/deeplab1/" #"/datasets/EMID/20210504/DOF/All/"

EXP_FOLDER="exp" #expLevels
INIT_FOLDER="${DATASET_DIR}/init_models"
TRAIN_LOGDIR="${DATASET_DIR}/${EXP_FOLDER}/train/${EXP_NAME}/"
EVAL_LOGDIR="${DATASET_DIR}/${EXP_FOLDER}/eval/${EXP_NAME}/"
VIS_LOGDIR="${DATASET_DIR}/${EXP_FOLDER}/vis/${EXP_NAME}/"
METRICS_LOGDIR="${DATASET_DIR}/${EXP_FOLDER}/metrics/${EXP_NAME}/"
EXPORT_DIR="${DATASET_DIR}/${EXP_FOLDER}/export/${EXP_NAME}/"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${VIS_LOGDIR}/vis"
mkdir -p "${METRICS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="http://download.tensorflow.org/models"

#TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz" #xception65
TF_INIT_CKPT="deeplabv3_xception_2018_01_04.tar.gz" #xception65 best?

#TF_INIT_CKPT="deeplab_cityscapes_xception71_trainfine_2018_09_08.tar.gz" #X71 worse 
TF_INIT_CKPT="xception_71_2018_05_09.tar.gz" # X71 best

#TF_INIT_CKPT="resnet_v1_50_2018_05_04.tar.gz"
#TF_INIT_CKPT="xception_41_2018_05_09.tar.gz"

MODEL_NAME='xception_71'
MODEL_DIR="xception/model.ckpt" #"train_fine/model.ckpt" #

cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

INITIAL="${INIT_FOLDER}/${MODEL_DIR}" 
#If you want to reuse an old experiment
#EXP_NAME_CP='E012-02'
#INITIAL="${DATASET_DIR}/${EXP_FOLDER}/train/${EXP_NAME_CP}/model.ckpt-8454"


TF_DATASET="${DATASET_DIR}/tfrecords/" #tfrecordbinary512x384/" #tfrecordmulti512x384

DATASET='emid-avi3-multi5-day' #'emid-avi3-multi4-night' #emid #-avi-multi'
LEARNING_RATE=0.00005
BATCH_SIZE=10
IMAGES_TRAINING=970 #428 #970 #634 #764 #3750 #2250 #1500 #1125 #634 #1125
EPOCHS=300
NUM_ITERATIONS=$((IMAGES_TRAINING * EPOCHS / BATCH_SIZE))

#### TRAIN ###
python "${WORK_DIR}"/train.py \
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
  --fine_tune_batch_norm=true \
  --tf_initial_checkpoint="${INITIAL}"  \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${TF_DATASET}" \
  --dataset="${DATASET}" 

 #--min_scale_factor=1.0  \
  #--max_scale_factor=1.0  \
  #--scale_factor_step_size=0  \
#### TESTING ###

#Levels

#python "${WORK_DIR}"/runEval.py "${EXP_NAME}" "${MODEL_NAME}" "Test1" "${DATASET}" "${DATASET_DIR}/" "${TF_DATASET}"  "${EXP_FOLDER}" 
#python "${WORK_DIR}"/getMetrics.py "${EVAL_LOGDIR}/Test1/" "${METRICS_LOGDIR}/Test1/" 
#########

python "${WORK_DIR}"/runEval.py "${EXP_NAME}" "${MODEL_NAME}" "Test1" "${DATASET}" "${DATASET_DIR}/" "${TF_DATASET}" "${EXP_FOLDER}"
python "${WORK_DIR}"/getMetrics.py "${EVAL_LOGDIR}/Test1/" "${METRICS_LOGDIR}/Test1/"
python "${WORK_DIR}"/runEval.py "${EXP_NAME}" "${MODEL_NAME}" "Train1" "${DATASET}" "${DATASET_DIR}/" "${TF_DATASET}" "${EXP_FOLDER}"
python "${WORK_DIR}"/getMetrics.py "${EVAL_LOGDIR}/Train1/" "${METRICS_LOGDIR}/Train1/"




#
#python "${WORK_DIR}"/runEval.py "${EXP_NAME}" "${MODEL_NAME}" "Test1_1x4" "${DATASET}" "${DATASET_DIR}/" "${TF_DATASET}" 
#python "${WORK_DIR}"/getMetrics.py "${EVAL_LOGDIR}/Test1_1x4/" "${METRICS_LOGDIR}/Test1_1x4/" 

#python "${WORK_DIR}"/runEval.py "${EXP_NAME}" "${MODEL_NAME}" "Real_test1" "${DATASET}" "${DATASET_DIR}/" "${TF_DATASET}" 
#python "${WORK_DIR}"/getMetrics.py "${EVAL_LOGDIR}/Real_test1/" "${METRICS_LOGDIR}/Real_test1/"

#python "${WORK_DIR}"/runEval.py "${EXP_NAME}" "${MODEL_NAME}" "Real_test1_SuperSaneado" "${DATASET}" "${DATASET_DIR}/" "${TF_DATASET}" 
#python "${WORK_DIR}"/getMetrics.py "${EVAL_LOGDIR}/Real_test1_SuperSaneado/" "${METRICS_LOGDIR}/Real_test1_SuperSaneado/"
##
#python"${WORK_DIR}"/runEval.py "${EXP_NAME}" "${MODEL_NAME}" "Train1" "${DATASET}" "${DATASET_DIR}/" "${TF_DATASET}" 
#python "${WORK_DIR}"/getMetrics.py "${EVAL_LOGDIR}/Train1/" "${METRICS_LOGDIR}/Train1/" 
######
#python "${WORK_DIR}"/runEval.py "${EXP_NAME}" "${MODEL_NAME}" "Test1" "${DATASET}" "${DATASET_DIR}/" "${TF_DATASET}" 
#python "${WORK_DIR}"/getMetrics.py "${EVAL_LOGDIR}/Test1/" "${METRICS_LOGDIR}/Test1/" 
##


##python "${WORK_DIR}"/runEval.py "${EXP_NAME}" "${MODEL_NAME}" "Train1_1x4" "${DATASET}" "${DATASET_DIR}/" "${TF_DATASET}" 
##python "${WORK_DIR}"/getMetrics.py "${EVAL_LOGDIR}/Train1_1x4/" "${METRICS_LOGDIR}/Train1_1x4/" 
#
#
#
##
##### Save visualization examples
# python "${WORK_DIR}"/vis.py \
#   --logtostderr \
#   --vis_split="Vis" \
#   --model_variant="${MODEL_NAME}" \
#   --atrous_rates=6 \
#   --atrous_rates=12 \
#   --atrous_rates=18 \
#   --output_stride=16 \
#   --decoder_output_stride=4 \
#   --vis_crop_size="385,513" \
#   --checkpoint_dir="${TRAIN_LOGDIR}" \
#   --vis_logdir="${VIS_LOGDIR}" \
#   --dataset_dir="${TF_DATASET}" \
#   --max_number_of_iterations=1 \
#   --dataset="${DATASET}"

##
#### Export the trained checkpoint.
 #NUM_ITERATIONS=30000 #48558
 CKPT_PATH="${TRAIN_LOGDIR}model.ckpt-${NUM_ITERATIONS}"
 EXPORT_PATH="${EXPORT_DIR}/${EXP_NAME}.pb"
#
python "${WORK_DIR}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="${MODEL_NAME}" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=6 \
  --crop_size=385 \
  --crop_size=513 \
  --inference_scales=1.0
#
#### SAVE PREDICTED TEST
##python "${WORK_DIR}"/detect.py "${EXP_NAME}" "${EXPORT_DIR}" '/datasets/EMID/20210504/AVI/All/images_test512x384/' "${VIS_LOGDIR}/detections/" 
python "${WORK_DIR}"/detect.py "${EXP_NAME}" "${EXPORT_DIR}" '/datasets/EMID/20220121/AVI/test/RGB/' "${VIS_LOGDIR}/detections/" 
python "${WORK_DIR}"/detect.py "${EXP_NAME}" "${EXPORT_DIR}" '/datasets/EMID/20220121/AVI/vis/RGB/' "${VIS_LOGDIR}/vis/" 
#
##
### GET CONFUSION MATRIX
##python "${WORK_DIR}"/getConfusionMatrix.py "${METRICS_LOGDIR}" '/datasets/EMID/20210504/AVI/All/maskmulti_test512x384/' "${VIS_LOGDIR}/detections/" 
python "${WORK_DIR}"/getConfusionMatrix.py "${METRICS_LOGDIR}" '/datasets/EMID/20220121/AVI/test/masks_gray_1channel/' "${VIS_LOGDIR}/detections/"
import os
import subprocess
import sys

#Command to run evaluation
def getEvalCMD(WORK_DIR, MODEL_NAME, TRAIN_LOGDIR, EVAL_LOGDIR, TF_RECORD_DIR, SPLIT, DATASET):
    if "mobilenet_v3" in MODEL_NAME:
        return '''
        python "''' + WORK_DIR +'''/eval.py" \
        --logtostderr \
        --eval_split="''' + SPLIT +'''" \
        --model_variant="''' + MODEL_NAME +'''" \
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
        --eval_crop_size="385,513" \
        --checkpoint_dir="''' + TRAIN_LOGDIR +'''" \
        --eval_logdir="''' + EVAL_LOGDIR +'''" \
        --dataset_dir="''' + TF_RECORD_DIR +'''" \
        --max_number_of_evaluations=1 \
        --dataset="''' + DATASET +'''" 
        '''
    else:
        return '''
        python "''' + WORK_DIR +'''/eval.py" \
        --logtostderr \
        --eval_split="''' + SPLIT +'''" \
        --model_variant="''' + MODEL_NAME +'''"\
        --atrous_rates=6 \
        --atrous_rates=12 \
        --atrous_rates=18 \
        --output_stride=16 \
        --decoder_output_stride=4 \
        --eval_crop_size="385,513" \
        --checkpoint_dir="''' + TRAIN_LOGDIR +'''" \
        --eval_logdir="''' + EVAL_LOGDIR +'''" \
        --dataset_dir="''' + TF_RECORD_DIR +'''" \
        --max_number_of_evaluations=1 \
        --dataset="''' + DATASET +'''" 
        '''

#Get all checkpoints
def getCheckpointsList(TRAIN_LOGDIR):
    cplist = []
    with open(TRAIN_LOGDIR+'/original.checkpoint', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'all_model_checkpoint_paths' in line:
                cplist.append(line.split(':')[1].replace('"','').replace('\n','').strip())
    return cplist

# Need to rewrite the checkpoint if i want to to change the checkpoint to run
def generateCheckpoint(checkpoint, TRAIN_LOGDIR):
    model = 'model_checkpoint_path: "'+checkpoint+'"\nall_model_checkpoint_paths: "'+checkpoint+'"\n'
    try:
        with open(TRAIN_LOGDIR+'/checkpoint','w') as f:
            f.write(model)
        print('Checkpoint edited to: ' + model)
    except:
        print('ERROR: Checkpoint was not edited')



    

WORK_DIR=os.path.dirname(os.path.abspath(__file__))

EXP_NAME= sys.argv[1]+"/"
MODEL_NAME= sys.argv[2]
SPLIT= sys.argv[3]
DATASET = sys.argv[4]
DATASET_DIR= sys.argv[5]

TF_RECORD_DIR = sys.argv[6]

EXP_FOLDER = sys.argv[7]
TRAIN_LOGDIR= DATASET_DIR + EXP_FOLDER + "/train/" + EXP_NAME
EVAL_LOGDIR= DATASET_DIR + EXP_FOLDER + "/eval/" + EXP_NAME + "/" + SPLIT +"/"

try:
    os.mkdir(EVAL_LOGDIR)

except OSError as error:
    print(error)

try:
    if os.path.isfile(TRAIN_LOGDIR+'/original.checkpoint'):
        print('Checkpoint was not renamed. Already renamed.')
    else:
        os.rename(TRAIN_LOGDIR+'/checkpoint',TRAIN_LOGDIR+'/original.checkpoint' )
        print('Original checkpoint renamed to original.checkpoint.')
except:
    print('Error renaming the checkpoint file.')


checkpointsList = getCheckpointsList(TRAIN_LOGDIR)

for cp in checkpointsList:
    generateCheckpoint(cp, TRAIN_LOGDIR)
    command = getEvalCMD(WORK_DIR, MODEL_NAME, TRAIN_LOGDIR, EVAL_LOGDIR, TF_RECORD_DIR, SPLIT, DATASET)
    print(command)
    subprocess.run(command,shell=True)
  

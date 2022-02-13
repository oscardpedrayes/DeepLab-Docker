import tensorflow as tf
import sys 
import os

if len(sys.argv)-1 < 3:
    print("Wrong usage. getMetricsTrain.py DirToRead DirToSave ExpName")
    exit()

expName= sys.argv[3]
dir_data = sys.argv[1]
dir_output = sys.argv[2]
print(expName,'\n',dir_data,'\n',dir_output)
#dir_data = '/media/oscar/New Volume/DATASETS/PNOA/20200902/exp/train_on_trainval_set_xception65/train/E002-01/'
#dir_output ='/media/oscar/New Volume/DATASETS/PNOA/20200902/exp/train_on_trainval_set_xception65/metrics/E002-01/'

for fil in os.listdir(dir_data):
    if fil.endswith(".atc61"):
        # Create empty file
        print('found')
        open(dir_output + "loss.csv","w").close()

        # Iterate throught tags, get global metrics first
        for e in tf.train.summary_iterator(dir_data + fil):
            for v in e.summary.value:
                #print(v.tag)
                if 'total_loss' in v.tag :
                    print(v)
                    with open(dir_output + "loss.csv", 'a') as f:
                        print(str(v.simple_value), file=f)
import tensorflow as tf
import sys 
import os

if len(sys.argv)-1 < 2:
    print("Wrong usage. getMetrics.py DirToRead DirToSave")
    exit()

dir_data = sys.argv[1]
dir_output = sys.argv[2]
print(dir_data,'\n',dir_output)

try:
    os.mkdir(dir_output)
except:
    print('Eval directory already created.')

for fil in os.listdir(dir_data):
    #if fil.endswith(".atc61"):
    # Create empty file
    open(dir_output + fil,"w").close()
    # Iterate throught tags, get global metrics first
    for e in tf.train.summary_iterator(dir_data + fil):
        for v in e.summary.value:
            if 'class' not in v.tag :
                with open(dir_output + fil, 'a') as f:
                    print(v.tag + ' ' + str(v.simple_value), file=f)

        oldClass = -1
        row = ''
        # Iterate throught tags, get all class metrics
        for e in tf.train.summary_iterator(dir_data + fil):
            for v in e.summary.value:
                if 'class' in v.tag:      
                    classNumber = int(v.tag.split('_')[-1])
                    if classNumber>oldClass:
                        row = row + '\n' + str(classNumber) + ' ' + str(v.simple_value) + ' '
                    else:
                        row = row + ' ' + str(v.simple_value)
                    oldClass = classNumber
            if row.strip():
                with open(dir_output +  fil, 'a') as f:
                    print(row, file=f)

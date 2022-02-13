import os

List1 = []
Path1= '/media/oscar/New Volume/DATASETS/EMID/20210526/AVI/All/multi.txt'
with open(Path1,'r') as f:
    for line in f:
        List1.append(line)

List2 = []
Path2= '/media/oscar/New Volume/DATASETS/EMID/20210526/AVI/All/index/multi/Train1.txt'
with open(Path2,'r') as f:
    for line in f:
        List2.append(line)

textfile= open("/media/oscar/New Volume/DATASETS/EMID/20210526/AVI/All/index/multi/Train1_.txt",'w')              
for line in List1:
    if line in List2:
        textfile.write(line)
textfile.close()
print('done.')



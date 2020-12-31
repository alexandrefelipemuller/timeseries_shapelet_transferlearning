import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

#USAGE command shapelets original_serie
x_train, y_train = readucr(sys.argv[1])
x_train2, y_train2 = readucr(sys.argv[2])

x_train3 = []
i=0
for example in x_train:
    x_train3.append([])
    j=0
    x_train3[i].append(y_train2[i])
    for val in example:
        if val != 0.5:
            x_train3[i].append(random.uniform(-1,1))
        else:
            try:
                x_train3[i].append(x_train2[i][j])
            except:
                x_train3[i].append(random.uniform(-1,1))
                continue
        j=j+1
    i=i+1

fout=open("OUTPUT.txt","w+")
i=0
for data in x_train3:
    fout.write("%s\r\n" % str(list(data)[0:-1]))
    i=i+1

fout.close()


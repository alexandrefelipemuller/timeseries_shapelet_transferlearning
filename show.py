import sys
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

x_train, y_train = readucr(sys.argv[1])
i=0
for val in x_train:
    plt.plot(val)
    #plt.show()
    plt.savefig('imgs/'+str(i)+'.png')
    i=i+1


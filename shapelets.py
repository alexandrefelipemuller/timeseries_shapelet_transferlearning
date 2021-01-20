import itertools
import numpy as np
import os
import matplotlib.image as mpimg       # reading images to numpy arrays
import keras
import matplotlib.pyplot as plt        # to plot any graph
import sys
from numba import njit, prange

import scipy.ndimage as ndi            # to determine shape centrality

from pylab import rcParams
rcParams['figure.figsize'] = (6, 6)      # setting default size of plots

@njit(nopython=True, parallel=True)
def generate_candidates(data, max_len=5, min_len=2):
    step = 1
    candidates, l = [], max_len
    while l >= min_len:
        for i in range(len(data)):
            time_serie, label = data[i][0], data[i][1]
            for k in range(len(time_serie)-l+1): candidates.append((time_serie[k:k+l], label,k))
        l -= step
    return candidates

def check_candidate(data, shapelet, verbose):
    histogram = {}
    for entry in data:
        try:
            # TODO: entropy pre-pruning in each iteration
            time_serie, label = entry[0], entry[1]
            d, idx = subsequence_dist(time_serie, shapelet)
        except:
            continue
        if d is not None:
            try:
                histogram[d] = [(time_serie, label)] if d not in histogram else histogram[d].append((time_serie, label))
            except:
                if verbose:
                    print("iterate problem with", d)
                return -1,-1,-1,-1
    return find_best_split_point(histogram)

def calculate_dict_entropy(data):
    counts = {}
    for entry in data:
        if entry[1] in counts: counts[entry[1]] += 1
        else: counts[entry[1]] = 1
    return calculate_entropy(np.divide(list(counts.values()), float(sum(list(counts.values())))))

def find_best_split_point(histogram):
    try:
        hv=list(histogram.values())
        hvl=itertools.chain.from_iterable(hv)
        histogram_values = list(hvl)
    except:
        return -1,-1,-1,-1
    prior_entropy = calculate_dict_entropy(histogram_values)
    best_distance, max_ig = 0, 0
    best_left, best_right = None, None
    for distance in histogram:
        data_left = []
        data_right = []
        for distance2 in histogram:
            if distance2 <= distance: data_left.extend(histogram[distance2])
            else: data_right.extend(histogram[distance2])
        ig = prior_entropy - (float(len(data_left))/float(len(histogram_values))*calculate_dict_entropy(data_left) + \
             float(len(data_right))/float(len(histogram_values)) * calculate_dict_entropy(data_right))
        if ig > max_ig: best_distance, max_ig, best_left, best_right = distance, ig, data_left, data_right
    return max_ig, best_distance, best_left, best_right

@njit(nopython=True, parallel=True)
def manhattan_distance(a, b, min_dist=float('inf')):
    dist = 0
    for x, y in zip(a, b):
        dist += np.abs(float(x)-float(y))
        if dist >= min_dist: return None
    return dist

def calculate_entropy(probabilities):
    return sum([-prob * np.log(prob)/np.log(2) if prob != 0 else 0 for prob in probabilities])


def subsequence_dist(time_serie, sub_serie):
    if len(sub_serie) < len(time_serie):
        try:
            min_dist, min_idx = float("inf"), 0
            for i in range(len(time_serie)-len(sub_serie)+1):
                dist = manhattan_distance(sub_serie, time_serie[i:i+len(sub_serie)], min_dist)
                if dist is not None and dist < min_dist: min_dist, min_idx = dist, i
            return min_dist, min_idx
        except:
            return None, None
        else:
            return None, None

def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

def find_k_best_shapelets(data, max_len=100, min_len=1, k=3, plot=True, verbose=True):
    candidates = generate_candidates(data, max_len, min_len)
    bsf_gain, bsf_shapelet = 0, None
    dlist = []
    if verbose: candidates_length = len(candidates)
    for idx, candidate in enumerate(candidates):
        index_candidate = candidate[2]
        gain, dist, data_left, data_right = check_candidate(data, candidate[0],verbose)
        dlist.append({'gain': gain,'candidate': candidate[0], 'index_candidate': index_candidate})
        if verbose: print(idx, '/', candidates_length, ":", gain, dist)
    sorteddlist = sorted(dlist, key = lambda i: i['gain'], reverse=True)
    most_relevant_shapelet=sorteddlist[0] 
    result = []
    max_shapelets = k
    num_shapelets = 1

    result.append(sorteddlist[0]['candidate'])
    for shapelet in sorteddlist:
        if (num_shapelets >= max_shapelets):
            break;
        #If no intersection
        if (int(shapelet['index_candidate'])+len(shapelet['candidate'])) <  int(most_relevant_shapelet['index_candidate']) or int(most_relevant_shapelet['index_candidate'])+len(most_relevant_shapelet['candidate']) < int(shapelet['index_candidate']):
            result.append(shapelet['candidate'])
            num_shapelets = num_shapelets+1
    print("Serie with ",num_shapelets," shapelets")
    return result

def extract_shapelets(data, min_len=10, max_len=100, verbose=1):
    _classes = np.unique([x[1] for x in data])
    kshapelets = {}
    for _class in _classes:
        print('Extracting shapelets for', _class)
        transformed_data = []
        for entry in data:
            time_serie, label = entry[0], entry[1]
            if label == _class: transformed_data.append((time_serie, 1))
            else: transformed_data.append((time_serie, 0))
        kshapelets[_class] = find_k_best_shapelets(transformed_data, max_len=max_len, min_len=min_len, k=2, plot=False, verbose=verbose)
    return kshapelets

data1 = []
fname = sys.argv[1]
_dist=0
_idx=0
print("_dist" + str(_dist) + " _idx = " + str(_idx))

x_train, y_train = readucr(fname+'/'+fname+'_TRAIN')
x_test, y_test = readucr(fname+'/'+fname+'_TEST')

import csv

with open(fname+'/'+fname+'_TRAIN') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    i = 0
    for row in csv_reader:
        data1.append((np.asarray(list(map(float,row))),int(y_train[i])))
        i+=1

nb_classes = len(np.unique(y_test))

batch_size = min(x_train.shape[0]/10, 16)
   
y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)

 
Y_train = keras.utils.to_categorical(y_train, nb_classes)
Y_test = keras.utils.to_categorical(y_test, nb_classes)
 
x_train_mean = x_train.mean()
x_train_std = x_train.std()
x_train = (x_train - x_train_mean)/(x_train_std)
 
x_test = (x_test - x_train_mean)/(x_train_std)
x_train = x_train.reshape(x_train.shape + (1,1,))
x_test = x_test.reshape(x_test.shape + (1,1,))
print ("class:"+fname+", number of classes: "+str(nb_classes))
x = keras.layers.Input(x_train.shape[1:])

print("lenght:"+str(len(data1)))

for val in data1:
    print ("y class: " + str(val))

shapelet_dict = extract_shapelets(data1, min_len=6, max_len=18, verbose=0)

def cutSerie(_idx, shapelet, origSerie, shapeletResult):
    ShapeletSize = len(shapelet)
    newSerie = []
    newSerie.extend(shapeletResult[0:_idx])
    newSerie.extend(origSerie[_idx:_idx+ShapeletSize])
    newSerie.extend(shapeletResult[_idx+ShapeletSize:])
    return newSerie

def cutInvertedSerie(_idx, shapelet, origSerie):
    TSsize = len(origSerie)
    ShapeletSize = len(shapelet)
    supressed = [0.5] * (ShapeletSize) 
    newSerie = []
    newSerie.extend(origSerie[1:_idx])
    newSerie.extend(supressed)
    newSerie.extend(origSerie[_idx+ShapeletSize:])
    return newSerie

fout=open(fname + "_result_series_shapelet.txt","w+")
fout_inverted=open(fname + "_result_series_no_shapelet.txt","w+")
i=0
result=[]
invertedSerie=[]
for data in data1:
    index=data[0][0]
    kshapelets = shapelet_dict[index]
    print ("CLASSE: " + str(index))
    invertedSerie=data[0]
    shapeletResult=[0.5] * (len(data[0]))
    k=0
    for shapelet in kshapelets:
        print("shapelet "+str(k))
        k=k+1
        _dist, _idx = subsequence_dist(data[0], shapelet)
        shapeletResult = cutSerie(_idx, shapelet, data[0], shapeletResult)
        invertedSerie = cutInvertedSerie(_idx, shapelet, invertedSerie)
    result = []
    fout.write("%d, %s\r\n" % (index, str(list(shapeletResult))[1:]))
    fout_inverted.write("%d, %s\r\n" % (index, str(list(invertedSerie))[1:]))
    i=i+1

fout.close()
fout_inverted.close()


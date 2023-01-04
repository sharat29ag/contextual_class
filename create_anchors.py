
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
from collections import Counter
import glob
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.stats import entropy


def get_arguments():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--file",type=str,help="path to text file")
    parser.add_argument("--features",type=str,help="path to features")
    return parser.parse_args()

args = get_arguments()

def Kmeans(x,n):
	return KMeans(n_clusters = n,random_state = 0).fit(x)

file_selected = open(args.file)
path = args.features+'/'

temp = file_selected.readlines()
names = []
for i in temp:
    i = i.strip()
    names.append(i.split('/')[-1].split('.')[0])

clusters = []
for j in range(19): ## number of classes for cityscapes = 19
    temp = []
    for i in names:
        f = np.load(path+i+'.npy')
        temp.append(f[j])
    temp = np.asarray(temp)
    kmeans = Kmeans(temp,1).cluster_centers_
    clusters.append(kmeans)

clusters = np.asarray(clusters)
torch.save(clusters,'anchors.npy')





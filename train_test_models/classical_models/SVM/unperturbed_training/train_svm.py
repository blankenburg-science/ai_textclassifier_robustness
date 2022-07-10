import time
import sys
import random
import copy
import os
import inflect
import re
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pickle as pkl

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

import pandas as pd
import seaborn as sns

import src.data_helper as data_helper
import src.perturbations_func as perturbations_func

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

data_set = 'ag_news'
data_folder_name = 'datasets/' + data_set + '/raw/'
N_gram = 2
N_features = 100
number_of_trials = 5
c_value = 2**15

samples = 5

random_seed = int(sys.argv[1])
print("seed:"+str(random_seed))

train_data = pd.read_csv(data_folder_name + 'train_'+str(random_seed)+'.csv', header=None)
test_data = pd.read_csv(data_folder_name + 'test_'+str(random_seed)+'.csv', header=None)

y_train_original = np.array(train_data[0]).reshape(len(train_data[0]), 1)
X_train_original = list(train_data[2])
X_train_original = [data_helper.basic_text_cleaner(text) for text in X_train_original]

y_test_original = np.array(test_data[0]).reshape(len(test_data[0]), 1)
X_test_original = list(test_data[2])
X_test_original = [data_helper.basic_text_cleaner(text) for text in X_test_original]

## Vectorize Data and Reduce Dimensions
vectorizer_x, svd, X_train_original_reduced, X_test_original_reduced = data_helper.TFIDF_SVD(N_gram, N_features, X_train_original, X_test_original)

scale = 1.0
## Train Model ##
X = X_train_original_reduced[0:int(scale*len(X_train_original_reduced))]
y = y_train_original[0:int(scale*len(X_train_original_reduced))]

X2 = X_test_original_reduced[0:int(scale*len(X_train_original_reduced))]
y2 = y_test_original[0:int(scale*len(X_train_original_reduced))]

start = time.time()
clf = OneVsRestClassifier(SVC(kernel='rbf', probability=True, random_state = random_seed))
clf.fit(X, y)
end = time.time()
print("Single SVC " + str(end - start)+" "+str(clf.score(X,y)))
print("Single SVC " + str(end - start)+" "+str(clf.score(X2,y2)))

# save model
models_dir = 'results/' + data_set + '/models/'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
pkl.dump(clf, open(models_dir + 'svm_' + str(random_seed) + '.pkl', 'wb'))
pkl.dump([vectorizer_x, svd], open(models_dir + 'vecto_svd_' + str(random_seed) + '.pkl', 'wb'))

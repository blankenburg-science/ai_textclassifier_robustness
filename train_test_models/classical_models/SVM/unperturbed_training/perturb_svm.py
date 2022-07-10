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

data_set = 'db_pedia'
data_folder_name = 'datasets/' + data_set + '/raw/'
N_gram = 2
N_features = 300
samples = 5

perturbation_prob_vec=np.append(np.array(range(0,99,10)),99)
attack_list = ["HomoPhones", "SimilarSymbols", "NeighborKeyboard"]
df_results_original = pd.DataFrame({'Attack': [], 'Mode': [], 'Sample': [], 'Random_Seed': [], 'Perturbation_Strength': [], 'ACC': []})

models_dir = 'results/' + data_set + '/models/'
results_dir = 'results/' + data_set + '/unperturbed/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

random_seed = int(sys.argv[1])
print("random seed: " + str(random_seed))

test_data = pd.read_csv(data_folder_name + 'test_'+str(random_seed)+'.csv', header=None)
y_test_original = np.array(test_data[0]).reshape(len(test_data[0]), 1)[0:10]
X_test_original = list(test_data[2])[0:10]
X_test_original = [data_helper.basic_text_cleaner(text) for text in X_test_original]

## Load SVD and Vectorizer
[vectorizer_x, svd] = pkl.load(open(models_dir + 'vecto_svd_' + str(random_seed) + '.pkl', 'rb'))

## Load Model
svm = pkl.load(open(models_dir + 'svm_' + str(random_seed) + '.pkl', 'rb'))


for sample in range(samples):
   for attack_idx in range(len(attack_list)):	
	    filename = results_dir + "results_unperturbed_svm_"+ data_set + "_seed_"+ str(random_seed) + ".pkl"
	    df_results_log_temp = data_helper.robustness_curve(filename, svm, 'svm' , vectorizer_x, svd, perturbation_prob_vec, X_test_original, y_test_original, sample, attack_list, attack_idx, random_seed)

	    df_results_original = df_results_original.append(df_results_log_temp)
	    df_results_original = df_results_original.reset_index().drop("index", axis = 1)

	    suitcase = [df_results_original]
	    pkl.dump(suitcase, open(results_dir + "results_unperturbed_svm_"+ data_set + "_seed_"+ str(random_seed) + ".pkl","wb"), protocol = 4 )

	    df_results_original = df_results_original.reset_index().drop("index", axis = 1)

## Save ##
suitcase = [df_results_original]
pkl.dump(suitcase, open(results_dir + "results_unperturbed_svm_"+ data_set + "_seed_"+ str(random_seed) + ".pkl","wb"), protocol = 4 )
print("Saved")

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

oxford_dict = pkl.load(open("homophone_dict.pkl", "rb"))

data_set = 'ag_news'
data_folder_name = 'datasets/' + data_set + '/raw/'
models_dir = 'results/' + data_set + '/models/'
vec_svd_dir = 'results_adv/' + data_set + '/models/'

perturbation_prob_vec=np.append(np.array(range(0,99,10)),99)
samples = 5
attack_list = ["HomoPhones", "NeighborKeyboard", "SimilarSymbols"]

adv_prob = 50
adv_perturbation_strength = 50
N_features_adv = 300
N_gram_adv = 2

df_results_original = pd.DataFrame({'Attack': [], 'Mode': [], 'Sample': [], 'Random_Seed': [], 
                                    'Perturbation_Strength': [], 'ACC': []})
df_results_adv = pd.DataFrame({'Perturbation_Train': [], 'Perturbation_Test': [], 'Mode': [], 'Sample': [], 'Random_Seed': [], 
                                    'Perturbation_Strength': [], 'ACC': []})

random_seed = int(sys.argv[1])
test_data = pd.read_csv(data_folder_name + 'test_'+str(random_seed)+'.csv', header=None)

y_test_original = np.array(test_data[0]).reshape(len(test_data[0]), 1)
X_test_original = list(test_data[2])
X_test_original = [data_helper.basic_text_cleaner(text) for text in X_test_original]

results_dir = 'results_adv/' + data_set + '/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
# Checking for Robustness of perturbed trained models #

for attack_train_idx in range(len(attack_list)):
    attack_name_train = attack_list[attack_train_idx]


    # load pretrained model
    clf = pkl.load(open(models_dir + 'svm_adv_seed_' + str(random_seed) + '_'+ attack_name_train + '_' + 
                       str(adv_prob) + '_' + str(adv_perturbation_strength) + '.pkl', 'rb'))


    # load vectorizer and svd
    [vectorizer_adv, svd_adv] = pkl.load(open(models_dir + 'svm_vectorizer_svd_adv_seed_' + str(random_seed) +
                        '_'+ attack_name_train + '_' + str(adv_prob) + '_' + str(adv_perturbation_strength) + '.pkl', 'rb'))

    for sample in range(samples):
        for attack_test_idx in range(len(attack_list)):
            
            df_results_adv_log_temp = data_helper.robustness_curve_adv(attack_name_train, clf, 'svm_adv' , vectorizer_adv, svd_adv, perturbation_prob_vec, X_test_original, y_test_original, sample, attack_list, attack_test_idx, random_seed)

            df_results_adv = df_results_adv.append(df_results_adv_log_temp)
            df_results_adv = df_results_adv.reset_index().drop("index", axis = 1)

            ## Save ##
            suitcase = [df_results_adv]
            pkl.dump(suitcase, open(results_dir + "df_results_agnews_svm_adv_seed_"+ str(random_seed) + "_" + str(adv_prob) + '_' + str(adv_perturbation_strength) + ".pkl","wb") )
    del clf
    del vectorizer_adv
    del svd_adv
    print("Results saved")


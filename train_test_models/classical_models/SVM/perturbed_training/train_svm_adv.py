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

def create_adv_data(adv_prob, adv_perturbation_strength, X_train_original_text, y_train_original_flat, function_list, function_id):

    attack_name = function_list[function_id]
    
    X_train_text_tmp = copy.deepcopy(X_train_original_text)
    y_train_flat_tmp = copy.deepcopy(y_train_original_flat)

    vec = np.array(range(len(X_train_text_tmp)))
    np.random.shuffle(vec)

    X_train_text_adv = list(np.array(X_train_text_tmp)[vec[0:int((len(X_train_text_tmp)-1) * adv_prob/100.0)]])
    y_train_flat_adv = list(np.array(y_train_flat_tmp)[vec[0:int((len(y_train_flat_tmp)-1) * adv_prob/100.0)]])
    
    #X_train_text_adv_perturbed = perturb_text(X_train_text_adv, adv_prob, function_list, function_id)
    # perturb
    X_train_text_adv_perturbed = copy.deepcopy(X_train_text_adv)
    inputs = range(len(X_train_text_adv_perturbed))

    if attack_name == "HomoPhones":
        X_train_text_adv_perturbed = data_helper.perturb_text_homophones(oxford_dict, X_train_text_adv, adv_perturbation_strength, function_list, function_id)

    elif attack_name == "SimilarSymbols":
        X_train_text_adv_perturbed = Parallel(n_jobs=num_cores)(delayed(data_helper.perturb_text)([X_train_text_adv[i]], adv_perturbation_strength, [perturbations_func.SubstituteSimilarSymbolsFunction], 0) for i in inputs)

    elif attack_name == "NeighborKeyboard":
        X_train_text_adv_perturbed = Parallel(n_jobs=num_cores)(delayed(data_helper.perturb_text)([X_train_text_adv[i]], adv_perturbation_strength, [perturbations_func.SubstituteNeighborKeyboardFunction], 0) for i in inputs)

    # flatten
    X_train_text_adv_perturbed_flat = []
    for item in X_train_text_adv_perturbed:
        if isinstance(item, list):
            item = item[0]
        X_train_text_adv_perturbed_flat.append(item)
    
    X_train_text_adv = X_train_text_tmp + X_train_text_adv_perturbed_flat

    y_train_adv = []

    for i in range(len(X_train_text_tmp)):
        y_train_adv.append(y_train_flat_tmp[i])

    for i in range(len(X_train_text_adv_perturbed)):
        y_train_adv.append(y_train_flat_adv[i])
        
    y_train_adv = np.matrix(y_train_adv).astype('int')

    return X_train_text_adv, y_train_adv

oxford_dict = pkl.load(open("homophone_dict.pkl", "rb"))

data_set = 'ag_news'
data_folder_name = 'datasets/' + data_set + '/raw/'

perturbation_prob_vec=np.append(np.array(range(0,99,10)),99)
attack_list = ["HomoPhones", "SimilarSymbols", "NeighborKeyboard"]

adv_prob = 50
adv_perturbation_strength = 50
N_features_adv = 300
N_gram_adv = 2

random_seed = int(sys.argv[1])

train_data = pd.read_csv(data_folder_name + 'train_'+str(random_seed)+'.csv', header=None)
test_data = pd.read_csv(data_folder_name + 'test_'+str(random_seed)+'.csv', header=None)

y_train_original = np.array(train_data[0]).reshape(len(train_data[0]), 1)
X_train_original = list(train_data[2])
X_train_original = [data_helper.basic_text_cleaner(text) for text in X_train_original]

y_test_original = np.array(test_data[0]).reshape(len(test_data[0]), 1)
X_test_original = list(test_data[2])
X_test_original = [data_helper.basic_text_cleaner(text) for text in X_test_original]

models_dir = 'results/' + data_set + '/models/'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Adv Training #
for attack_idx in range(len(attack_list)):
    attack_name = attack_list[attack_idx]
    print(attack_name)
    # Create Adv Data
    X_train_adv, y_train_adv = create_adv_data(adv_prob, adv_perturbation_strength, 
                                               X_train_original, y_train_original,
                                               attack_list, attack_idx)
    

    vectorizer_adv, svd_adv, X_train_adv_reduced, X_test_adv_reduced = data_helper.TFIDF_SVD(N_gram_adv, N_features_adv, X_train_adv, X_test_original)

    #adversarial training
    clf = OneVsRestClassifier(SVC(kernel='rbf', probability=True, random_state = random_seed), n_jobs = -1)
    clf.fit(X_train_adv_reduced, y_train_adv)
    pred = clf.score(X_train_adv_reduced, y_train_adv)

    # save model
    pkl.dump(clf, open(models_dir + 'svm_adv_seed_' + str(random_seed) + '_'+ attack_name + '_' + 
                           str(adv_prob) + '_' + str(adv_perturbation_strength) + '.pkl', 'wb'), protocol = 4)

    suitcase = [vectorizer_adv, svd_adv]
    pkl.dump(suitcase, open(models_dir + 'svm_vectorizer_svd_adv_seed_' + str(random_seed) +
                            '_'+ attack_name + '_' + str(adv_prob) + '_' + str(adv_perturbation_strength) + '.pkl', 'wb'), protocol=4)

    print("Model and Vectorizer saved")


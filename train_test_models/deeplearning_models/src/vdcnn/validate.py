# -*- coding: utf-8 -*-
"""
@author:
        - Charles-Emmanuel DIAS  <Charles-Emmanuel.Dias@lip6.fr>
        - Ardalan Mehrani <ardalan77400@gmail.com>
@brief:
"""

import os
import re
import torch
import lmdb
import pickle
import itertools
import argparse
import numpy as np
import torch.nn as nn
import pickle as pkl

from tqdm import tqdm
from collections import Counter
from sklearn import utils, metrics
from src.datasets import load_datasets


from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from src.vdcnn.net import VDCNN

# multiprocessing workaround
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


from src.perturbations_func import SubstituteSimilarSymbolsFunction, SubstituteNeighborKeyboardFunction, SwappingNeighborLetterFunction
from src.data_helper import perturb_text

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

import pandas as pd

def get_args():
    parser = argparse.ArgumentParser("""Very Deep CNN with optional residual connections (https://arxiv.org/abs/1606.01781)""")
    parser.add_argument("--dataset", type=str, default='ag_news')
    parser.add_argument("--model_folder", type=str, default="models/vdcnn/ag_news")
    parser.add_argument("--data_folder", type=str, default="datasets/ag_news/vdcnn")
    parser.add_argument("--depth", type=int, choices=[9, 17, 29, 49], default=9, help="Depth of the network tested in the paper (9, 17, 29, 49)")
    parser.add_argument("--maxlen", type=int, default=1024)
    parser.add_argument('--shortcut', action='store_true', default=False)
    parser.add_argument("--batch_size", type=int, default=128, help="number of example read by the gpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--solver", type=str, default="sgd", help="'agd' or 'adam'")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_halve_interval", type=float, default=10, help="Number of iterations before halving learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Number of iterations before halving learning rate")
    parser.add_argument("--snapshot_interval", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--nthreads", type=int, default=4)
    args = parser.parse_args()
    return args


class Preprocessing():

    def __init__(self, lowercase=True):
        self.lowercase = lowercase

    def transform(self, sentences):
        """
        sentences: list(str)
        output: list(str)
        """
        return [s.lower() for s in sentences]


class CharVectorizer():
    def __init__(self, maxlen=10, padding='pre', truncating='pre', alphabet="""abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/| #$%ˆ&*˜‘+=<>()[]{}"""):

        self.alphabet = alphabet
        self.maxlen = maxlen
        self.padding = padding
        self.truncating = truncating

        self.char_dict = {'_pad_': 0, '_unk_': 1, ' ': 2}
        for i, k in enumerate(self.alphabet, start=len(self.char_dict)):
            self.char_dict[k] = i

    def transform(self,sentences):
        """
        sentences: list of string
        list of review, review is a list of sequences, sequences is a list of int
        """
        sequences = []

        for sentence in sentences:
            seq = [self.char_dict.get(char, self.char_dict["_unk_"]) for char in sentence]

            if self.maxlen:
                length = len(seq)
                if self.truncating == 'pre':
                    seq = seq[-self.maxlen:]
                elif self.truncating == 'post':
                    seq = seq[:self.maxlen]

                if length < self.maxlen:

                    diff = np.abs(length - self.maxlen)

                    if self.padding == 'pre':
                        seq = [self.char_dict['_pad_']] * diff + seq

                    elif self.padding == 'post':
                        seq = seq + [self.char_dict['_pad_']] * diff
            sequences.append(seq)

        return sequences

    def get_params(self):
        params = vars(self)
        return params


class TupleLoader(Dataset):

    def __init__(self, path=""):
        self.path = path

        self.env = lmdb.open(path, max_readers=opt.nthreads, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)

    def __len__(self):
        return list_from_bytes(self.txn.get('nsamples'.encode()))[0]

    def __getitem__(self, i):
        xtxt = list_from_bytes(self.txn.get(('txt-%09d' % i).encode()), np.int)
        lab = list_from_bytes(self.txn.get(('lab-%09d' % i).encode()), np.int)[0]
        return xtxt, lab


def get_metrics(cm, list_metrics):
    """Compute metrics from a confusion matrix (cm)
    cm: sklearn confusion matrix
    returns:
    dict: {metric_name: score}

    """
    dic_metrics = {}
    total = np.sum(cm)

    if 'accuracy' in list_metrics:
        out = np.sum(np.diag(cm))
        dic_metrics['accuracy'] = out/total

    if 'pres_0' in list_metrics:
        num = cm[0, 0]
        den = cm[:, 0].sum()
        dic_metrics['pres_0'] =  num/den if den > 0 else 0

    if 'pres_1' in list_metrics:
        num = cm[1, 1]
        den = cm[:, 1].sum()
        dic_metrics['pres_1'] = num/den if den > 0 else 0

    if 'recall_0' in list_metrics:
        num = cm[0, 0]
        den = cm[0, :].sum()
        dic_metrics['recall_0'] = num/den if den > 0 else 0

    if 'recall_1' in list_metrics:
        num = cm[1, 1]
        den = cm[1, :].sum()
        dic_metrics['recall_1'] =  num/den if den > 0 else 0

    return dic_metrics


def predict(net,dataset,device,msg="prediction"):

    net.eval()

    y_probs, y_trues = [], []

    for iteration, (tx, ty) in tqdm(enumerate(dataset), total=len(dataset), desc="{}".format(msg)):

        data = (tx, ty)
        data = [x.to(device) for x in data]
        out = net(data[0])
        ty_prob = F.softmax(out, 1) # probabilites
        y_probs.append(ty_prob.detach().cpu().numpy())
        y_trues.append(data[1].detach().cpu().numpy())

    return np.concatenate(y_probs, 0), np.concatenate(y_trues, 0).reshape(-1, 1)


"""
Sven Blankenburg, 28.06.2021
"""
def load(net, path):
    """
    load a model's state and it's embedding dic by piggybacking torch's load_state_dict function
    """
    net.load_state_dict(torch.load(path))



def list_to_bytes(l):
    return np.array(l).tobytes()


def list_from_bytes(string, dtype=np.int):
    return np.frombuffer(string, dtype=dtype)



if __name__ == "__main__":

    opt = get_args()
    print("parameters: {}".format(vars(opt)))
    function_list = [SubstituteSimilarSymbolsFunction, SubstituteNeighborKeyboardFunction, SwappingNeighborLetterFunction]
    #function_list = [ SubstituteNeighborKeyboardFunction, SwappingNeighborLetterFunction]
    df_results_original = pd.DataFrame({'Mode': [], 'Perturbation Training': [], 'Dataset': [],'Epoch': [], 'Random_Seed': [],  'ACC_Train': [],  'ACC_Test': []})
    #epoch = 30
    #max_epoch = 30
    #for epoch in range(1,max_epoch):

    os.makedirs(opt.model_folder, exist_ok=True)
    os.makedirs(opt.data_folder, exist_ok=True)

    dataset = load_datasets(names=[opt.dataset])[0]
    dataset_name = dataset.__class__.__name__
    n_classes = dataset.n_classes
    print("dataset: {}, n_classes: {}".format(dataset_name, n_classes))

    tr_path =  "{}/train.lmdb".format(opt.data_folder)
    te_path = "{}/test.lmdb".format(opt.data_folder)

    # check if datasets exist
    #all_exist = True if (os.path.exists(tr_path) and os.path.exists(te_path)) else False
    all_exist =  False

    preprocessor = Preprocessing()
    vectorizer = CharVectorizer(maxlen=opt.maxlen, padding='post', truncating='post')
    n_tokens = len(vectorizer.char_dict)

    if not all_exist:
        print("Creating datasets")
        tr_sentences = [txt for txt,lab in tqdm(dataset.load_train_data(), desc="counting train samples")]
        tr_labels = [lab for txt,lab in tqdm(dataset.load_train_data(), desc="counting train samples")] #SB
        te_sentences = [txt for txt,lab in tqdm(dataset.load_test_data(), desc="counting test samples")]
        te_labels = [lab for txt,lab in tqdm(dataset.load_test_data(), desc="counting test samples")] #SB


        n_tr_samples = len(tr_sentences)
        n_te_samples = len(te_sentences)

        #############################################################################################

        print("[{}] test samples".format( n_te_samples))
        ###################
        # transform train #
        ###################
        with lmdb.open(tr_path, map_size=1099511627776) as env:
            with env.begin(write=True) as txn:
                for i, (sentence, label) in enumerate(tqdm(dataset.load_train_data(), desc="transform train...", total= n_tr_samples)):

                    xtxt = vectorizer.transform(preprocessor.transform([sentence]))[0]
                    lab = label

                    txt_key = 'txt-%09d' % i
                    lab_key = 'lab-%09d' % i

                    txn.put(lab_key.encode(), list_to_bytes([lab]))
                    txn.put(txt_key.encode(), list_to_bytes(xtxt))

                txn.put('nsamples'.encode(), list_to_bytes([i+1]))


        ##################
        # transform test #
        ##################
        with lmdb.open(te_path, map_size=1099511627776) as env:
            with env.begin(write=True) as txn:
                #for i, (sentence, label) in enumerate(tqdm(dataset.load_test_data(), desc="transform test...", total= n_te_samples)):
                for i in tqdm(range(len(te_sentences)), desc="transform test...", total= n_te_samples):
                    sentence = te_sentences[i]
                    label = te_labels[i]
                    xtxt = vectorizer.transform(preprocessor.transform([sentence]))[0]
                    lab = label

                    txt_key = 'txt-%09d' % i
                    lab_key = 'lab-%09d' % i

                    txn.put(lab_key.encode(), list_to_bytes([lab]))
                    txn.put(txt_key.encode(), list_to_bytes(xtxt))

                txn.put('nsamples'.encode(), list_to_bytes([i+1]))

    tr_loader = DataLoader(TupleLoader(tr_path), batch_size=opt.batch_size, shuffle=True, num_workers=opt.nthreads, pin_memory=True)
    te_loader = DataLoader(TupleLoader(te_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.nthreads, pin_memory=False)

    max_epoch = 30
    for epoch in range(1,max_epoch):

        # select cpu or gpu
        device = torch.device("cuda:{}".format(opt.gpuid) if opt.gpuid >= 0 else "cpu")
        list_metrics = ['accuracy']

        print("Creating model...")
        net = VDCNN(n_classes=n_classes, num_embedding=n_tokens + 1, embedding_dim=16, depth=opt.depth, n_fc_neurons=2048, shortcut=opt.shortcut)
        criterion = torch.nn.CrossEntropyLoss()
        net.to(device)

        assert opt.solver in ['sgd', 'adam']
        if opt.solver == 'sgd':
            print(" - optimizer: sgd")
            optimizer = torch.optim.SGD(net.parameters(), lr = opt.lr, momentum=opt.momentum)
        elif opt.solver == 'adam':
            print(" - optimizer: adam")
            optimizer = torch.optim.Adam(net.parameters(), lr = opt.lr)

        scheduler = None
        if opt.lr_halve_interval and  opt.lr_halve_interval > 0:
            print(" - lr scheduler: {}".format(opt.lr_halve_interval))
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_halve_interval, gamma=opt.gamma, last_epoch=-1)

        print("loading saved state dictionary...")
        path = "{}/model_epoch_{}".format(opt.model_folder,epoch)
        load(net, path=path)


        y_probs, y_trues = predict(net,te_loader,device,msg="prediction")
        y_preds = []
        for i in range(0,len(y_probs)):
            y_preds.append([np.argmax(y_probs[i])])

        accuracy_test = metrics.accuracy_score(y_preds, y_trues)

        y_probs, y_trues = predict(net,tr_loader,device,msg="prediction")
        y_preds = []
        for i in range(0,len(y_probs)):
            y_preds.append([np.argmax(y_probs[i])])

        accuracy_train = metrics.accuracy_score(y_preds, y_trues)


        model_str = 'VDCNN(29)'
        idx = 0

        df_results_original = df_results_original.append(pd.DataFrame({'Mode': [model_str], 'Perturbation Training': ['No Training'], 'Dataset': [dataset_name],'Epoch': [epoch], 'Random_Seed': [idx],  'ACC_Train': [accuracy_train], 'ACC_Test': [accuracy_test]}))
        suitcase = [df_results_original]
        filename = "accuracy_curve_samples_"+model_str+"_"+dataset_name+"_models"
        pkl.dump(suitcase, open(filename+".pkl", "wb") )

        print(df_results_original)

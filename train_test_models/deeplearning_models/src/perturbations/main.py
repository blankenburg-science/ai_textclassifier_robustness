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

from src.perturbations.perturbations_func import SwappingNeighborLetterFunction
from src.perturbations.data_helper import perturb_text

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

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


def train(epoch,net,dataset,device,msg="val/test",optimize=False,optimizer=None,scheduler=None,criterion=None):

    net.train() if optimize else net.eval()

    epoch_loss = 0
    nclasses = len(list(net.parameters())[-1])
    cm = np.zeros((nclasses,nclasses), dtype=int)

    with tqdm(total=len(dataset),desc="Epoch {} - {}".format(epoch, msg)) as pbar:
        for iteration, (tx, ty) in enumerate(dataset):

            data = (tx, ty)
            data = [x.to(device) for x in data]

            if optimize:
                optimizer.zero_grad()

            out = net(data[0])
            ty_prob = F.softmax(out, 1) # probabilites

            #metrics
            y_true = data[1].detach().cpu().numpy()
            y_pred = ty_prob.max(1)[1].cpu().numpy()

            cm += metrics.confusion_matrix(y_true, y_pred, labels=range(nclasses))
            dic_metrics = get_metrics(cm, list_metrics)

            loss =  criterion(out, data[1])
            epoch_loss += loss.item()
            dic_metrics['logloss'] = epoch_loss/(iteration+1)

            if optimize:
                loss.backward()
                optimizer.step()
                dic_metrics['lr'] = optimizer.state_dict()['param_groups'][0]['lr']

            pbar.update(1)
            pbar.set_postfix(dic_metrics)

    if scheduler:
        scheduler.step()


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


def save(net, path):
    """
    Saves a model's state and it's embedding dic by piggybacking torch's save function
    """
    dict_m = net.state_dict()
    torch.save(dict_m,path)

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

    #os.makedirs(opt.model_folder, exist_ok=True)
    #os.makedirs(opt.data_folder, exist_ok=True)

    dataset = load_datasets(names=[opt.dataset])[0]
    dataset_name = dataset.__class__.__name__
    n_classes = dataset.n_classes
    print("dataset: {}, n_classes: {}".format(dataset_name, n_classes))

    tr_path =  "{}/train.lmdb".format(opt.data_folder)
    te_path = "{}/test.lmdb".format(opt.data_folder)

    # check if datasets exis
    #all_exist = True if (os.path.exists(tr_path) and os.path.exists(te_path)) else False
    all_exist = False;

    preprocessor = Preprocessing()
    vectorizer = CharVectorizer(maxlen=opt.maxlen, padding='post', truncating='post')
    n_tokens = len(vectorizer.char_dict)

    if not all_exist:
        print("Creating datasets")
        tr_sentences = [txt for txt,lab in tqdm(dataset.load_train_data(), desc="counting train samples")]
        te_sentences = [txt for txt,lab in tqdm(dataset.load_test_data(), desc="counting test samples")]

        n_tr_samples = len(tr_sentences)
        n_te_samples = len(te_sentences)
        #del tr_sentences
        #del te_sentences

        print("[{}/{}] train/test samples".format(n_tr_samples, n_te_samples))

    print(tr_sentences[0])
    print("******************************************************************************")
    text_mod = SwappingNeighborLetterFunction(tr_sentences, 0, 0, 0)
    #tr_sentences_mod = perturb_text(tr_sentences, 50.0, [SwappingNeighborLetterFunction], 0)
    perturbation_prob = 50.0
    function_list = [SwappingNeighborLetterFunction]
    fun_index = 0
    inputs = range(len(tr_sentences))
    tr_sentences_mod = Parallel(n_jobs=num_cores)(delayed(perturb_text)([tr_sentences[i]], perturbation_prob, function_list, fun_index) for i in tqdm(inputs))

    print(tr_sentences_mod[0])

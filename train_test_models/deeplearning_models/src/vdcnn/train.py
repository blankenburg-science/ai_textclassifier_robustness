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
import pandas as pd

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

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

import re
from torchsummary import summary

"""
Sven Blankenburg, 23.08.2021
"""
def basic_text_cleaner(text):

    # remove hrefs (links)
    text = re.sub('(?:\s)&lt;A HREF=[^, ]*', '', text)
    text = re.sub('(?:\s)target=/stocks/[^, ]*', '', text)

    text = text.replace(".", "")
    text = text.replace(":", " ")
    text = text.replace(";", "")
    text = text.replace("?", "")
    text = text.replace("!", "")
    text = text.replace("[", " ")
    text = text.replace(",", "")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("\n", " ")
    text = text.replace("\\n", " ")
    text = text.replace("- ", "")
    text = text.replace("-", " ")
    text = text.replace("=", " ")
    text = text.replace('\\"', ' ')
    text = text.replace("\\", " ")
    text = text.replace("\\'", "'")
    text = text.replace("&lt", "")
    text = text.replace("&gt", "")
    text = text.replace("/B", "")
    #remove everything containing with #
    #text = re.sub(r'(\s)#\w+', '', text)
    text = re.sub(r'\w*#\w*', '', text)

    # spaces
    text = text.replace("    ", " ")
    text = text.replace("   ", " ")
    text = text.replace("  ", " ")

    return text.lower()


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
    parser.add_argument("--fold", type=int, default=0)
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

"""
Sven Blankenburg, 30.08.2021
"""
def transform_batch(prepro, vecto,  te_sentences, batch_no):
    txt = []
    for i in tqdm(range(len(te_sentences)), desc="transform test batch " + str(batch_no)+" ...", total= len(te_sentences)):
        txt.append(vecto.transform(prepro.transform(te_sentences[i]))[0])

    return txt

if __name__ == "__main__":

    opt = get_args()
    print("parameters: {}".format(vars(opt)))

    os.makedirs(opt.model_folder, exist_ok=True)
    os.makedirs(opt.data_folder, exist_ok=True)

    dataset = load_datasets(names=[opt.dataset], fold=opt.fold)[0]
    dataset_name = dataset.__class__.__name__
    n_classes = dataset.n_classes
    print("dataset: {}, n_classes: {}".format(dataset_name, n_classes))

    tr_path =  "{}/train_{}_{}.lmdb".format(opt.data_folder,str(opt.fold),str(opt.depth))
    te_path = "{}/test_{}_{}.lmdb".format(opt.data_folder,str(opt.fold),str(opt.depth))

    print(tr_path)
    print(te_path)
    # check if datasets exis
    #all_exist = True if (os.path.exists(tr_path) and os.path.exists(te_path)) else False
    all_exist = False
    if (os.path.exists(tr_path) and os.path.exists(te_path)):
        print('Dataset already created !!!')
        recreate_dataset = input("Do you want to re-create the dataset (y/n)?:")
        if recreate_dataset=='y':
            all_exist = False
        else:
            all_exist = True


    preprocessor = Preprocessing()
    vectorizer = CharVectorizer(maxlen=opt.maxlen, padding='post', truncating='post')
    n_tokens = len(vectorizer.char_dict)

    batches = 18

    if not all_exist:
        print("Creating datasets")
        #tr_sentences = [txt for txt,lab in tqdm(dataset.load_train_data(), desc="counting train samples")]
        #te_sentences = [txt for txt,lab in tqdm(dataset.load_test_data(), desc="counting test samples")]

        #n_tr_samples = len(tr_sentences)
        #n_te_samples = len(te_sentences)
        #del tr_sentences
        #del te_sentences

        tr_sentences = [basic_text_cleaner(txt) for txt, lab in tqdm(dataset.load_train_data(), desc="counting train samples")]
        tr_labels = [lab for txt, lab in tqdm(dataset.load_train_data(), desc="counting train samples")]

        te_sentences = [basic_text_cleaner(txt) for txt, lab in tqdm(dataset.load_test_data(), desc="counting test samples")]
        te_labels = [lab for txt, lab in tqdm(dataset.load_test_data(), desc="counting test samples")]

        n_tr_samples = len(tr_sentences)
        n_te_samples = len(te_sentences)

        print("[{}/{}] train/test samples".format(n_tr_samples, n_te_samples))

        min_seq = 999999
        max_seq = 0
        for sentence in tqdm(tr_sentences, desc="analyse min/max sequence...", total=n_tr_samples):
            if len(sentence)<min_seq:
                min_seq = len(sentence)
            if len(sentence)>max_seq:
                max_seq = len(sentence)

        print(" - shortest sequence: {}, longest sequence: {}".format(min_seq, max_seq))
        print(" - [{}/{}] train/test samples".format(n_tr_samples, n_te_samples))


        ###################
        # transform train #
        ###################
        for i in tqdm(range(len(tr_sentences)), desc="text cleaning train ...", total=n_tr_samples):
            tr_sentences[i] = str(tr_sentences[i])

        vectorizer.transform(preprocessor.transform(tr_sentences[0]))
        print(tr_sentences[0:5])
        inputs = range(batches)
        #xtxt_vec = Parallel(n_jobs=int(np.ceil(num_cores*1.5)))(delayed(transform_batch)(preprocessor, vectorizer, tr_sentences[batch_no*int(np.ceil(len(tr_sentences)/batches)):(batch_no+1)*int(np.ceil(len(tr_sentences)/batches))], batch_no) for batch_no in tqdm(inputs))
        #xtxt_vec = np.array(xtxt_vec).flatten()

        #xtxt_vec_tmp=[]
        #if len(xtxt_vec)==len(inputs):
    #        for batch in range(len(inputs)):
#                for sample_txt in xtxt_vec[batch]:
#                    xtxt_vec_tmp.append(sample_txt)
            #xtxt_vec = xtxt_vec_tmp
        #del xtxt_vec_tmp


        #print('Length of xtxt_vec:')
        #print(len(tr_labels))
        #print(xtxt_vec[0])

        with lmdb.open(tr_path, map_size=1099511627776) as env:
            with env.begin(write=True) as txn:
                #for i in tqdm(range(len(tr_sentences)), desc="transform train...", total= n_tr_samples):
                for i, (sentence, label) in enumerate(tqdm(dataset.load_train_data(), desc="transform train...", total= n_tr_samples)):
                    xtxt = vectorizer.transform(preprocessor.transform([basic_text_cleaner(sentence)]))[0]
                    lab = label #tr_labels[i]
                    #xtxt = xtxt_vec[i] #vecto.transform(prepro.transform(sentence))
                    #lab = tr_labels[i]
                    #xtxt = vectorizer.transform(preprocessor.transform([sentence]))[0]
                    #lab = label

                    txt_key = 'txt-%09d' % i
                    lab_key = 'lab-%09d' % i

                    txn.put(lab_key.encode(), list_to_bytes([lab]))
                    txn.put(txt_key.encode(), list_to_bytes(xtxt))

                txn.put('nsamples'.encode(), list_to_bytes([i+1]))

        ##################
        # transform test #
        ##################
        for i in tqdm(range(len(te_sentences)), desc="text cleaning test ...", total=n_te_samples):
            te_sentences[i] = basic_text_cleaner(str(te_sentences[i]))

        inputs = range(batches)
        #xtxt_vec = Parallel(n_jobs=int(np.ceil(num_cores*1.5)))(delayed(transform_batch)(prepro_vec[batch_no], vecto_vec[batch_no], te_sentences[batch_no*int(np.ceil(len(te_sentences)/batches)):(batch_no+1)*int(np.ceil(len(te_sentences)/batches))], batch_no) for batch_no in tqdm(inputs))
        #xtxt_vec = Parallel(n_jobs=int(np.ceil(num_cores*1.5)))(delayed(transform_batch)(preprocessor, vectorizer, te_sentences[batch_no*int(np.ceil(len(te_sentences)/batches)):(batch_no+1)*int(np.ceil(len(te_sentences)/batches))], batch_no) for batch_no in tqdm(inputs))
        #xtxt_vec = np.array(xtxt_vec).flatten()



        #print('Length of xtxt_vec:')
        #print(len(te_labels))

        with lmdb.open(te_path, map_size=1099511627776) as env:
            with env.begin(write=True) as txn:
                for i, (sentence, label) in enumerate(tqdm(dataset.load_test_data(), desc="transform test...", total= n_te_samples)):
                #for i in tqdm(range(len(te_sentences)), desc="transform test...", total= n_te_samples):
                    xtxt = vectorizer.transform(preprocessor.transform([basic_text_cleaner(sentence)]))[0]
                    lab = label #te_labels[i]
                    #xtxt = xtxt_vec[i] #vecto_vec[0].transform(prepro_vec[0].transform(te_sentences[i]))
                    #lab = te_labels[i]

                    txt_key = 'txt-%09d' % i
                    lab_key = 'lab-%09d' % i

                    txn.put(lab_key.encode(), list_to_bytes([lab]))
                    txn.put(txt_key.encode(), list_to_bytes(xtxt))

                txn.put('nsamples'.encode(), list_to_bytes([i+1]))

    print("Loading datasets")
    tr_loader = DataLoader(TupleLoader(tr_path), batch_size=opt.batch_size, shuffle=True, num_workers=opt.nthreads, pin_memory=True)
    te_loader = DataLoader(TupleLoader(te_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.nthreads, pin_memory=False)

    # select cpu or gpu
    device = torch.device("cuda:{}".format(opt.gpuid) if opt.gpuid >= 0 else "cpu")
    list_metrics = ['accuracy']


    #check if already saved models exists
    first_epoch_exists = False
    last_epoch_saved = 0
    for epoch in range(opt.epochs + 1, 1, -1):
        #path = "{}/model_{}_epoch_{}".format(opt.model_folder,str(opt.depth),epoch)
        path = "{}/model_{}_fold_{}_epoch_{}".format(opt.model_folder, str(opt.depth), str(opt.fold),epoch)
        if os.path.exists(path):
            last_epoch_saved = epoch
            break

        #if epoch==1 and os.path.exists(path):
        #    first_epoch_exists=True
        #if first_epoch_exists and os.path.exists(path)==False:
        #   last_epoch_saved = epoch + 1
        #    break


    print("Creating model...")
    net = VDCNN(n_classes=n_classes, num_embedding=n_tokens + 1, embedding_dim=16, depth=opt.depth, n_fc_neurons=2048, shortcut=opt.shortcut)
    summary(net)
    input("STOPPPPPP!!!")

    if last_epoch_saved>0:
        print('Found saved model:'+path);
        pretrained_model = input("Do you want to use this already pretrained model (y/n)?:")
        if pretrained_model=='y':
            print("loading pre-trained model...")
            path = "{}/model_{}_fold_{}_epoch_{}".format(opt.model_folder, str(opt.depth), str(opt.fold),last_epoch_saved)
            print(path)
            load(net, path=path)
        else:
            print('deleting old models ...')
            cmd = "rm "+"{}/model_{}_fold_{}_epoch_{}".format(opt.model_folder, str(opt.depth), str(opt.fold),"*")
            print(cmd)
            output = os.system(cmd)
            print(output)
            print("using fresh model...")
            last_epoch_saved = 0
    else:
        print("using fresh model...")
        last_epoch_saved = 0


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

    if last_epoch_saved == 0:
        df_results_original = pd.DataFrame({'Mode': [], 'Perturbation Training': [], 'Dataset': [],'Epoch': [], 'Random_Seed': [],  'ACC_Train': [],  'ACC_Test': []})
    else:
        path_pkl = "{}/model_{}_fold_{}.pkl".format(opt.model_folder,str(opt.depth),str(opt.fold))
        df_results_original = pkl.load( open(path_pkl,"rb") )
        print(df_results_original)

    for epoch in range(last_epoch_saved + 1, opt.epochs + 1):
        dic_metrics_train = train(epoch,net, tr_loader, device, msg="training", optimize=True, optimizer=optimizer, scheduler=scheduler, criterion=criterion)
        dic_metrics_test = train(epoch,net, te_loader, device, msg="testing ", criterion=criterion)

        accuracy_train = dic_metrics_train['accuracy']
        accuracy_test = dic_metrics_test['accuracy']
        model_str = 'VDCNN('+str(opt.depth)+')'
        idx = opt.fold #folds as random seed
        df_results_original = df_results_original.append(pd.DataFrame({'Mode': [model_str], 'Perturbation Training': ['No Training'], 'Dataset': [dataset_name],'Epoch': [epoch], 'Random_Seed': [idx],  'ACC_Train': [accuracy_train], 'ACC_Test': [accuracy_test]}))
        print(df_results_original)
        #path_pkl = "{}/model_{}.pkl".format(opt.model_folder,str(opt.depth))
        path_pkl = "{}/model_{}_fold_{}.pkl".format(opt.model_folder,str(opt.depth),str(opt.fold))
        pkl.dump(df_results_original, open(path_pkl,"wb"))

        if (epoch % opt.snapshot_interval == 0) and (epoch > 0):
            #path = "{}/model_{}_epoch_{}".format(opt.model_folder,str(opt.depth),epoch)
            path = "{}/model_{}_fold_{}_epoch_{}".format(opt.model_folder, str(opt.depth), str(opt.fold),epoch)
            print("snapshot of model saved as {}".format(path))
            save(net, path=path)


    if opt.epochs > 0:
        #path = "{}/model_{}_epoch_{}".format(opt.model_folder,str(opt.depth),epoch)
        path = "{}/model_{}_fold_{}_epoch_{}".format(opt.model_folder, str(opt.depth), str(opt.fold),epoch)
        print("snapshot of model saved as {}".format(path))
        save(net, path=path)

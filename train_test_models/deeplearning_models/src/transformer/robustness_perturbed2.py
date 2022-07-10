# -*- coding: utf-8 -*-
"""
@author: Ardalan Mehrani <ardalan77400@gmail.com>

@brief:
"""

import os
import lmdb
import argparse
import numpy as np
import pickle as pkl
import time

from tqdm import tqdm
from sklearn import metrics
from pprint import pprint

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# multiprocessing workaround
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

from src.datasets import load_datasets
from src.transformer.net import TransformerCls
from src.transformer.lib import Preprocessing, Vectorizer, list_to_bytes, list_from_bytes

from src.perturbations_func import SubstituteSimilarSymbolsFunction, SubstituteNeighborKeyboardFunction, SwappingNeighborLetterFunction
from src.data_helper import perturb_text, perturb_text_homophones

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

import pandas as pd

import re

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
    parser = argparse.ArgumentParser("""paper: Attention Is All You Need (https://arxiv.org/abs/1706.03762)""")
    parser.add_argument("--dataset", type=str, default='ag_news')
    parser.add_argument("--data_folder", type=str, default="datasets/ag_news/transformer")
    parser.add_argument("--model_folder", type=str, default="models/transformer/ag_news")

    # preprocessing
    parser.add_argument("--word_min_count", type=int, default=5, help="")
    parser.add_argument('--curriculum', default=False, action='store_true', help="curriculum learning, sort training set by lenght")

    #model
    parser.add_argument("--attention_dim", type=int, default=16, help="")
    parser.add_argument("--n_heads", type=int, default=2, help="")
    parser.add_argument("--n_layers", type=int, default=2, help="")
    parser.add_argument("--maxlen", type=int, default=20, help="truncate longer sequence while training")
    parser.add_argument("--dropout", type=float, default=0.1, help="")
    parser.add_argument("--ff_hidden_size", type=int, default=128, help="point wise feed forward nn")

    #optimizer
    parser.add_argument("--opt_name", type=str, default='adam_warmup_linear', choices=['adam', 'adam_warmup_linear'])
    parser.add_argument("--lr", type=float, default=0.0001) #0.001
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--n_warmup_step", type=int, default=1000, help="scheduling optimizer warmup step. set to -1 for regular adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=None, help="gradient clipping")

    # training
    parser.add_argument("--batch_size", type=int, default=32, help="number of example read by the gpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--snapshot_interval", type=int, default=10, help="Save model every n epoch")
    parser.add_argument('--gpuid', type=int, default=0, help="select gpu index. -1 to select cpu")
    parser.add_argument('--nthreads', type=int, default=8, help="number of cpu threads")
    parser.add_argument('--use_all_gpu', type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--fold", type=int, default=0)

    parser.add_argument("--input_folder", type=str, default="raw")
    parser.add_argument("--wait_minutes", type=int, default=0)
    args = parser.parse_args()
    return args


def predict(net,dataset,device,msg="prediction",optimize=False,optimizer=None,criterion=None):

    net.eval()
    epoch_loss = 0
    epoch_acc = 0
    dic_metrics= {'loss':0, 'acc':0, 'lr':0}
    nclasses = len(list(net.parameters())[-1])

    y_probs, y_trues = [], []

    for iteration, (tx,mask,ty) in enumerate(dataset):

        data = (tx,mask,ty)
        data = [x.to(device) for x in data]

        if optimize:
            optimizer.zero_grad()

        out = net(data[0],data[1])
        #loss =  criterion(out, data[2])

        ty_prob = F.softmax(out, 1) # probabilites
        y_probs.append(ty_prob.detach().cpu().numpy())
        y_trues.append(data[2].detach().cpu().numpy())

    return np.concatenate(y_probs, 0), np.concatenate(y_trues, 0).reshape(-1, 1)


def train(epoch,net,dataset,device,msg="val/test",optimize=False,optimizer=None,criterion=None):

    net.train() if optimize else net.eval()

    epoch_loss = 0
    epoch_acc = 0
    dic_metrics= {'loss':0, 'acc':0, 'lr':0}
    nclasses = len(list(net.parameters())[-1])

    with tqdm(total=len(dataset),desc="Epoch {} - {}".format(epoch, msg)) as pbar:
        for iteration, (tx,mask,ty) in enumerate(dataset):

            data = (tx,mask,ty)
            data = [x.to(device) for x in data]

            if optimize:
                optimizer.zero_grad()

            out = net(data[0],data[1])
            loss =  criterion(out, data[2])

            #metrics
            epoch_loss += loss.item()
            epoch_acc += (data[-1] == out.argmax(-1)).sum().item() / len(out)

            dic_metrics['loss'] = epoch_loss/(iteration+1)
            dic_metrics['acc'] = epoch_acc/(iteration+1)

            if optimize:
                loss.backward()
                dic_metrics['lr'] = optimizer.state_dict()['param_groups'][0]['lr']
                optimizer.step()

            pbar.update(1)
            pbar.set_postfix(dic_metrics)


def save(net, txt_dict, path):
    """
    Saves a model's state and it's embedding dic by piggybacking torch's save function
    """
    dict_m = net.state_dict()
    dict_m["txt_dict"] = txt_dict
    torch.save(dict_m,path)


"""
Sven Blankenburg, 28.06.2021
"""
def load(net, path):
    """
    load a model's state and it's embedding dic by piggybacking torch's load_state_dict function
    """
    net.load_state_dict(torch.load(path), strict=False)


def collate_fn(l):

    sequence, labels = zip(*l)
    local_maxlen = max(map(len, sequence))

    Xs = [np.pad(x, (0, local_maxlen-len(x)), 'constant') for x in sequence]
    tx = torch.LongTensor(Xs)
    tx_mask = tx.ne(0).unsqueeze(-2)
    ty = torch.LongTensor(labels)
    return tx, tx_mask, ty


class TupleLoader(Dataset):

    def __init__(self, path=""):
        self.path = path
        self.env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)

    def __len__(self):
        return list_from_bytes(self.txn.get('nsamples'.encode()))[0]

    def __getitem__(self, i):
        """
        i: int
        xtxt: np.array([maxlen])
        """
        xtxt = list_from_bytes(self.txn.get(('txt-%09d' % i).encode()), np.int)
        lab = list_from_bytes(self.txn.get(('lab-%09d' % i).encode()), np.int)[0]
        xtxt = xtxt[:opt.maxlen]
        return xtxt, lab


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


"""
Sven Blankenburg, 18.08.2021
"""
def transform_batch(prepro, vecto,  te_sentences, batch_no):
    txt = []
    for i in tqdm(range(len(te_sentences)), desc="transform test batch " + str(batch_no)+" ...", total= len(te_sentences)):
        txt.append(vecto.transform(prepro.transform(te_sentences[i])))

    return txt

if __name__ == "__main__":

    print('1')
    opt = get_args()
    print('2')

    os.makedirs(opt.model_folder, exist_ok=True)
    os.makedirs(opt.data_folder, exist_ok=True)

    print("parameters:")
    pprint(vars(opt))
    torch.manual_seed(opt.seed)

    dataset = load_datasets(names=[opt.dataset], fold=opt.fold, input_folder=opt.input_folder)[0]
    dataset_name = dataset.__class__.__name__
    n_classes = dataset.n_classes
    print("dataset: {}, n_classes: {}".format(dataset_name, n_classes))

    #######################
    # load trained vector #
    #######################
    print("load pre-trained vectorizer")
    #path = "{}/prepro_vecto.pkl".format(opt.data_folder)
    #path = "{}/prepro_vecto_{}.pkl".format(opt.data_folder,opt.attention_dim)
    path = "{}/prepro_vecto_{}_fold_{}.pkl".format(opt.data_folder,opt.attention_dim, opt.fold)
    [prepro, vecto] = pkl.load( open(path, "rb") )

    #prepro_vec = []
    #vecto_vec = []
    batches = 18
    #for i in range(batches):
    #    [prepro, vecto] = pkl.load( open(path, "rb") )
    #    vecto.transform("HI")
    #    print(vecto.n_transform)
    #    prepro_vec.append(prepro)
    #    vecto_vec.append(vecto)


    attack_list = ["NeighborKeyboard", "SimilarSymbols", "HomoPhones"]
    function_list = [SubstituteNeighborKeyboardFunction, SubstituteSimilarSymbolsFunction, SwappingNeighborLetterFunction]
    epoch = opt.epochs
    print('Using Epoch:'+str(epoch))


    for fun_index in range(0,2):
        df_results_original = pd.DataFrame({'Mode': [], 'Perturbation Training': [], 'Dataset': [],'Sample': [], 'Random_Seed': [], 'Perturbation': [], 'Perturbation_Strength': [], 'ACC': []})
        perturbation_prob_vec=np.append(np.array(range(0,99,10)),99)
        samples = 5
        for sample_run in range(samples):
            for perturbation_prob in perturbation_prob_vec:

                variables = {
                    'train': {'var': None, 'path': "{}/train_{}_fold_{}.lmdb".format(opt.data_folder,opt.attention_dim, opt.fold)},
                    'test': {'var': None, 'path': "{}/test_{}_fold_{}.lmdb".format(opt.data_folder,opt.attention_dim, opt.fold)},
                    'perturbed': {'var': None, 'path': "{}/perturbed_{}_fold_{}.lmdb".format(opt.data_folder, opt.attention_dim, opt.fold)},
                    'params': {'var': None, 'path': "{}/params_{}_fold_{}.pkl".format(opt.data_folder,opt.attention_dim, opt.fold)},
                }

                # check if datasets exis
                all_exist = True if os.path.exists(variables['params']['path']) else False
                print(all_exist)
                print('----------')
                #all_exist = False

                if all_exist:

                    #variables['params']['var'] = pkl.load(open(variables['params']['path'],"rb"))
                    vecto.transform(prepro.transform("Hi Du"))
                    variables['params']['var'] = vars(vecto)
                    longuest_sequence = variables['params']['var']['longest_sequence']
                    n_tokens = len(variables['params']['var']['word_dict'])
                    #print(variables['params']['var']['word_dict'])
                    #print(n_tokens)


                #else:
                    print("Loading Test dataset")
                    #tr_examples = [(txt,lab) for txt, lab in tqdm(dataset.load_train_data(), desc="counting train samples")]
                    #te_examples = [(txt,lab) for txt, lab in tqdm(dataset.load_test_data(), desc="counting test samples")]

                    #tr_sentences = [txt for txt,lab in tqdm(dataset.load_train_data(), desc="counting train samples")]
                    #tr_labels = [lab for txt,lab in tqdm(dataset.load_train_data(), desc="counting train samples")] #SB

                    te_sentences = [basic_text_cleaner(txt) for txt,lab in tqdm(dataset.load_test_data(), desc="counting test samples")]
                    te_labels = [lab for txt,lab in tqdm(dataset.load_test_data(), desc="counting test samples")] #SB


                    #############################################################################################
                    print("perturbing test dataset")

                    #attack_name = str(function_list[fun_index]).split(' ')[1]
                    attack_name = attack_list[fun_index]
                    inputs = range(len(te_sentences))

                    #te_sentences_mod = Parallel(n_jobs=num_cores)(delayed(perturb_text)([te_sentences[i]], perturbation_prob, function_list, fun_index) for i in tqdm(inputs))
                    if attack_name == "HomoPhones":
                        te_sentences_mod = Parallel(n_jobs=num_cores)(delayed(perturb_text_homophones)([te_sentences[i]], perturbation_prob, function_list, fun_index) for i in tqdm(inputs))
                        #te_sentences_mod = [perturb_text_homophones([te_sentences[i]], perturbation_prob, function_list, fun_index) for i in tqdm(inputs)]
                    elif attack_name == "SimilarSymbols":
                        #te_sentences_mod = [perturb_text([te_sentences[i]], perturbation_prob, [SubstituteSimilarSymbolsFunction], 0) for i in tqdm(inputs)]
                        te_sentences_mod = Parallel(n_jobs=num_cores)(delayed(perturb_text)([te_sentences[i]], perturbation_prob, [SubstituteSimilarSymbolsFunction], 0) for i in tqdm(inputs))

                    elif attack_name == "NeighborKeyboard":
                        #te_sentences_mod = [perturb_text([te_sentences[i]], perturbation_prob, [SubstituteNeighborKeyboardFunction], 0) for i in tqdm(inputs)]
                        te_sentences_mod = Parallel(n_jobs=num_cores)(delayed(perturb_text)([te_sentences[i]], perturbation_prob, [SubstituteNeighborKeyboardFunction], 0) for i in tqdm(inputs))

                    te_sentences = list(np.array(te_sentences_mod).flat)

                    #n_tr_samples = len(tr_sentences)
                    n_te_samples = len(te_sentences)

                    #############################################################################################

                    #prepro = Preprocessing(lowercase=True)
                    #vecto = Vectorizer(min_word_count= opt.word_min_count)


                    ##################
                    # transform test #
                    ##################
                    for i in range(len(te_sentences)):
                        te_sentences[i] = str(te_sentences[i])

                    inputs = range(batches)
                    #xtxt_vec = Parallel(n_jobs=int(np.ceil(num_cores*1.5)))(delayed(transform_batch)(prepro_vec[batch_no], vecto_vec[batch_no], te_sentences[batch_no*int(np.ceil(len(te_sentences)/batches)):(batch_no+1)*int(np.ceil(len(te_sentences)/batches))], batch_no) for batch_no in tqdm(inputs))
                    xtxt_vec = Parallel(n_jobs=int(np.ceil(num_cores*1.5)))(delayed(transform_batch)(prepro, vecto, te_sentences[batch_no*int(np.ceil(len(te_sentences)/batches)):(batch_no+1)*int(np.ceil(len(te_sentences)/batches))], batch_no) for batch_no in tqdm(inputs))
                    xtxt_vec = np.array(xtxt_vec).flatten()

                    xtxt_vec_tmp=[]
                    if len(xtxt_vec)==len(inputs):
                        for batch in range(len(inputs)):
                            for sample_txt in xtxt_vec[batch]:
                                xtxt_vec_tmp.append(sample_txt)
                        xtxt_vec = xtxt_vec_tmp
                    del xtxt_vec_tmp

                    print('Length of xtxt_vec:')
                    print(len(xtxt_vec))

                    with lmdb.open(variables['perturbed']['path'], map_size=1099511627776) as env:
                        with env.begin(write=True) as txn:
                            for i in tqdm(range(len(te_sentences)), desc="transform test...", total= n_te_samples):
                                xtxt = xtxt_vec[i] #vecto_vec[0].transform(prepro_vec[0].transform(te_sentences[i]))
                                lab = te_labels[i]

                                txt_key = 'txt-%09d' % i
                                lab_key = 'lab-%09d' % i

                                txn.put(lab_key.encode(), list_to_bytes([lab]))
                                txn.put(txt_key.encode(), list_to_bytes(xtxt))

                            txn.put('nsamples'.encode(), list_to_bytes([i+1]))

                    #print(type(txn))
                    #print(txn)
                    #print(variables)
                    variables['params']['var'] = vars(vecto)
                    longuest_sequence = variables['params']['var']['longest_sequence']
                    n_tokens = len(variables['params']['var']['word_dict'])

                    ###############
                    # saving data #
                    ###############
                    print("  - saving to {}".format(variables['params']['path']))
                    pkl.dump(variables['params']['var'],open(variables['params']['path'],"wb"))

                te_loader = DataLoader(TupleLoader(variables['perturbed']['path']), batch_size=opt.batch_size, collate_fn=collate_fn,  shuffle=False, num_workers=opt.nthreads, pin_memory=False)
                #te_loader = DataLoader(TupleLoader(variables['test']['path']), batch_size=opt.batch_size, collate_fn=collate_fn,  shuffle=False, num_workers=opt.nthreads, pin_memory=False)


                # select cpu or gpu
                device = torch.device("cuda:{}".format(opt.gpuid) if opt.gpuid >= 0 else "cpu")

                print("Creating model...")
                net = TransformerCls(nclasses=n_classes,
                                     src_vocab_size=n_tokens,
                                     h=opt.n_heads,
                                     d_model=opt.attention_dim,
                                     d_ff=opt.ff_hidden_size,
                                     dropout=opt.dropout,
                                     n_layer=opt.n_layers)


                #epoch = 15
                print("loading saved state dictionary...")
                #path = "{}/model_{}_epoch_{}".format(opt.model_folder,str(opt.depth),epoch)
                #path = "{}/model_epoch_{}".format(opt.model_folder, epoch)
                #path = "{}/model_{}_epoch_{}".format(opt.model_folder,opt.attention_dim,epoch)
                path = "{}/model_{}_fold_{}_epoch_{}".format(opt.model_folder,opt.attention_dim, str(opt.fold), epoch)
                print(path)
                #path = "{}/model_epoch_{}".format(opt.model_folder,epoch)
                load(net, path=path)

                net.to(device)



                if opt.use_all_gpu==1:
                    print(" - Using all gpus")
                    net = nn.DataParallel(net)

                if opt.max_grad_norm:
                    print(" - gradient clipping: {}".format(opt.max_grad_norm))
                    torch.nn.utils.clip_grad_norm_(net.parameters(), opt.max_grad_norm)

                scheduler = None
                if opt.opt_name == 'adam_warmup_linear':
                    optimizer_ = torch.optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), betas=(0.9, 0.98), eps=1e-09, weight_decay=opt.weight_decay)
                    optimizer = NoamOpt(opt.attention_dim, 1, opt.n_warmup_step, optimizer_)
                elif opt.opt_name == 'adam':
                    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, weight_decay=opt.weight_decay)
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.step_size, gamma=opt.gamma, last_epoch=-1)
                else:
                    raise
                print(opt.opt_name,optimizer, scheduler)

                criterion = torch.nn.CrossEntropyLoss()

                y_probs, y_trues = predict(net,te_loader,device,msg="prediction")

                #train(epoch,net, te_loader, device, msg="testing ", criterion=criterion)
                y_preds = []
                for i in range(0,len(y_probs)):
                    y_preds.append([np.argmax(y_probs[i])])

                accuracy = metrics.accuracy_score(y_preds, y_trues)
                #model_str = 'VDCNN('+str(opt.depth)+')'+"_fold_"+str(opt.fold)+'_epoch_'+str(int(epoch))
                model_str = 'Transformer('+str(opt.attention_dim)+')'+'_epoch_'+str(int(epoch))
                idx = opt.fold
                perturbed_training = opt.input_folder.split('/')[1]
                df_results_original = df_results_original.append(pd.DataFrame({'Mode': [model_str], 'Perturbation Training': [perturbed_training],'Dataset': [dataset_name] , 'Sample': [sample_run], 'Random_Seed': [idx], 'Perturbation': [attack_name], 'Perturbation_Strength': [perturbation_prob], 'ACC': [accuracy]}))

                suitcase = [df_results_original]
                filename = "training_"+perturbed_training+"/"+"perturbation_curve_samples_"+str(samples)+"_"+model_str+"_"+dataset_name+"_"+attack_name+"_"+str(opt.fold)
                pkl.dump(suitcase, open(filename+".pkl", "wb") )

                print(df_results_original)

            print('waiting '+str(opt.wait_minutes)+' minutes to cool down the gpus!!!')
            time.sleep(60*opt.wait_minutes)

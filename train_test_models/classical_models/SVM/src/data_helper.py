import copy
import numpy as np
import os
import random
import re
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk
nltk.download('punkt')

from joblib import Parallel, delayed
import multiprocessing 
import inflect
import copy
import pickle as pkl
from sklearn.metrics import accuracy_score

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import src.perturbations_func as perturbations_func
import time

oxford_dict = pkl.load(open("homophone_dict.pkl", "rb"))

num_cores = multiprocessing.cpu_count()
num_cores = 5

def test_of_subset(test_list, sub_list):

    flag = 0
    if(all(x in test_list for x in sub_list)):
        flag = 1

    if (flag) :
        #print ("Yes, list is subset of other.")
        return True
    else :
        #print ("No, list is not subset of other.")
        return False

def text_cleaner(text):
    """
    text = text.replace(".", "")
    text = text.replace("'", "")
    text = text.replace(":", "")
    text = text.replace(";", "")
    text = text.replace("?", "")
    text = text.replace("!", "")
    text = text.replace("[", " ")
    text = text.replace(",", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("\"", "")
    text = text.replace("-", " ")
    text = text.replace("=", "")
    text = text.replace("%", "")
    text = text.replace("<", "")
    text = text.replace(">", "")
    text = text.replace("&", "")
    text = text.replace("/", "")
    text = text.replace("+", "")
    text = text.replace("*", "")
    text = text.replace("#", "")
    text = text.replace("\\n", " ")
    """

    text = text.replace(".", " ")
    text = text.replace(":", " ")
    text = text.replace(";", " ")
    text = text.replace("?", " ")
    text = text.replace("!", " ")
    text = text.replace("[", " ")
    text = text.replace(",", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("\n", " ")
    text = text.replace("\\n", " ")
    text = text.replace("-", " ")
    text = text.replace("=", " ")
    text = text.replace("/", " or ")
    text = text.replace("+", " plus ")

    rules = [
        #{r'\w*\d\w*': u''}, # remove all words containing digits
        {r'\b\d+\b': u''}, # remove all single digits
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.<\s(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]>.</a>': r'\1'},  # show links instead of texts
        {r'[ \t]<[^<]?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning

    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()

    text = text.lower()

    # stopwords
    text_file = open("stopwords.txt", "r")
    lines = text_file.readlines()
    text_file.close()

    lines_clean=[]
    for i in range(0,len(lines)):
        lines_clean.append(lines[i].replace("\n", ""))

    stop_words = lines_clean

    example_sent=text
    word_tokens = word_tokenize(example_sent)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    joined=' '.join(filtered_sentence)
    text = joined

    return text

def loadData_flat(DATA_SET, DATA_SIZE, random_seed):

    path_WOS = "data_WOS"
    fname = os.path.join(path_WOS,"WebOfScience/"+DATA_SET+"/X.txt")
    fnameY = os.path.join(path_WOS,"WebOfScience/"+DATA_SET+"/Y.txt")

    with open(fname) as f: # X
        content = f.readlines()

    with open(fnameY) as fy: #Y
        contentY = fy.readlines()

    # shuffle data under the constraints that data is correlated
    shuffle_index = np.arange(0,len(content))
    random.Random(random_seed).shuffle(shuffle_index) # shuffle data with random seed!!

    content_save = copy.deepcopy(content)
    contentY_save = copy.deepcopy(contentY)

    for i in range(0, len(content_save)):
        content[i] = content_save[shuffle_index[i]]
        contentY[i] = contentY_save[shuffle_index[i]]

    del content_save, contentY_save

    Label_flat = np.matrix(contentY, dtype=int)
    Label_flat = np.transpose(Label_flat)

    random.seed(random_seed)
    np.random.seed(random_seed)

    return content, Label_flat

def data_splitting(content, Label_flat, split_vec, random_seed):
    # Trying to partition the data such that test and val are subsets of training data

    if np.sum(split_vec) == 100:

        for i in range(0,10000):
            #balanced partioning of data set
            X_train = copy.deepcopy(content)
            y_train = Label_flat

            my_test_val_size = (100 - split_vec[0])

            # split data into train and rest (test and val)
            X_train_text, X_test_text, y_train_flat, y_test_flat  = train_test_split(X_train, y_train, test_size=my_test_val_size/100,random_state= random_seed+i+1)

            if len(split_vec) == 3:
                my_val_size = split_vec[2]/my_test_val_size
                # split rest into test and val
                X_test_text, X_val_text, y_test_flat, y_val_flat  = train_test_split(X_test_text, y_test_flat, test_size=my_val_size,random_state= random_seed+i+2)

                train_bool = test_of_subset(np.unique(np.array(Label_flat)) , np.unique(np.array(y_train_flat)) )
                test_bool = test_of_subset(np.unique(np.array(y_train_flat)) , np.unique(np.array(y_test_flat)) )
                val_bool = test_of_subset(np.unique(np.array(y_train_flat)) , np.unique(np.array(y_val_flat)) )

                if train_bool and test_bool and val_bool:
                    break

            train_bool = test_of_subset(np.unique(np.array(Label_flat)) , np.unique(np.array(y_train_flat)) )
            test_bool = test_of_subset(np.unique(np.array(y_train_flat)) , np.unique(np.array(y_test_flat)) )

            if train_bool and test_bool:
                break

        classes_level2 = np.unique(np.array(y_train_flat))
        if len(split_vec) == 3:
            return X_train_text, y_train_flat, X_test_text, y_test_flat, X_val_text, y_val_flat, classes_level2

        else:
            return X_train_text, y_train_flat, X_test_text, y_test_flat, classes_level2

    else:
        print("ERROR - sum of data split size unequal 100!")


def clean_data(X_text):
    X_text_cleaned = [text_cleaner(x) for x in X_text]
    return X_text_cleaned

def perturb_text(X_test_text, prob, function_list, fun_index):
    X_test_text_modified = copy.deepcopy(X_test_text)
    percentage = prob #of text words to be modified by the perturbation
    for k in range(0,len(X_test_text_modified)):
        text_num=k

        l = len(X_test_text_modified[text_num].split())
        p = int(percentage/100.0 * l)

        sample = random.sample(range(1,len(X_test_text_modified[text_num].split())),  p)

        for i in range(0,len(sample)):
            word = sample[i]
            X_test_text_modified[text_num] = function_list[fun_index](X_test_text_modified, text_num, word, 3)

    return X_test_text_modified


def create_adv_data(adv_prob, adv_perturbation_strength, X_train_original_text, y_train_original_flat, function_list, function_id):

    X_train_text_tmp = copy.deepcopy(X_train_original_text)
    y_train_flat_tmp = copy.deepcopy(y_train_original_flat)

    vec = np.array(range(len(X_train_text_tmp)))
    np.random.shuffle(vec)

    X_train_text_adv = list(np.array(X_train_text_tmp)[vec[0:int((len(X_train_text_tmp)-1) * adv_prob/100.0)]])
    y_train_flat_adv = list(np.array(y_train_flat_tmp)[vec[0:int((len(y_train_flat_tmp)-1) * adv_prob/100.0)]])

    #single threading
    #X_train_text_adv_perturbed = perturb_text(X_train_text_adv, adv_prob, function_list, function_id)

    #multi-threading
    inputs = range(len(X_train_text_adv))
    X_train_text_adv_perturbed = Parallel(n_jobs=num_cores)(delayed(perturb_text)([X_train_text_adv[i]], adv_prob, function_list, function_id) for i in inputs)
    X_train_text_adv_perturbed = list(np.array(X_train_text_adv_perturbed).flat)



    X_train_text_adv = X_train_text_tmp + X_train_text_adv_perturbed

    y_train_adv = []
    for i in range(0,len(X_train_text_tmp)):
        y_train_adv.append(int(y_train_flat_tmp[i]))

    for i in range(0,len(X_train_text_adv_perturbed)):
        y_train_adv.append(int(y_train_flat_adv[i]))

    return X_train_text_adv, y_train_adv


def text_cleaner_homophones(text):
    text = text.replace(".", " ")
    text = text.replace(":", " ")
    text = text.replace(";", " ")
    text = text.replace("?", " ")
    text = text.replace("!", " ")
    text = text.replace("[", " ")
    text = text.replace(",", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("\n", " ")
    text = text.replace("\n", " ")
    text = text.replace("-", " ")
    text = text.replace("=", " ")
    text = text.replace("/", " or ")
    text = text.replace("+", " plus ")

    return text

def text_cleaner_homophones(text):
    text = text.replace("/", " or ")
    text = text.replace("+", " plus ")

    return text


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

def perturb_text_homophones(oxford_dict, content, pert_prob, function_list, fun_index):
    pert_prob = pert_prob/100
    pert_strength_max = []
    content_perturbed = []
    word_list3 = []

    p = inflect.engine()
    for pos in range(0,len(content)):
        word_list = content[pos].split(' ')
        if pert_prob > 0:
            word_list=content[pos].replace('-',' ').split(' ')
            #word_list = [text_cleaner(word).replace(' ','').lower() for word in word_list]
            word_list = [text_cleaner_homophones(word).lower() for word in word_list]
            [word_list.remove(element) for element in word_list if len(element) == 0]
            word_list = ' '.join([word.replace('%',' percent') for word in word_list]).split(' ')
            word_list_tmp = [x for x in word_list if x]
            word_list = word_list_tmp


        word_list2=[]
        for l in range(0,len(word_list)):
            if word_list[l].isdigit():
                word_list[l]=p.number_to_words(word_list[l])
        word_list_tmp = ' '.join(word_list).split(' ')
        word_list = word_list_tmp


        number_of_perturbed_words = int(np.floor(len(word_list)*pert_prob))
        number_of_perturbed_words

        vec = [word in oxford_dict for word in word_list]
        vec2 = np.array(np.where(np.array(vec)==True)[0])
        np.random.shuffle(vec2)
        vec3 = vec2[0:number_of_perturbed_words]

        orginal_text = ' '.join(word_list)

        word_list2 = copy.deepcopy(word_list)

        for i in range(0,len(vec3)):
            if isinstance(oxford_dict[word_list2[vec3[i]]], list):
                word_list2[vec3[i]] = np.random.choice(oxford_dict[word_list2[vec3[i]]])
            else:
                word_list2[vec3[i]] = oxford_dict[word_list2[vec3[i]]]


        perturbed_text = ' '.join(word_list2)
        content_perturbed.append(perturbed_text)

    return content_perturbed

# QT 03.01.2022
def pred_chunks(model, X_test):
    pred = model.predict(X_test)
    return pred
 
def robustness_curve(filename, models_dir, model_str, perturbation_prob_vec, X_test_text, y_test_flat, sample_run, function_list, fun_index, random_seed):
    start = time.time()
    df_results_original = pd.DataFrame({'Attack': [], 'Mode': [], 'Sample': [], 'Random_Seed': [], 'Perturbation_Strength': [], 'ACC': []})
    attack_name = function_list[fun_index]
    print(attack_name)
    
    for prob_index in range(0,len(perturbation_prob_vec)):
        start = time.time()
        #Perturb the Test Data -------------------
        prob = perturbation_prob_vec[prob_index]
       
        X_test_text_modified = copy.deepcopy(X_test_text)
        inputs = range(len(X_test_text_modified))

        if attack_name == "HomoPhones":
            X_test_text_modified = perturb_text_homophones(oxford_dict, X_test_text, prob, function_list, fun_index)

        elif attack_name == "SimilarSymbols":
            X_test_text_modified = Parallel(n_jobs=num_cores)(delayed(perturb_text)([X_test_text[i]], prob, [perturbations_func.SubstituteSimilarSymbolsFunction], 0) for i in inputs)

        elif attack_name == "NeighborKeyboard":
            X_test_text_modified = Parallel(n_jobs=num_cores)(delayed(perturb_text)([X_test_text[i]], prob, [perturbations_func.SubstituteNeighborKeyboardFunction], 0) for i in inputs)

        ##############################################################################################

        if len(X_test_text_modified[0]) == 1:
            X_test_text_modified = [item for sublist in X_test_text_modified for item in sublist]
        start = time.time()
        ## Load SVD and Vectorizer
        [vectorizer_x, svd] = pkl.load(open(models_dir + 'vecto_svd_' + str(random_seed) + '.pkl', 'rb'))
    
        X_test_mod = vectorizer_x.transform(X_test_text_modified) #TF-IDF
        X_test_mod_reduced = svd.transform(X_test_mod) #Singular Value Decomposition -> dim. reduction
        ##############################################################################################
        del vectorizer_x, svd
        end = time.time()
        print(end-start)
        print("Start Predicting "+ str(random_seed))
        
        # QT 03.01.2022 Parallize over dataset 
        X_test_chunks = np.array_split(X_test_mod_reduced, 4)

        ## Load Model
        model = pkl.load(open(models_dir + 'svm_' + str(random_seed) + '.pkl', 'rb'))
        
        pred = Parallel(n_jobs=num_cores)(delayed(pred_chunks)(model, i) for i in X_test_chunks)
        del model
        flat_pred = [item for sublist in pred for item in sublist]
        accuracy = accuracy_score(flat_pred, y_test_flat)

        df_results_original = df_results_original.append(pd.DataFrame({'Attack': [attack_name], 'Mode': [model_str], 'Sample': [sample_run], 'Random_Seed': [random_seed], 'Perturbation_Strength': [prob], 'ACC': [accuracy]}))
        
        suitcase = [df_results_original]
        pkl.dump(suitcase, open(filename,"wb") )

    end = time.time()
    print(end - start)
    return df_results_original


def TFIDF_SVD(N_gram, N_features, X_train_text, X_test_text):
    
    if len(X_train_text[0]) == 1:
        X_train_text = [item for sublist in X_train_text for item in sublist]
    if len(X_test_text[0]) == 1:
        X_test_text = [item for sublist in X_test_text for item in sublist]
            
    vectorizer_x = TfidfVectorizer(ngram_range=(1, N_gram))
    X_train = vectorizer_x.fit_transform(X_train_text)
    X_test = vectorizer_x.transform(X_test_text)

    svd = TruncatedSVD(n_components=N_features)
    X_train_reduced = svd.fit_transform(X_train)
    X_test_reduced = svd.transform(X_test)
  
    return vectorizer_x, svd, X_train_reduced, X_test_reduced

def vectorize_decompose_features(loaded_vectorizer, loaded_svd, X_train_text, X_test_text):
    
    if len(X_train_text[0]) == 1:
        X_train_text = [item for sublist in X_train_text for item in sublist]
    if len(X_test_text[0]) == 1:
        X_test_text = [item for sublist in X_test_text for item in sublist]
            
    X_train = loaded_vectorizer.fit_transform(X_train_text)
    X_test = loaded_vectorizer.transform(X_test_text)

    X_train_reduced = loaded_svd.fit_transform(X_train)
    X_test_reduced = loaded_svd.transform(X_test)
  
    return X_train_reduced, X_test_reduced

   
def robustness_curve_adv(attack_name_train, model, model_str, vectorizer_x, svd, perturbation_prob_vec, 
                     X_test_text, y_test_flat, samples, function_list, fun_index, random_seed):
    
    df_results_original = pd.DataFrame({'Perturbation_Train': [], 'Perturbation_Test': [], 'Mode': [], 'Sample': [], 'Random_Seed': [], 'Perturbation_Strength': [], 'ACC': []})
    attack_name = function_list[fun_index]
    #print(attack_name)
    
    for prob_index in range(0,len(perturbation_prob_vec)):
        #Perturb the Test Data -------------------
        prob = perturbation_prob_vec[prob_index]
        
        for sample_run in range(samples):
            X_test_text_modified = copy.deepcopy(X_test_text)
            inputs = range(len(X_test_text_modified))

            if attack_name == "HomoPhones":
                X_test_text_modified = perturb_text_homophones(oxford_dict, X_test_text, prob, function_list, fun_index)

            elif attack_name == "SimilarSymbols":
                X_test_text_modified = Parallel(n_jobs=num_cores)(delayed(perturb_text)([X_test_text[i]], prob, [SubstituteSimilarSymbolsFunction], 0) for i in inputs)

            elif attack_name == "NeighborKeyboard":
                X_test_text_modified = Parallel(n_jobs=num_cores)(delayed(perturb_text)([X_test_text[i]], prob, [SubstituteNeighborKeyboardFunction], 0) for i in inputs)

            ##############################################################################################
            
            if len(X_test_text_modified[0]) == 1:
                X_test_text_modified = [item for sublist in X_test_text_modified for item in sublist]

            X_test_mod = vectorizer_x.transform(X_test_text_modified) #TF-IDF
            X_test_mod_reduced = svd.transform(X_test_mod) #Singular Value Decomposition -> dim. reduction
            ##############################################################################################

            X_test_chunks = np.array_split(X_test_mod_reduced, num_cores)
            pred = Parallel(n_jobs=num_cores)(delayed(pred_chunks)(model, i) for i in X_test_chunks)
            flat_pred = [item for sublist in pred for item in sublist]
            accuracy = accuracy_score(flat_pred, y_test_flat)

            df_results_original = df_results_original.append(pd.DataFrame({'Perturbation_Train': [attack_name_train], 'Perturbation_Test': [attack_name], 'Mode': [model_str], 'Sample': [sample_run], 'Random_Seed': [random_seed], 'Perturbation_Strength': [prob], 'ACC': [accuracy]}))
            
    return df_results_original
# Comparing the Robustness of Classical and Deep Learning Techniques for Text Classification
Quynh Tran, Krystsina Shpileuskaya, Elaine Zaunseder, Larissa Putzar, Sven Blankenburg

WCCI 2022, Padua, Italy

This repository will contain our implementation of our WCCI 2022 paper submission with the
title "Comparing the Robustness of Classical and Deep Learning Techniques for Text Classification" and will be available as of July 2022.


# Folder structure

    ├── results_analysis          # numerical robustness curves of perturbed training as dataframe in pickle files
    │   ├── plots                 # numerical robustness curves for Microphone perturbation
    │   ├── results_unperturbed   # numerical robustness curves of classical and deep learning models (unperturbed training)
    │   └── training_perturbed    # numerical robustness curves of classical and deep learning models (perturbed training)
    └── train_test_models         # python scripts and notebooks to re-train classical and deep learning models 
    │   ├── classical_models      # 
    │   |   ├── SVM               # python scripts and notebooks to train and test support vector machines 
    │   └── deeplearning_model    # 
    │   |   ├── src               # python scripts to train and test deep learning models (VDCNN, Transformer) 
    │   |   |   ├── vdcnn         # python scripts to train and test robustness of Very Deep CNN  
    │   |   |   ├── transformer   # python scripts to train and test robustness of Transformer Models
    │   |   |   ├── ...
    
# How to run training scripts and download datasets
## SVM
To download the datasets (AG News, DbPedia, Yelp Review Polarity), please follow the link https://github.com/ArdalanM/nlp-benchmarks to the according datasets. 
You should ask for access and click on the "Request access" button. 
After downloading the datasets, create a folder called "datasets", including all datasets. The folder structure should look like that: 

    ├── datasets                              
    │   ├── ag_news                         
    │   ├── db_pedia                        
    │   └── yelp_review_polarity     
    │   │   ├── raw                         # for every dataset
    │   |   |   ├── prepare_data_new.ipynb  # for every dataset, creates 80-20 train-test splitting with five folds and two repeats. 

The raw folder should be created for every dataset.
The file "prepare_data_new.ipynb" should get copied from the SVM folder to the raw folder of the respective dataset. 

    

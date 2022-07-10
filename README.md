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
    

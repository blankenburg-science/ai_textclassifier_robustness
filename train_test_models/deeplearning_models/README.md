# nlp-benchmark base repository (origin)
This is a fork of the Github repository by ArdalanM: https://github.com/ArdalanM .

The original code can be found at: https://github.com/ArdalanM/nlp-benchmarks .

We have modified the original repository such that the robustness analysis of the 
chosen deep learning models (transformer and VDCNN) is possible. 
We have re-trained all models and kept the data-preprocessing and model parameters 
as were presented in the original repository.

Please consult the original repository for more further details regarding datasets and models.

All Models were trained on a Workstation with Ubuntu OS (20.04.4 LTS), 128GB RAM and 2 GPUs (Nvidia RTX 3090).


# Folder structure

    ├── training_perturbed              # numerical robustness curves of perturbed training as dataframe in pickle files
    │   ├── training_HomoPhones         # numerical robustness curves for Microphone perturbation
    │   ├── training_NeighborKeyboard   # numerical robustness curves for Keyboard perturbation
    │   └── training_SimilarSymbols     # numerical robustness curves for OCR perturbation 
    └── training_raw                    # numerical robustness curves for the 3 perturbations using unperturbed (raw) training
    └── src                             # modified source files of https://github.com/ArdalanM/nlp-benchmarks to include robustness
    

# Examples (Shell Scripts)
## 1. Step: Unperturbed (Raw) Training Example

```#!/bin/sh
$ src/vdcnn/train_ag_news.sh 
```

## 2. Step: Assess Robustness Example Curves for Trained (Raw) Model

```#!/bin/sh
$ src/vdcnn/robustness_ag_news.sh 
```

## 3. Step: Perturbed Training Example

```#!/bin/sh
$ src/vdcnn/train_perturbed_ag_news.sh 
```

## 4. Step: Assess Robustness Example Curves for Trained (Perturbed) Model

```#!/bin/sh
$ src/vdcnn/robustness_perturbed_ag_news.sh 
```

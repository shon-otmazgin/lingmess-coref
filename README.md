# LingMess: Linguistically Informed Multi Expert Scorers for Coreference Resolution
This repository is the official implementation of the paper ["LingMess: Linguistically Informed Multi Expert Scorers for Coreference Resolution"](https://arxiv.org/abs/2205.12644).

Credit: Many code parts were taken from [s2e-coref](https://github.com/yuvalkirstain/s2e-coref#requirements) repo.

## Table of contents

- [Environments and Requirements](#environments-and-requirements)
- Create Datasets
   * [Prepare OntoNotes dataset](#prepare-ontonotes-dataset)
   
     OR
  
   * [Prepare your own custom dataset](#prepare-your-own-custom-dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)

## Environments and Requirements

Below tested on `Ubuntu 20.04.3 LTS` with `Python 2.7` and `Python 3.7`
```
conda create -y --name py27 python=2.7
conda create -y --name lingmess-coref python=3.7 && conda activate lingmess-coref && pip install -r requirements.txt
```
Note: Python 2.7 is for OntoNotes dataset preprocess. 

## Create Datasets

### Prepare OntoNotes dataset

Download [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) corpus (registration is required).

Important: `ontonotes-release-5.0_LDC2013T19.tgz` must be under `data` folder.

Setup (~45 min):
```
cd prepare_onotonotes
chmod 755 setup.sh
./setup.sh
``` 
Credit: This script was taken from the [e2e-coref](https://github.com/kentonl/e2e-coref/) repo.

### Prepare your own custom dataset

Our implementation supports also custom dataset, both for training and inference.

Custom dataset guidelines:
1. Each dataset split (train/dev/test) should be in separate file.
2. Each file should be in `jsonlines` format
3. Each line in the file should have either `text` attribute or `tokens` attribute

option #1:
```
    {"doc_key": "DOC_KEY_1, "text": "this is document number 1, its text is raw text"},
```   
option #2:
```
    {"doc_key": "DOC_KEY_2, "tokens": ["this", "is", "document", "number", "1", ",", "it", "'s", "text", "is", "tokenized"],
```   
Recommended: add `doc_key` attribute as well for easy prediction tracking.

Note: For training on you own dataset the train file should contain gold clusters information as well.

## Training

## Evaluation

## Citation
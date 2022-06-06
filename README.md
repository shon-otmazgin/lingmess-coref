# LingMess: Linguistically Informed Multi Expert Scorers for Coreference Resolution
This repository is the official implementation of the paper ["LingMess: Linguistically Informed Multi Expert Scorers for Coreference Resolution"](https://arxiv.org/abs/2205.12644).

Credit: Many code parts were taken from [s2e-coref](https://github.com/yuvalkirstain/s2e-coref#requirements) repo.

# Table of contents

- [Environments and Requirements](#environments-and-requirements)
- Create Datasets
   * [Prepare OntoNotes dataset](#prepare-ontonotes-dataset)
   
     OR
  
   * [Prepare your own custom dataset](#prepare-your-own-custom-dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)

### Environments and Requirements

Below tested on `Ubuntu 20.04.3 LTS` with `Python 2.7` and `Python 3.7`
```
conda create -y --name py27 python=2.7
conda create -y --name lingmess-coref python=3.7 && conda activate lingmess-coref && pip install -r requirements.txt
```

### Create Datasets

#### Prepare OntoNotes dataset

Download [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) corpus (registration is required).

Important: `ontonotes-release-5.0_LDC2013T19.tgz` must be under `data` folder.

Run (~45 min):
```
cd data
chmod 755 setup_ontonotes.sh
./setup_ontonotes.sh
``` 
Credit: This script was taken from the [e2e-coref](https://github.com/kentonl/e2e-coref/) repo.

#### Prepare your own custom dataset

TBD

### Training

### Evaluation

### Citation
# lingmess-coref

### Table of contents

- [Environments and Requirements](#environments-and-requirements)
- Datasets
   * Ontonotes
     * [Prepare Ontonotes dataset](#onotonotes-prepare-dataset)
     * [Training](#ontonotes-train)
     * [Evaluation](#ontonotes-eval)
   * Custom
     * [Prepare your own custom dataset](#custom-prepare-dataset)
     * [Training](#custom-train)
     * [Evaluation](#custom-eval)
- [Cite](#cite)

## Environments and Requirements

Below tested on `Ubuntu 20.04.3 LTS` with `Python 2.7` and `Python 3.7`
```
conda create -y --name py27 python=2.7
conda create -y --name lingmess-coref python=3.7 && conda activate lingmess-coref && pip install -r requirements.txt
```

## Datasets

### Ontonotes

#### Prepare Ontonotes dataset

Download [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) corpus (registration is required).

Important: `ontonotes-release-5.0_LDC2013T19.tgz` must be under `data` folder.

Run (~45 min):
```
cd data
chmod 755 setup_ontonotes.sh
./setup_ontonotes.sh
``` 
Credit: This script was taken from the [e2e-coref](https://github.com/kentonl/e2e-coref/) repo.
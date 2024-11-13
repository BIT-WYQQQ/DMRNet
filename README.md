# DMRNet
DMRNet: Effective Network for Accurate Discharge Medication Recommendation  
For reproduction of medication prediction results in our paper, see instructions below.

## Overview
This repository contains code necessary to run DMRNet model. DMRNet is a three-module Discharge Medication Recommendation Network, called DMRNet, for accurate discharge medication recommendations. GAMENet is tested on real-world clinical dataset MIMIC-III and outperformed several state-of-the-art deep learning methods in heathcare area in all effectiveness measures from existing EHR data.

## Requirements
* Python == 3.8
* Pytorch == 1.11.0
* numpy == 1.20.1
* pandas == 1.2.4
* scikit-learn == 0.24.1
* dill == 0.3.6

## Running the code

### Data
In ./data, you can find the well-preprocessed data in pickle form.

Data information in ./data:
* records_final.pkl is the input data.
* voc_final.pkl is the vocabulary list to transform medical word to corresponding idx.
* ddi_A_final.pkl and ehr_adj_final.pkl are drug-drug adjacency matrix constructed from EHR and DDI dataset.
* dmo_adj_final.pkl is drug-medication co-occurence matrix constructed from EHR dataset.

### Model Comparasion
The codes of the baseline models can be found in ./src/baseline:
* **Nearest** simply recommends the discharge medications of the last visit for the current visit.
* **Leap** uses a sequence-to-sequence model based on LSTM. It predicts medications based on the current visit's diagnoses.
* **Transformer** adopts the traditional Transformer model. It encodes the patient's current diagnoses and procedures and uses multi-head attention layers to recommend discharge medications one after another.
* **Retain** is an RNN-based model. It uses the attention and gate mechanism to detect influential past visits of the patient and extract information from the useful diagnoses, procedures and medications of these visits.
* **GAMENet** is an RNN-based model that uses the Graph Convolution Network to capture the DDI and EHR relationship between medications to recommend the medications.
* **COGNet** is a Transformer-based model. It introduces a copy-or-predict mechanism that uses the patient's historical medications to generate an accurate medication recommendation.

### Ablation Study
The codes of the ablation models can be found in ./src/ablation:
* **DMRNet w/o Reserve:** The Medication Retention Module is removed.
* **DMRNet w/o History:** The History Retrieval Module is removed.
* **DMRNet w/o DMC\_e:** DMC Extraction is removed.
* **DMRNet w/o DMC\_g:** DMC Graph Enocder is removed.
* **DMRNet w/o Similarity:** The similarity between the historical visits and the current visit is removed.
* **DMRNet w/o Feature:** We remove the four features of the medications on admission and inpatient medications that are put into the encoders of the Medication Retention Module.

### Run DMRNet
```bash
python train_DMRNet.py [--model_name DMRNet] # train
python train_DMRNet.py --Test [--model_name DMRNet] [--resume_path Epoch_{}_JA_{}.model] # test
```

## Cite
Please cite our paper if you use this code in your own work:
```
@inproceedings{shi2024dmrnet,
  title={DMRNet: Effective Network for Accurate Discharge Medication Recommendation},
  author={Shi, Jiyun and Wang, Yuqiao and Zhang, Chi and Luo, Zhaojing and Chai, Chengliang and Zhang, Meihui},
  booktitle={2024 IEEE 40th International Conference on Data Engineering (ICDE)},
  pages={3393--3406},
  year={2024},
  organization={IEEE}
}
```

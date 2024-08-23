# EMGE-main
Code for paper EMGE：Entities and Mentions Gradual Enhancement with Semantics and Connection Modelling for Document-level Relation Extraction

# DataSet
The CDR and GDA datasets can be download from [FILR](https://github.com/luguo-ry/filr
)
# File Structure
the expected structure of files is:
 ```
EMGE-main/
│
├── dataset/
│   ├── cdr
|   |    |-- train_filter.data
|   |    |-- dev_filter.data
|   |    |-- test_filter.data
│   └── gda
|   |    |-- train.data
|   |    |-- dev.data
|   |    |-- test.data
│
├── meta/ 
│
└── saved_model/
│
└── sci_bert/
 ```
# Training and Evaluation
## Training
Train CDA and GDA model with the following command:
 ```
>> python trainCDR.py  # for CDR
>> python trainGDA.py  # for GDA
 ```
## Evaluation
You can download the saved models we reported in paper from [Baidu Netdisk](https://pan.baidu.com/s/1xTJlppSdb-2bxthQL0y10g?pwd=k5gd) and place them in --save_path. Then, you can evaluate the saved model by setting the --load_path argument, then the code will skip training and evaluate the saved model on benchmarks.

## related repo
Codes are adapted from the repo of the ACL2022 paper FILR [Document-level Biomedical Relation Extraction Based on Multi-Dimensional Fusion Information and Multi-Granularity Logical Reasoning.](https://aclanthology.org/2022.coling-1.183/#:~:text=Document-level%20biomedical%20relation%20extraction%20%28Bio-DocuRE%29%20is%20an%20important,extract%20all%20relation%20facts%20from%20the%20biomedical%20text.)


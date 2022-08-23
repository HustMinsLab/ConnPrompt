# ConnPrompt
This repository provides the code of ConnPrompt model for the COLING 2022 paper: ConnPrompt: Connective-cloze Prompt Learning for Implicit Discourse Relation Recognition.

# Data
We use the PDTB 3.0 corpus for evaluation. Due to the LDC policy, we cannot release the PDTB data. If you have bought data from LDC, please put the PDTB .tsv file in dataset.

# Requirements
python 3.7.9  
torch == 1.8.1  
transformers == 4.15.0

# How to use
- You have to put the PDTB corpus file in dataset file first.
- For each Pre-trained Language Model (PLM), run file
```
python main.py
```
- For the case of Multi-Prompt, after saving the `.pth` results of three different prompt template, run file：
```
python voting.py
```

# Citation
Please cite our paper if you use the code!

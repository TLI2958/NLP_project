# NLP Project
## Dataset
- Training set
    - [JTC](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data?select=test.csv.zip) | binary, used by [Das & Das](https://arxiv.org/pdf/2206.13284.pdf) to create weighted target score
    - [Unintended](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data) | averaged across 10 annotators
- Validation Set
    - [JRSTC](https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating/data) | paired validation data
        - Repetitions of pairs. Groupby and take the first pair?

## Models
- TF-IDF
    - [+ Ridge Regression](https://www.kaggle.com/code/nkitgupta/jigsaw-ridge-ensemble-tfidf-fasttext-0-868) | no pairs
    - [+ Naive Bayes](https://www.kaggle.com/code/julian3833/jigsaw-incredibly-simple-naive-bayes-0-768) | no pairs
    - [+ Logistic Regression](https://www.kaggle.com/code/kishalmandal/most-detailed-eda-tf-idf-and-logistic-reg/notebook) | do not use. use proba $\to$ score. 
- RoBERTa
- BERT
- M-BERT
    
## Progress
- [x] Preprocessing: see [preprocessing.ipynb](https://github.com/TLI2958/NLP_project/blob/main/preprocessing.ipynb) 
    - seed = 1011
    - downsample with threshold 0.01, rate = 0.1
    - make pairs: 1e5
    - text cleaning (optional): see [text_clean](https://github.com/TLI2958/NLP_project/blob/main/text_clean.py)
    - training set should now be downsampled, paired, and cleaned, named `train_paired_cleaned.csv`
    - val set should be cleaned, named `val_cleaned.csv`

- [ ] Core Train and Eval script: [main.py](https://github.com/TLI2958/NLP_project/blob/main/main.py)
    - [x] modified for this project
    - [ ] debug

- [ ] Run baseline models
    - [ ] TF-IDF
    - [ ] BERT-cased
    - [ ] RoBERTa
    - [ ] M-BERT

- [ ] [Augmentations](https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations)
    - random deletion/order switch
    - [butterfinger + homophones](https://github.com/TLI2958/NLP_project/blob/main/utils.py)
    <!-- - [homophones](https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/close_homophones_swap/transformation.py) | but slow. dictionary not useful. -->
    <!-- - [backtranslation](https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/back_translation) -->

- [ ] Run augmented models
    - [ ] TF-IDF
    - [ ] BERT-cased
    - [ ] RoBERTa
    - [ ] M-BERT
    
- [ ] Report
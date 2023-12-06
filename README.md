# NLP Project
## Dataset
- Training set
    - [Unintended](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data) | averaged across 10 annotators
<!--     - [JTC](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data?select=test.csv.zip) | binary, used by [Das & Das](https://arxiv.org/pdf/2206.13284.pdf) to create weighted target score -->
- Validation Set
    - [JRSTC](https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating/data) | paired validation data
        - Repetitions of pairs. Groupby and take the first pair?

## Models
- TF-IDF
- BERT
- RoBERTa
- M-BERT
    
## Progress
- [x] Preprocessing: see [preprocessing.ipynb](https://github.com/TLI2958/NLP_project/blob/main/preprocessing.ipynb) 
    - seed = 1011
    - downsample with threshold 0.01, rate = 0.1
    - make pairs: 1e5
    - text cleaning (optional): see [text_clean](https://github.com/TLI2958/NLP_project/blob/main/text_clean.py
        - training set should now be downsampled, paired, and cleaned, named `train_paired_cleaned.json`
        - val set should be cleaned, named `val_cleaned.json`
    - create DataDict object (optional)
  - [wordcloud](https://github.com/TLI2958/NLP_project/blob/main/words_visual.ipynb)


- [ ] Core Train and Eval script: [main.py](https://github.com/TLI2958/NLP_project/blob/main/main.py)
    - [x] modified for this project
    - [ ] debug

- [ ] Run baseline models
  - [ ] [TF-IDF](https://github.com/TLI2958/NLP_project/blob/main/TFIDF.ipynb)
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
    - [Proposal](https://www.overleaf.com/project/6536febce2491147b3a0598f)
    - slides
    - final report
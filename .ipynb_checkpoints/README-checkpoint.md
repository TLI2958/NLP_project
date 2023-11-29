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
    
## Questions
- Are we comparing pairs of texts? If so, we need to construct pairs manually, but by what criterion?
    - training with validation set (need [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)): 
        - [jigsaw_starter](https://www.kaggle.com/code/debarshichanda/pytorch-w-b-jigsaw-starter/notebook)
        - [roberta](https://www.kaggle.com/code/manabendrarout/pytorch-roberta-ranking-baseline-jrstc-train)
    
    - training with self-constructed pairs: 
        - [Das & Das](https://arxiv.org/pdf/2206.13284.pdf) | algorithm: $12 \cdot \text{severe toxic} + 9 \cdot \text{identity hate} + 8 \cdot \text{threat} + 6 \cdot \text{insult} + 5 \cdot \text{obscene} + 4 \cdot \text{toxic}$
        - If we don't construct pairs, we are likely to use RMSE as loss function. We can compare...
        
- Also track train & val loss
    - not classification any more. Do we still report auc_roc, f1, etc.? include them regardless. But we don't have labels if using their validation set.
    - Do we use [weights and bias](https://github.com/wandb/wandb) or build up a [loss monitor](https://www.kaggle.com/code/manabendrarout/pytorch-roberta-ranking-baseline-jrstc-train/notebook)(an example) to track avg loss? That is to say, we save model per epoch (or even per fold if we are replicating their implementations)?

- How many models to compare?
    - Now it seems redundant to compare ++ models if we are trying augmentations. 
    - Do we use our own baseline (i.e., plain texts)? Can we replicate results (If we are not using the same training set)?
    
- [Augmentations](https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations)
    - [butterfinger](https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/butter_fingers_perturbation/transformation.py)
    - [homophones](https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/close_homophones_swap/transformation.py) | but slow. dictionary not useful.
    - [backtranslation](https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/back_translation)
    
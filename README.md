# NLP Project
## Dataset
- Training set
    - [JTC](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data?select=test.csv.zip) | binary, used by [Das & Das](https://arxiv.org/pdf/2206.13284.pdf) to create weighted target score
    - [Unintended](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data) | averaged across 10 annotators
- Validation Set
    - [JRSTC](https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating/data) | paired validation data

## Questions
- Are we comparing pairs of texts? If so, we need to construct pairs manually, but by what criterion?
    - training with validation set (need [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)): 
        - [jigsaw_starter](https://www.kaggle.com/code/debarshichanda/pytorch-w-b-jigsaw-starter/notebook)
        - [roberta](https://www.kaggle.com/code/manabendrarout/pytorch-roberta-ranking-baseline-jrstc-train)
    
    - training with self-constructed pairs: 
        - [Das & Das](https://arxiv.org/pdf/2206.13284.pdf) | algorithm: $12 \prod \text{severe toxic} + 9 \prod \text{identity hate} + 8 \prod \text{threat} + 6 \prod \text{insult} + 5 \prod \text{obscene} + 4 \prod \text{toxic}$
        
- Also track train & val loss
    - not classification any more. Do we still report auc_roc, f1, etc.? include them regardless
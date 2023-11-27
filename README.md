# NLP_project
- Are we comparing pairs of texts? If so, we need to construct pairs manually, but by what criterion?
    - training with validation set (need [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)): 
        - [jigsaw_starter](https://www.kaggle.com/code/debarshichanda/pytorch-w-b-jigsaw-starter/notebook)
        - [roberta](https://www.kaggle.com/code/manabendrarout/pytorch-roberta-ranking-baseline-jrstc-train)
    
    - training with self-constructed pairs: 
        - [tf-idf](https://www.kaggle.com/code/kishalmandal/most-detailed-eda-tf-idf-and-logistic-reg/notebook)
        
- Also track train & val loss
    - not classification any more. Do we still report auc_roc, f1, etc.? include them regardless
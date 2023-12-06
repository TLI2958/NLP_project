from cProfile import label
from collections import defaultdict
from json import load
import datasets
import numpy as np
from datasets import DatasetDict, load_dataset, Features, Value
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm
import evaluate
import random
import argparse
import pickle
from utils import *
import os


## ref: https://www.kaggle.com/code/manabendrarout/pytorch-roberta-ranking-baseline-jrstc-train
# Random Seed Initialize
RANDOM_SEED = 1011

def seed_everything(seed=RANDOM_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

def tokenize_function(examples):
    tokenized_more_toxic = tokenizer(
        examples['more_toxic_text'],
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        return_attention_mask=True,
        return_token_type_ids=True,
    )

    tokenized_less_toxic = tokenizer(
        examples['less_toxic_text'],
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    ids_more_toxic = tokenized_more_toxic['input_ids']
    mask_more_toxic = tokenized_more_toxic['attention_mask']
    token_type_ids_more_toxic = tokenized_more_toxic['token_type_ids']
    
    ids_less_toxic = tokenized_less_toxic['input_ids']
    mask_less_toxic = tokenized_less_toxic['attention_mask']
    token_type_ids_less_toxic = tokenized_less_toxic['token_type_ids']
    
    return  {'ids_more_toxic': torch.tensor(ids_more_toxic, dtype=torch.long),
            'mask_more_toxic': torch.tensor(mask_more_toxic, dtype=torch.long),
            'token_type_ids_more_toxic': torch.tensor(token_type_ids_more_toxic, dtype=torch.long),
            'ids_less_toxic': torch.tensor(ids_less_toxic, dtype=torch.long),
            'mask_less_toxic': torch.tensor(mask_less_toxic, dtype=torch.long),
            'token_type_ids_less_toxic': torch.tensor(token_type_ids_less_toxic, dtype=torch.long),
            'target': torch.tensor(1, dtype=torch.float)} 
    
# Marginal Ranking Loss
# margin is a hyperparam to tune
# https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html
def criterion(more_toxic, less_toxic, target, margin = 0):
    return torch.nn.MarginRankingLoss(margin = margin)\
        (more_toxic, less_toxic, target)


# Core training function
def do_train(args, model, train_dataloader, scheduler = 'CosineAnnealingLR', save_dir="/.out"):
    # scheduler
    # CosineAnnealingLR: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    # CosineAnnealingWarmRestarts: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name=f"{scheduler}", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    model.train()
    progress_bar = tqdm(range(num_training_steps))
    
    ## to track training loss
    loss_tracker = defaultdict(lambda :{'loss': 0, 'count':0, 'avg': 0})
    
    ## metrics in case 
    metric_acc = evaluate.load("accuracy")
    metric_roc_auc = evaluate.load("roc_auc")
    metric_precision = evaluate.load("Precision")
    metric_recall = evaluate.load("Recall")
    metric_F1 = evaluate.load('F1')
    
    ## to calculate ECE/calibration
    with open(os.path.join(save_dir, f"{save_dir}.txt"), "w") as file:
        file.write("Confidence\tPrediction\tLabel\n")    
        for epoch in range(num_epochs): 
            for i, data in enumerate(train_dataloader):
                ## ref: https://www.kaggle.com/code/debarshichanda/pytorch-w-b-jigsaw-starter 
                
                ## use MarginRankingLoss
                ## MarginRankingLoss
                ## MarginRankingLoss
                more_toxic_ids = data['ids_more_toxic'].to(device, dtype=torch.long)
                more_toxic_mask = data['mask_more_toxic'].to(device, dtype=torch.long)
                more_toxic_labels = data['labels_more_toxic'].to(device, dtype=torch.long)

                less_toxic_ids = data['ids_less_toxic'].to(device, dtype=torch.long)
                less_toxic_mask = data['mask_less_toxic'].to(device, dtype=torch.long)
                less_toxic_labels = data['labels_less_toxic'].to(device, dtype=torch.long)
                targets = data['target'].to(device, dtype=torch.long)
                
                more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
                less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)
                
                ## more_toxic
                softmaxes = F.softmax(more_toxic_outputs.logits, dim=1)
                confidences, predictions = torch.max(softmaxes, 1)
                
                metric_acc.add_batch(predictions=predictions, references=data["labels_more_toxic"])
                metric_roc_auc.add_batch(predictions=predictions, references=data["labels_more_toxic"])
                metric_precision.add_batch(predictions=predictions, references=data["labels_more_toxic"])
                metric_recall.add_batch(predictions=predictions, references=data["labels_more_toxic"])
                metric_F1.add_batch(predictions=predictions, references=data["labels_more_toxic"])
                file.write(f'{confidences.to_list()}\t{predictions.to_list()}\t{more_toxic_labels.to_list()}\n')
                
                ## less_toxic
                softmaxes = F.softmax(less_toxic_outputs.logits, dim=1)
                confidences, predictions = torch.max(softmaxes, 1)
                
                metric_acc.add_batch(predictions=predictions, references=data["labels_less_toxic"])
                metric_roc_auc.add_batch(predictions=predictions, references=data["labels_less_toxic"])
                metric_precision.add_batch(predictions=predictions, references=data["labels_less_toxic"])
                metric_recall.add_batch(predictions=predictions, references=data["labels_less_toxic"])
                metric_F1.add_batch(predictions=predictions, references=data["labels_less_toxic"])
                file.write(f'{confidences.to_list()}\t{predictions.to_list()}\t{less_toxic_labels.to_list()}\n')
                
                ## track loss
                loss = criterion(more_toxic_outputs, less_toxic_outputs, targets)
                loss_tracker['loss'].append(loss.item())
                loss_tracker['count'] += 1
                loss_tracker['avg'].append(sum(loss_tracker['loss']) / loss_tracker['count'])

                ## backpropogation
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                optimizer.zero_grad()
                progress_bar.update(1)
                
        score_acc = metric_acc.compute()
        score_roc_auc = metric_roc_auc.compute()
        score_precision = metric_precision.compute()
        score_recall = metric_recall.compute()
        score_F1 = metric_F1.compute()
        metric_name = ['accuracy', 'roc_auc', 'Precision', 'Recall', 'F1']
        score = dict(zip(metric_name, [score_acc, score_roc_auc, score_precision, score_recall, score_F1]))
        print("Training completed...")
        print("Saving Model....")
        ## currently not compare models by epoch
        model.save_pretrained(save_dir)
    file.close()   
    
    # save metrics
    with open(os.path.join(save_dir, "train_metrics.pkl"), "wb") as pickle_file:
        pickle.dump(score, pickle_file)
    pickle_file.close()
    
    # save loss
    with open(os.path.join(save_dir, "train_loss_tracker.pkl"), "wb")  as pickle_file:
        pickle.dump(loss_tracker, pickle_file)
    pickle_file.close()
    return


# Core evaluation function
def do_eval(eval_dataloader, output_dir, out_file):
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()
    ## to track eval loss & acc
    loss_tracker = defaultdict(lambda :{'loss': 0, 'count':0, 'avg': 0})
    metric_acc = evaluate.load("accuracy")
    
    for batch in tqdm(eval_dataloader):
        ids_more_toxic = batch['ids_more_toxic'].to(device)
        mask_more_toxic = batch['mask_more_toxic'].to(device)
        token_type_ids_more_toxic = batch['token_type_ids_more_toxic'].to(device)
        ids_less_toxic = batch['ids_less_toxic'].to(device)
        mask_less_toxic = batch['mask_less_toxic'].to(device)
        token_type_ids_less_toxic = batch['token_type_ids_less_toxic'].to(device)
        target = batch['target'].to(device)
        with torch.no_grad():
            logits_more_toxic = model(ids_more_toxic, mask_more_toxic)
            logits_less_toxic = model(ids_less_toxic, mask_less_toxic)
            
            ## leave accuracy as metric only
            loss = criterion(logits_more_toxic, logits_less_toxic, target)
            loss_tracker['loss'].append(loss.item())
            loss_tracker['count'] += 1
            loss_tracker['avg'].append(sum(loss_tracker['loss']) / loss_tracker['count'])
            metric_acc.add_batch(predictions= (logits_more_toxic >= logits_less_toxic).long(), references=data["target"])
    
    with open(os.path.join(save_dir, "eval_metrics.pkl"), "wb") as pickle_file:
        pickle.dump(metric_acc.compute(), pickle_file)

    with open(os.path.join(save_dir, "eval_loss_tracker.pkl"), "wb") as pickle_file:
        pickle.dump(loss_tracker, pickle_file)
        
    print('Eval completed...')
    print(f'avg eval loss = {loss_tracker["avg"][-1]}\navg accuracy = {metric_acc.compute()}')
    return 



# Created a dataladoer for the augmented training dataset
def create_augmented_dataloader(args, dataset):

    chosen_dataset = dataset["train"].shuffle(seed = RANDOM_SEED).select(range(5000))
    augmented_dataset = chosen_dataset.map(custom_transform, batched=True, load_from_cache_file =False)
    augmented_dataset = torch.utils.data.ConcatDataset([dataset, augmented_dataset])

    # tokenize, remove, rename
    augmented_tokenized_dataset = augmented_dataset.map(tokenize_function, batched=True, load_from_cache_file=False)
    ## TODO: suppose we have such dataset
    augmented_tokenized_dataset = augmented_tokenized_dataset.remove_columns(["more_toxic_text", "less_toxic_text"])
    augmented_tokenized_dataset.set_format("torch")
    augmented_val_dataset = augmented_tokenized_dataset
    # create aug dataloader
    train_dataloader = DataLoader(augmented_val_dataset, shuffle = True, batch_size=args.batch_size)

    return train_dataloader


# Create a dataloader for the transformed test set
def create_transformed_dataloader(args, dataset, debug_transformation):
    # Print 5 random transformed examples
    if debug_transformation:
        small_dataset = dataset["test"].shuffle(seed = RANDOM_SEED).select(range(5))
        small_transformed_dataset = small_dataset.map(custom_transform, load_from_cache_file=False)
        for k in range(5):
            print("Original Example ", str(k))
            print(small_dataset[k])
            print("\n")
            print("Transformed Example ", str(k))
            print(small_transformed_dataset[k])
            print('=' * 30)

        exit()

    transformed_dataset = dataset["test"].map(custom_transform, load_from_cache_file=False)
    transformed_tokenized_dataset = transformed_dataset.map(tokenize_function, batched=True, load_from_cache_file=False)
    transformed_tokenized_dataset = transformed_tokenized_dataset.remove_columns(["more_toxic_text", "less_toxic_text"])
    transformed_tokenized_dataset.set_format("torch")

    transformed_val_dataset = transformed_tokenized_dataset
    eval_dataloader = DataLoader(transformed_val_dataset, batch_size=args.batch_size)

    return eval_dataloader


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--train", action="store_true", help="train a model on the training data")
    parser.add_argument("--train_augmented", action="store_true", help="train a model on the augmented training data")
    parser.add_argument("--eval", action="store_true", help="evaluate model on the test set")
    parser.add_argument("--eval_transformed", action="store_true", help="evaluate model on the transformed test set")
    parser.add_argument("--checkpoint", type = str, default = 'bert-base-cased')
    parser.add_argument("--save_dir", type = str, default = "./out")
    parser.add_argument("--model_dir", type=str, default="./out")
    parser.add_argument("--debug_train", action="store_true",
                        help="use a subset for training to debug your training loop")
    parser.add_argument("--debug_transformation", action="store_true",
                        help="print a few transformed examples for debugging")
    parser.add_argument("--transform_type", type=str, default="butterfinger")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--is_training", type = bool, default = True, help="Specify whether it's training or evaluation")


    args = parser.parse_args()
    
    global device
    global tokenizer
    # suppose we are in /scratch/<netID>
    ## TODO: we need to create a dataset object, combining train and val
    data_path =  '/scratch/' + os.environ.get("USER", "") + '/data/'
    seed_everything()

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") # tokenize

    # Tokenize the dataset
    ## load train and val into load_dataset
    dataset = DatasetDict.load_from_disk(data_path)
    tokenized_dataset = dataset.map(tokenize_function, batched = True)
    
    # Prepare dataset for use by model
    tokenized_dataset = tokenized_dataset.remove_columns(["more_toxic_text", "less_toxic_text"])
    tokenized_dataset.set_format("torch")
    
    small_train_dataset = tokenized_dataset["train"].shuffle(seed = RANDOM_SEED).select(range(4000))
    small_eval_dataset = tokenized_dataset["test"].shuffle(seed = RANDOM_SEED).select(range(1000))

    # Create dataloaders for iterating over the dataset
    if args.debug_train:
        train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=args.batch_size)
        eval_dataloader = DataLoader(small_eval_dataset, batch_size=args.batch_size)
        print(f"Debug training...")
        print(f"len(train_dataloader): {len(train_dataloader)}")
        print(f"len(eval_dataloader): {len(eval_dataloader)}")
    else:
        train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=args.batch_size)
        eval_dataloader = DataLoader(tokenized_dataset["test"], batch_size=args.batch_size)
        print(f"Actual training...")
        print(f"len(train_dataloader): {len(train_dataloader)}")
        print(f"len(eval_dataloader): {len(eval_dataloader)}")

    # Train model on the original training dataset
    if args.train:
        save_dir = os.path.basename(os.path.normpath(args.save_dir))
        ## TODO: we also need a model class so as to add dropout & fc layers.
        model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, num_labels=2)
        model.to(device)
        do_train(args, model, train_dataloader, save_dir=f"{save_dir}")
        # Change eval dir
        args.model_dir = f"{save_dir}"

    # Train model on the augmented training dataset
    if args.train_augmented:
        save_dir = os.path.basename(os.path.normpath(args.save_dir))
        train_dataloader = create_augmented_dataloader(args, dataset)
        model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, num_labels=2)
        model.to(device)
        do_train(args, model, train_dataloader, save_dir=f"{save_dir}")
        # Change eval dir
        args.model_dir = f"{save_dir}"

    # Evaluate the trained model on the original test dataset
    if args.eval:
        out_file = os.path.basename(os.path.normpath(args.model_dir))
        out_file = out_file + "_original.txt"
        do_eval(eval_dataloader, args.model_dir, out_file)
        # print(f"Marginal Ranking Loss: {mrl:.4f}")
        # for metric, value in score.items():
        #     print(f"{metric}: {value}")  

    # Evaluate the trained model on the transformed test dataset
    if args.eval_transformed:
        out_file = os.path.basename(os.path.normpath(args.model_dir))
        out_file = out_file + f"{args.transform_type}.txt"
        eval_transformed_dataloader = create_transformed_dataloader(args, dataset, args.debug_transformation)
        do_eval(eval_transformed_dataloader, args.model_dir, out_file)
        # print(f"Marginal Ranking Loss: {mrl:.4f}")
        # for metric, value in score.items():
        #     print(f"{metric}: {value}")  

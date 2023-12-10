from cProfile import label
from collections import defaultdict
from json import load
import datasets
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, Features, Value
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW, lr_scheduler
from transformers import get_scheduler
import torch
from torch.nn import functional as F
from tqdm import tqdm
import evaluate
import random
import argparse
import pickle
from utils import *
from BertDataSet import *
from ToxicityModel import *
import os
import torch.nn as nn



## ref: https://www.kaggle.com/code/manabendrarout/pytorch-roberta-ranking-baseline-jrstc-train
# Random Seed Initialize
RANDOM_SEED = 1011

def save_checkpoint(model, optimizer, epoch, i, args):
    checkpoint_path = f'{args.save_dir}/trained_{args.label}_checkpoint.pth'
    torch.save({
        'epoch': epoch,
        'iteration': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

    
def load_checkpoint(model, optimizer, args):
    checkpoint_path = f'{args.save_dir}/trained_{args.label}_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
        print(f"Checkpoint found. Resuming training from epoch {epoch}, iteration {iteration}.")
        return model, optimizer, epoch, iteration
    else:
        return model, optimizer, 0, 0

def seed_everything(seed=RANDOM_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

def small_data_set(dataset, split, size = 100):
    rng = np.random.default_rng(seed = RANDOM_SEED)
    small_set = [dataset[split][int(i)] for i in rng.integers(0, len(dataset[split]), size = size)]
    small_dataset = Dataset.from_list(small_set)

    return small_dataset

    
# Marginal Ranking Loss
# margin is a hyperparam to tune
# https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html
def criterion(more_toxic, less_toxic, target, margin = 0.5):
    return torch.nn.MarginRankingLoss(margin = margin)\
        (more_toxic, less_toxic, target)

# Core training function
def do_train(args, model, train_dataloader):
    # scheduler
    # CosineAnnealingLR: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
   
    optimizer = AdamW(model.parameters(), lr= args.learning_rate)
    model, optimizer, start_epoch, start_iteration = load_checkpoint(model, optimizer, args)

    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = args.T_max)
    
    model.train()
    progress_bar = tqdm(range(num_training_steps))
    
    ## to track training loss
    loss_tracker = {'loss': [], 'count':[0], 'avg': []}
    
    ## metrics in case 
    metric_acc = evaluate.load("accuracy")
    metric_roc_auc = evaluate.load("roc_auc")
    
    ## to calculate ECE/calibration
    with open(os.path.join(args.save_dir, f"ECE_{args.label}.txt"), "w") as file:
        file.write("Confidence\tPrediction\tLabel\n")    
        file.flush()
        for epoch in range(start_epoch, num_epochs): 
            if epoch < start_epoch:
                continue
            elif epoch > start_epoch:
                start_iteration = -1
            for i, data in enumerate(train_dataloader):
                if i < start_iteration:
                    continue
                else:
                    ## use MarginRankingLoss
                    more_toxic_ids = data['ids_more_toxic'].to(device, dtype=torch.long)
                    more_toxic_mask = data['mask_more_toxic'].to(device, dtype=torch.long)
                    more_toxic_labels = data['labels_more_toxic'].to(device, dtype=torch.long)

                    less_toxic_ids = data['ids_less_toxic'].to(device, dtype=torch.long)
                    less_toxic_mask = data['mask_less_toxic'].to(device, dtype=torch.long)
                    less_toxic_labels = data['labels_less_toxic'].to(device, dtype=torch.long)
                    targets = data['target'].to(device, dtype=torch.long)

                    more_toxic_single, more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
                    less_toxic_single, less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)
                    ## more_toxic
                    softmaxes = F.softmax(more_toxic_outputs, dim = 1)
                    confidences, predictions = torch.max(softmaxes, 1, keepdim = True)
                    predictions = predictions.to(dtype = torch.float32)

                    metric_acc.add_batch(predictions=predictions, references= more_toxic_labels)
                    metric_roc_auc.add_batch(prediction_scores =predictions.to(dtype = torch.int32), references = more_toxic_labels.to(dtype = torch.int32))
                    file.write(f'{confidences.tolist()}\t{predictions.tolist()}\t{more_toxic_labels.tolist()}\n')
                    file.flush()
                    ## less_toxic
                    softmaxes = F.softmax(less_toxic_outputs, dim = 1)
                    confidences, predictions = torch.max(softmaxes, 1, keepdim = True)
                    predictions = predictions.to(dtype = torch.float32)

                    metric_acc.add_batch(predictions=predictions, references=less_toxic_labels)
                    metric_roc_auc.add_batch(prediction_scores =predictions, references=less_toxic_labels)
                    file.write(f'{confidences.tolist()}\t{predictions.tolist()}\t{less_toxic_labels.tolist()}\n')
                    file.flush()
                    ## track loss
                    loss = criterion(more_toxic_single, less_toxic_single, targets.unsqueeze(1))
                    loss_tracker['loss'].append(loss.item())
                    loss_tracker['count'][-1] += 1
                    loss_tracker['avg'].append(sum(loss_tracker['loss']) / loss_tracker['count'][-1])

                    ## backpropogation
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    optimizer.zero_grad()
                    progress_bar.update(1)
                    if i % 1000 == 0:
                        save_checkpoint(model, optimizer, epoch, i, args)

        score_acc = metric_acc.compute()
        score_roc_auc = metric_roc_auc.compute()
        metric_name = ['accuracy', 'roc_auc']
        score = dict(zip(metric_name, [score_acc, score_roc_auc]))
        print("Training completed...")
        print("Saving Model....")
        ## currently not compare models by epoch
        torch.save(model.state_dict(), f'{args.save_dir}/trained_{args.label}.pth')
    file.close()   
    
    # save metrics
    with open(os.path.join(args.save_dir, "train_metrics.pkl"), "wb") as pickle_file:
        pickle.dump(score, pickle_file)
    pickle_file.close()
    
    # save loss
    with open(os.path.join(args.save_dir, "train_loss_tracker.pkl"), "wb")  as pickle_file:
        pickle.dump(loss_tracker, pickle_file)
    pickle_file.close()
    return


# Core evaluation function
def do_eval(eval_dataloader, output_dir):
    model = ToxicityModel()
    model.load_state_dict(torch.load(f'{output_dir}/trained_{args.label}.pth'))
    model.to(device)
    model.eval()
    ## to track eval loss & acc
    loss_tracker = {'loss': [], 'count':[0], 'avg': []}
    metric_acc = evaluate.load("accuracy")
    
    for batch in tqdm(eval_dataloader):
        ids_more_toxic = batch['ids_more_toxic'].to(device)
        mask_more_toxic = batch['mask_more_toxic'].to(device)
        token_type_ids_more_toxic = batch['token_type_ids_more_toxic'].to(device)
        ids_less_toxic = batch['ids_less_toxic'].to(device)
        mask_less_toxic = batch['mask_less_toxic'].to(device)
        token_type_ids_less_toxic = batch['token_type_ids_less_toxic'].to(device)
        targets = batch['target'].to(device, dtype=torch.long)

        with torch.no_grad():
            more_toxic_single, more_toxic_outputs = model(ids_more_toxic, mask_more_toxic)
            less_toxic_single, less_toxic_outputs = model(ids_less_toxic, mask_less_toxic)
            
            ## leave accuracy as metric only
            loss = criterion(more_toxic_single, less_toxic_single, targets.unsqueeze(1))
            loss_tracker['loss'].append(loss.item())
            loss_tracker['count'][-1] += 1
            loss_tracker['avg'].append(sum(loss_tracker['loss']) / loss_tracker['count'][-1])
            metric_acc.add_batch(predictions= (more_toxic_single >= less_toxic_single).to(dtype = torch.float32), 
                                 references=targets.unsqueeze(1))
    score = metric_acc.compute()
    with open(os.path.join(args.save_dir, "eval_metrics.pkl"), "wb") as pickle_file:
        pickle.dump(score, pickle_file)

    with open(os.path.join(args.save_dir, "eval_loss_tracker.pkl"), "wb") as pickle_file:
        pickle.dump(loss_tracker, pickle_file)
        
    print('Eval completed...')
#     print(f'avg eval loss = {loss_tracker["avg"][-1]}\navg accuracy = {metric_acc.compute()}')
    return 

# Created a dataloader for the augmented training dataset
def create_augmented_dataloader(args, train_dataset):
    # Print 5 random transformed examples
    if args.debug_augmentation:
        small_train_dataset = small_data_set(train_dataset, 'train', size = 5)
        small_augmented_dataset = small_train_dataset.map(custom_transform, load_from_cache_file=False)
        for k in range(5):
            print("Original Example ", str(k))
            print(small_train_dataset[k])
            print("\n")
            print("Augmented Example ", str(k))
            print(small_augmented_dataset[k])
            print('=' * 30)

        exit()

    chosen_dataset =  small_data_set(train_dataset, split ='train', size = 5000)
    
    augmented_dataset = chosen_dataset.map(custom_transform, batched=True, load_from_cache_file=False)
    augmented_dataset = datasets.concatenate_datasets([train_dataset['train'], augmented_dataset])
    # Tokenize, remove, rename
    augmented_tokenized_dataset = BERTDataset(
        more_toxic=augmented_dataset["more_toxic_text"],
        less_toxic=augmented_dataset["less_toxic_text"],
        labels_more_toxic=augmented_dataset["labels_more_toxic"],
        labels_less_toxic=augmented_dataset["labels_less_toxic"],
    )

    augmented_train_dataloader = DataLoader(augmented_tokenized_dataset,
                                            shuffle=True, num_workers = os.cpu_count(), 
                                            batch_size=args.batch_size)

    return augmented_train_dataloader


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--train", action="store_true", help="train a model on the training data")
    parser.add_argument("--train_augmented", action="store_true", help="train a model on the augmented training data")
    parser.add_argument("--eval", action="store_true", help="evaluate model on the test set")
    parser.add_argument("--eval_transformed", action="store_true", help="evaluate model on the transformed test set")
    parser.add_argument("--checkpoint", type = str, default = 'bert-base-cased')
    parser.add_argument("--save_dir", type = str, default = '/scratch/' + os.environ.get("USER", "") + '/out/')
    parser.add_argument("--model_dir", type=str, default="./out")
    parser.add_argument("--debug_train", action="store_true",
                        help="use a subset for training to debug your training loop")
    parser.add_argument("--debug_transformation", action="store_true",
                        help="print a few transformed examples for debugging")
    parser.add_argument("--debug_augmentation", action="store_true",
                        help="print a few augmented examples for debugging")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--T_max", type = int, default = 500)
    parser.add_argument("--label", type = str, default = "original")

    args = parser.parse_args()
    
    global device
    global tokenizer

    data_path =  '/scratch/' + os.environ.get("USER", "") + '/data/'
    seed_everything()

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Tokenize the dataset
    ## load train and val into load_dataset
    train_set, val_set = 'jigsaw_training/train_paired_cleaned.csv', 'jigsaw_validation/val_cleaned.csv' 
    print('begin loading...')
    train_dataset = load_dataset('csv', data_files = os.path.join(data_path, train_set))
    val_dataset = load_dataset('csv', data_files = {'test': os.path.join(data_path, val_set)})
    
    print('loaded dataset...')
    train_tokenized_dataset = BERTDataset(
        more_toxic=train_dataset["train"]["more_toxic_text"],
        less_toxic=train_dataset["train"]["less_toxic_text"],
        labels_more_toxic=train_dataset["train"]["labels_more_toxic"],
        labels_less_toxic=train_dataset["train"]["labels_less_toxic"],
    )

    test_tokenized_dataset = BERTDataset(
        more_toxic=val_dataset["test"]["more_toxic_text"],
        less_toxic=val_dataset["test"]["less_toxic_text"],
        labels_more_toxic=val_dataset["test"]["labels_more_toxic"],
        labels_less_toxic=val_dataset["test"]["labels_less_toxic"],
    )

    print('created BERTDataset class...')
    
    small_train_dataset = small_data_set(train_dataset, 'train')
    small_eval_dataset = small_data_set(val_dataset, 'test')
    small_train_tokenized = BERTDataset(more_toxic=small_train_dataset['more_toxic_text'],
                                          less_toxic= small_train_dataset['less_toxic_text'],
                                          labels_more_toxic= small_train_dataset['labels_more_toxic'],
                                          labels_less_toxic= small_train_dataset['labels_less_toxic'],)
    small_eval_tokenized = BERTDataset(more_toxic=small_eval_dataset['more_toxic_text'],
                                      less_toxic= small_eval_dataset['less_toxic_text'],
                                      labels_more_toxic= small_eval_dataset['labels_more_toxic'],
                                      labels_less_toxic= small_eval_dataset['labels_less_toxic'],)

    
    # Create dataloaders for iterating over the dataset
    if args.debug_train:
        train_dataloader = DataLoader(small_train_tokenized, shuffle=True, num_workers = os.cpu_count(), batch_size=args.batch_size)
        eval_dataloader = DataLoader(small_eval_tokenized, num_workers = os.cpu_count(), batch_size=args.batch_size)
        print(f"Debug training...")
        print(f"len(train_dataloader): {len(train_dataloader)}")
        print(f"len(eval_dataloader): {len(eval_dataloader)}")
    else:
        train_dataloader = DataLoader(train_tokenized_dataset, shuffle=True,  num_workers = os.cpu_count(), batch_size=args.batch_size)
        eval_dataloader = DataLoader(test_tokenized_dataset, num_workers = os.cpu_count(),
                                     batch_size=args.batch_size)
        print(f"Actual training...")
        print(f"len(train_dataloader): {len(train_dataloader)}")
        print(f"len(eval_dataloader): {len(eval_dataloader)}")

    # Train model on the original training dataset
    if args.train:
        save_dir = args.save_dir
        model = ToxicityModel()
        model.to(device)
      
        do_train(args, model, train_dataloader)
        # Change eval dir
        args.model_dir = f"{save_dir}"

    # Train model on the augmented training dataset
    if args.train_augmented:
        save_dir = args.save_dir
        train_dataloader = create_augmented_dataloader(args, train_dataset)
        model = ToxicityModel()
        model.to(device)
        
        do_train(args, model, train_dataloader)
        # Change eval dir
        args.model_dir = f"{save_dir}"

    # Evaluate the trained model on the original test dataset
    if args.eval:
        do_eval(eval_dataloader, args.model_dir)
        # print(f"Marginal Ranking Loss: {mrl:.4f}")
        # for metric, value in score.items():
        #     print(f"{metric}: {value}")  


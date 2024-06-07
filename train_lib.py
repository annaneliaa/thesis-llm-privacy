import csv
import os
import torch
from torch.utils.data import random_split

def text_to_csv(dir, train_file, val_file):
    with open(train_file, encoding='utf-8') as txtfile:
        all_text = txtfile.read()
    with open(os.path.join(dir, 'train.csv'), mode='w', encoding='utf-8') as csv_file:
        fieldnames = ['text']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'text': all_text})


    with open(val_file, encoding='utf-8') as txtfile:
        all_text = txtfile.read()
    with open(os.path.join(dir, 'validation.csv'), mode='w', encoding='utf-8') as csv_file:
        fieldnames = ['text']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'text': all_text})

def split_set_to_train_val(eval_percentage, dataset_path, output_dir, config):
    # Load the dataset
    with open(dataset_path, "r") as f:
        dataset = f.readlines()

    train_size = int(len(dataset) * (1- eval_percentage))
    eval_size = len(dataset) - train_size

    # Create a range of indices to map to exids later
    indices = list(range(len(dataset)))

    # Split the indices along with the dataset
    train_indices, eval_indices = random_split(indices, [train_size, eval_size])

    # Create the train and eval datasets using the indices
    train_dataset = [dataset[i] for i in train_indices]
    eval_dataset = [dataset[i] for i in eval_indices]

    # Save train and eval datasets to files
    train_out_file = config["train_file"]
    with open(train_out_file, "w") as f:
        f.writelines(train_dataset)

    # Save train and eval datasets to files
    val_out_file = config["validation_file"]
    with open(val_out_file, "w") as f:
        f.writelines(eval_dataset)

    text_to_csv(output_dir, train_out_file, val_out_file)
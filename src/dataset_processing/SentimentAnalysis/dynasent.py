import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_processing import set_seed, save_data
set_seed(0)

import re
import numpy as np
from datasets import load_from_disk, concatenate_datasets

def text_process(example):
    example["text"] = re.sub("\t", " ", example["text"])
    example["text"] = re.sub(" +", " ", example["text"])
    return example

label_mapping = {"negative":0, "positive":1, "neutral":2}
def label_process(example):
    example["label"] = label_mapping[example["label"]]
    return example


# load
dataset_r1 = load_from_disk("./datasets/raw/SentimentAnalysis/dynasent_r1")
dataset_r1 = dataset_r1.rename_column("sentence", "text").rename_column("gold_label", "label")
dataset_r1 = dataset_r1.map(text_process).map(label_process)
dataset_r2 = load_from_disk("./datasets/raw/SentimentAnalysis/dynasent_r2")
dataset_r2 = dataset_r2.rename_column("sentence", "text").rename_column("gold_label", "label")
dataset_r2 = dataset_r2.map(text_process).map(label_process)
# shuffle and split
train = concatenate_datasets([dataset_r1["train"], dataset_r2["train"]]).shuffle(0)
test = concatenate_datasets([dataset_r1["test"], dataset_r2["test"]]).shuffle(0)
labels = np.array(train["label"])

# compute max_length
train_len = len(train)
max_length = max(
                np.sum(np.where(labels==0, np.ones((train_len,)), np.zeros((train_len,)))),
                np.sum(np.where(labels==1, np.ones((train_len,)), np.zeros((train_len,)))),
                np.sum(np.where(labels==2, np.ones((train_len,)), np.zeros((train_len,))))
                )
print("max length:", max_length)
label_count = {0:0, 1:0, 2:0}

# train
train_dataset = []
for data in train:
    if label_count[data["label"]] < max_length:
        train_dataset.append((data["text"], data["label"]))
        label_count[data["label"]] += 1

# test
test_dataset = []
for data in test:
    test_dataset.append((data["text"], data["label"]))

# save
save_data(train_dataset, "./datasets/process/SentimentAnalysis/dynasent", "train")
save_data(test_dataset, "./datasets/process/SentimentAnalysis/dynasent", "test")
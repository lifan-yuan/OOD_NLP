import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_processing import set_seed, save_data
set_seed(0)

import re
import numpy as np
from datasets import load_from_disk

def text_process(example):
    example["text"] = re.sub("\t", " ", example["text"])
    example["text"] = re.sub(" +", " ", example["text"])
    return example
    
label_mapping = {0:0, 4:1, 2:2}
def label_process(example): # we only consider three categories
    example["label"] = label_mapping[example["label"]]
    return example


# load and shuffle
dataset = load_from_disk("./datasets/raw/SentimentAnalysis/yelp_review_full")
dataset = dataset.filter(lambda example: example["label"] in [0, 2, 4])
dataset = dataset.map(text_process).map(label_process).shuffle(0)
# split
train, test = dataset["train"], dataset["test"]
labels = np.array(train["label"])

# compute max_length
# train_len = len(train)
# max_length = max(
#                 np.sum(np.where(labels==0, np.ones((train_len,)), np.zeros((train_len,)))),
#                 np.sum(np.where(labels==1, np.ones((train_len,)), np.zeros((train_len,)))),
#                 np.sum(np.where(labels==2, np.ones((train_len,)), np.zeros((train_len,))))
#                 )
# print("max length:", max_length)
max_length = 10000
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
save_data(train_dataset, "./datasets/process/SentimentAnalysis/yelp", "train")
save_data(test_dataset, "./datasets/process/SentimentAnalysis/yelp", "test")
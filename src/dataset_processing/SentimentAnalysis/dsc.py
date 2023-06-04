import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_processing import set_seed, save_data
set_seed(0)

import os
import re
import random
import numpy as np
import pandas as pd

def text_process(text):
    text = re.sub("\t", " ", text)
    text = re.sub(" +", " ", text)
    return text

label_mapping = {-1:0, 1:1}
def label_process(label):
    label = label_mapping[label]
    return label

def read_data(path):
    dataset = pd.read_csv(path, sep="\t", header=None, names=["text", "label"])
    dataset["text"] = dataset["text"].apply(text_process)
    dataset["label"] = dataset["label"].apply(label_process)
    dataset = [{"text":data[0], "label":data[1]} for data in dataset.values]
    return dataset

# load and shuffle
base_path = "./datasets/raw/SentimentAnalysis/dsc"
files = os.listdir(base_path)
dataset = {"train":[], "test":[]}
for file in files:
    if "train.tsv" in file:
        dataset["train"].extend(read_data(f"{base_path}/{file}"))
    if "test.tsv" in file:
        dataset["test"].extend(read_data(f"{base_path}/{file}"))
random.shuffle(dataset["train"])
random.shuffle(dataset["test"])

# split
train, test = dataset["train"], dataset["test"]
labels = np.array([data["label"] for data in train])

# compute max_length
train_len = len(train)
max_length = max(
                np.sum(np.where(labels==0, np.ones((train_len,)), np.zeros((train_len,)))),
                np.sum(np.where(labels==1, np.ones((train_len,)), np.zeros((train_len,)))),
                )
print("max length:", max_length)
label_count = {0:0, 1:0}
    
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
save_data(train_dataset, "./datasets/process/SentimentAnalysis/dsc", "train")
save_data(test_dataset, "./datasets/process/SentimentAnalysis/dsc", "test")

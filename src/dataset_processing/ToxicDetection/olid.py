import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_processing import set_seed, save_data, clean
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

label_mapping = {"NOT": 0, "OFF": 1}
def label_process(label): 
    label = label_mapping[label]
    return label

def read_train_data(path):
    dataset = pd.read_csv(path, sep="\t", header=0)[["tweet", "subtask_a"]]
    dataset = dataset.rename(columns={"tweet":"text", "subtask_a":"label"})
    dataset["text"] = dataset["text"].apply(clean).apply(text_process)
    dataset["label"] = dataset["label"].apply(label_process)
    dataset = [{"text":data[0], "label":data[1]} for data in dataset.values]
    return dataset

def read_test_data(base_path, tweet_file, label_file):
    id_and_tweet = pd.read_csv(f"{base_path}/{tweet_file}", sep="\t").rename(columns={"tweet":"text"})
    id_and_tweet["text"] = id_and_tweet["text"].apply(clean)
    id_and_label = pd.read_csv(f"{base_path}/{label_file}", sep=",", names=["id", "label"])
    id_and_label["label"] = id_and_label["label"].apply(label_process)
    dataset = pd.merge(id_and_tweet, id_and_label, how='inner', on='id')[["text", "label"]]
    dataset = [{"text":data[0], "label":data[1]} for data in dataset.values]
    return dataset

# load and shuffle
base_path = "./datasets/raw/ToxicDetection/olid"
dataset = {}
dataset["train"] = read_train_data(f"{base_path}/olid-training-v1.0.tsv")
dataset["test"] = read_test_data(base_path, "testset-levela.tsv", "labels-levela.csv")
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
save_data(train_dataset, "./datasets/process/ToxicDetection/olid", "train")
save_data(test_dataset, "./datasets/process/ToxicDetection/olid", "test")

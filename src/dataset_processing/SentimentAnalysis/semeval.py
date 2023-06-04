import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_processing import set_seed, save_data
set_seed(0)

import re
import random
import numpy as np

def text_process(text):
    text = re.sub("\t", " ", text)
    text = re.sub(" +", " ", text)
    return text


label_mapping = {"negative":0, "positive":1, "neutral":2}
def read_data(path):
    dataset = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("\t")
            if len(line) != 3:
                print("len(line) != 3:", line)
                continue
            label = label_mapping[line[1].strip()]
            sentence = text_process(line[2].strip())
            if not str(label).isdigit():
                print("not str(label).isdigit()")
                continue
            dataset.append({"text":sentence, "label":int(label)})
    return dataset


# load and shuffle
base_path = "./datasets/raw/SentimentAnalysis/SemEval-2017-task-4//Subtask_A"
dataset = {
    "train": read_data(f"{base_path}/twitter-2016train-A.txt"), 
    "test": read_data(f"{base_path}/twitter-2016test-A.txt")
    }
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
save_data(train_dataset, "./datasets/process/SentimentAnalysis/semeval", "train")
save_data(test_dataset, "./datasets/process/SentimentAnalysis/semeval", "test")
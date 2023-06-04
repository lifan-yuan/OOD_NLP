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

# load and shuffle
hate_speech = load_from_disk("./datasets/raw/ToxicDetection/hate_speech")["train"].map(text_process).shuffle(0)
hate_speech = hate_speech.filter(lambda example: example["label"] in [0, 1])
print(hate_speech)
# split
train_len = int(0.8 * len(hate_speech))
train = hate_speech.select(range(len(hate_speech))[:train_len])
test = hate_speech.select(range(len(hate_speech))[train_len:])
labels = np.array(train["label"])

# compute max_length
max_length = max(
                np.sum(np.where(labels==0, np.ones((train_len,)), np.zeros((train_len,)))),
                np.sum(np.where(labels==1, np.ones((train_len,)), np.zeros((train_len,))))
                )
print("max length:", max_length)
label_count = {0:0, 1:0}

# train
print(train)
print(test)
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
save_data(train_dataset, "./datasets/process/ToxicDetection/hate_speech", "train")
save_data(test_dataset, "./datasets/process/ToxicDetection/hate_speech", "test")
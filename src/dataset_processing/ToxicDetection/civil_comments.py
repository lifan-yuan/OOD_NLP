import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_processing import set_seed, save_data
set_seed(0)

import re
from datasets import load_from_disk

def text_process(example):
    example["text"] = re.sub("\t", " ", example["text"])
    example["text"] = re.sub(" +", " ", example["text"])
    return example

def label_process(example): 
    example["label"] = 0 if example["toxicity"] < 0.5 else 1
    return example


# load and shuffle
dataset = load_from_disk("./datasets/raw/ToxicDetection/civil_comments").shuffle(0)
# split
train, test = dataset["train"].map(text_process).map(label_process), dataset["test"].map(text_process).map(label_process)

# manually set max_length
max_length = 30000
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
save_data(train_dataset, "./datasets/process/ToxicDetection/civil_comments", "train")
save_data(test_dataset, "./datasets/process/ToxicDetection/civil_comments", "test")
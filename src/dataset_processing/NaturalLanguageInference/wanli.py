import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_processing import set_seed, save_data
set_seed(0)

import re
from datasets import load_from_disk

def text_process(example):
    example["premise"] = re.sub("\t", " ", example["premise"])
    example["premise"] = re.sub(" +", " ", example["premise"])
    example["hypothesis"] = re.sub("\t", " ", example["hypothesis"])
    example["hypothesis"] = re.sub(" +", " ", example["hypothesis"])
    return example

LABEL_MAPPING = {"entailment": 0, "neutral": 1, "contradiction": 2}
def label_process(example):
    example["label"] = LABEL_MAPPING[example["gold"]]
    return example

# load and shuffle
dataset = load_from_disk("./datasets/raw/NaturalLanguageInference/wanli").map(text_process).map(label_process).shuffle(0)
# split
train, test = dataset["train"], dataset["test"]

# compute max_length
train_len = len(train)
max_length = 40000
label_count = {0:0, 1:0, 2:0}


# # train
train_dataset = []
for data in train:
    train_dataset.append((data["premise"], data["hypothesis"], data["label"]))


# test
test_dataset = []
for data in test:
    test_dataset.append((data["premise"], data["hypothesis"], data["label"]))

# save
save_data(train_dataset, "./datasets/process/NaturalLanguageInference/wanli", "train")
save_data(test_dataset, "./datasets/process/NaturalLanguageInference/wanli", "test")
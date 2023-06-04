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

# load and shuffle
dataset = load_from_disk("./datasets/raw/NaturalLanguageInference/multi_nli").map(text_process).shuffle(0)
# split
train, test = dataset["train"], dataset["validation_matched"]

# compute max_length
train_len = len(train)
max_length = 40000
label_count = {0:0, 1:0, 2:0}

# train
# train_dataset = []
# for data in train:
#     if label_count[data["label"]] < max_length:
#         train_dataset.append((data["premise"], data["hypothesis"], data["label"]))
#         label_count[data["label"]] += 1

# # train
train_dataset = []
for data in train:
    train_dataset.append((data["premise"], data["hypothesis"], data["label"]))


# test
test_dataset = []
for data in test:
    test_dataset.append((data["premise"], data["hypothesis"], data["label"]))

# save
save_data(train_dataset, "./datasets/process/NaturalLanguageInference/mnli", "train")
save_data(test_dataset, "./datasets/process/NaturalLanguageInference/mnli", "test")
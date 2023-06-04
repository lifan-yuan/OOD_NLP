import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_processing import set_seed, save_data
set_seed(0)

import re
from datasets import load_from_disk, concatenate_datasets

def text_process(example):
    example["premise"] = re.sub("\t", " ", example["premise"])
    example["premise"] = re.sub(" +", " ", example["premise"])
    example["hypothesis"] = re.sub("\t", " ", example["hypothesis"])
    example["hypothesis"] = re.sub(" +", " ", example["hypothesis"])
    return example

# load and shuffle
dataset = load_from_disk("./datasets/raw/NaturalLanguageInference/anli")
# split
train = concatenate_datasets([dataset["train_r1"], dataset["train_r2"], dataset["train_r3"]]).shuffle(0).map(text_process)
test = concatenate_datasets([dataset["test_r1"], dataset["test_r2"], dataset["test_r3"]]).shuffle(0).map(text_process)

# train
train_dataset = []
for data in train:
    train_dataset.append((data["premise"], data["hypothesis"], data["label"]))


# test
test_dataset = []
for data in test:
    test_dataset.append((data["premise"], data["hypothesis"], data["label"]))

# save
save_data(train_dataset, "./datasets/process/NaturalLanguageInference/anli", "train")
save_data(test_dataset, "./datasets/process/NaturalLanguageInference/anli", "test")
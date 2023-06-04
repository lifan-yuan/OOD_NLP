import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_processing import set_seed, save_data
set_seed(0)

import re
import json
import random

def text_process(text):
    text = re.sub("\t", " ", text)
    text = re.sub(" +", " ", text)
    return text

label_mapping = {"Entailment":0, "NotMentioned":1, "Contradiction":2}
def read_data(path):
    dataset = []
    with open(path, "r") as f:
        data = json.load(f)
        documents = data["documents"]
        for document in documents:
            premise = document["text"]
            for nda in data["labels"].keys():
                hypothesis = data["labels"][nda]["hypothesis"]
                label = label_mapping[document["annotation_sets"][0]["annotations"][nda]["choice"]]
                dataset.append({"premise":text_process(premise), "hypothesis":text_process(hypothesis), "label":label})
    return dataset

# load and shuffle
base_path = "./datasets/raw/NaturalLanguageInference/contract_nli"
dataset = {
    "train": read_data(f"{base_path}/train.json"),
    "test": read_data(f"{base_path}/test.json")
}

random.shuffle(dataset["train"])
random.shuffle(dataset["test"])
# split
train, test = dataset["train"], dataset["test"]

# train
train_dataset = []
for data in train:
    train_dataset.append((data["premise"], data["hypothesis"], data["label"]))

# test
test_dataset = []
for data in test:
    test_dataset.append((data["premise"], data["hypothesis"], data["label"]))

# save
save_data(train_dataset, "./datasets/process/NaturalLanguageInference/contract_nli", "train")
save_data(test_dataset, "./datasets/process/NaturalLanguageInference/contract_nli", "test")
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_processing import set_seed, save_ner_data
set_seed(0)

import pandas as pd
from datasets import Dataset

def label_process(example):
    example["tags"] = []
    for i, tag in enumerate(example["ner_tags"]):
        if i == 0:
            prefix = "B"
        else:
            prefix = "B" if example["ner_tags"][i-1] != tag else "I"

        tag = tag.split("-")[0]
        if tag == "O":
            example["tags"].append(tag)
        elif tag == "other":
            example["tags"].append(f"{prefix}-miscellaneous")
        else:
            example["tags"].append(f"{prefix}-{tag}")
    return example



def read_data(path):
    f = open(path, "r", encoding="utf-8")
    docs = f.read().split("\n\n")
    dataset = []

    for sentence in docs:
        token_list = []
        tag_list = []
        sentence = [line.strip().split("\t") for line in sentence.split("\n") if len(line.strip().split("\t"))==2]
        for token, tag in sentence:
            token_list.append(token)
            tag_list.append(tag)
        dataset.append({"tokens": token_list, "ner_tags": tag_list})
    
    dataset = pd.DataFrame(dataset, columns=["tokens", "ner_tags"])
    dataset = Dataset.from_pandas(dataset)
    return dataset

# load and shuffle
dataset = {
    "train": read_data("./datasets/raw/NameEntityRecognition/fewnerd/train.txt").map(label_process).shuffle(0),
    "test": read_data("./datasets/raw/NameEntityRecognition/fewnerd/test.txt").map(label_process).shuffle(0)
}

# split
train, test = dataset["train"], dataset["test"]

# train
train_dataset = []
for data in train:
    sentence = []
    for token, tag in zip(data["tokens"], data["tags"]):
        sentence.append((token, tag))
    train_dataset.append(sentence)

# test
test_dataset = []
for data in test:
    sentence = []
    for token, tag in zip(data["tokens"], data["tags"]):
        sentence.append((token, tag))
    test_dataset.append(sentence)

save_ner_data(train_dataset, "./datasets/process/NameEntityRecognition/fewnerd", "train")
save_ner_data(test_dataset, "./datasets/process/NameEntityRecognition/fewnerd", "test")
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_processing import set_seed, save_ner_data
set_seed(0)

from datasets import load_from_disk


label_mapping = {0: "O", 1: "B-organization", 2: "I-organization", 3: "B-art", 4: "I-art",
                 5: "B-organization", 6: "I-organization", 7: "B-location", 8: "I-location", 
                 9: "B-person", 10: "I-person", 11: "B-product", 12: "I-product"}


def label_process(example):
    example["tags"] = []
    for i, tag_id in enumerate(example["ner_tags"]):
        tag = label_mapping[tag_id]
        ###########################################################
        if tag != "O":
            prefix, tag = tag.split("-")
            tag = f"{prefix}-{tag}"
        ###########################################################
        example["tags"].append(tag)
    return example

# load and shuffle
dataset = load_from_disk("./datasets/raw/NameEntityRecognition/wnut").map(label_process).shuffle(0)

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

save_ner_data(train_dataset, "./datasets/process/NameEntityRecognition/wnut", "train")
save_ner_data(test_dataset, "./datasets/process/NameEntityRecognition/wnut", "test")
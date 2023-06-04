import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_processing import set_seed, save_ner_data
set_seed(0)

from datasets import load_from_disk

# ori_label_mapping = {0: "O", 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG',
#                      5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}

label_mapping = {0: "O", 1: "B-person", 2: "I-person", 3: "B-organization", 4: "I-organization",
                 5: "B-location", 6: "I-location", 7: "B-miscellaneous", 8: "I-miscellaneous"}

def label_process(example):
    example["tags"] = []
    for i, tag_id in enumerate(example["ner_tags"]):
        example["tags"].append(label_mapping[tag_id])
    return example

# load and shuffle
dataset = load_from_disk("./datasets/raw/NameEntityRecognition/conll").map(label_process).shuffle(0)

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

save_ner_data(train_dataset, "./datasets/process/NameEntityRecognition/conll", "train")
save_ner_data(test_dataset, "./datasets/process/NameEntityRecognition/conll", "test")
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_processing import set_seed, save_ner_data
set_seed(0)

import pandas as pd
from datasets import Dataset

label_mapping = {
    "O": "O",
    "LOC": "location",
    "PER": "person",
    "ORG": "organization",
    "MISC": "miscellaneous"
}

def label_process(example):
    example["tags"] = []
    for i, tag in enumerate(example["ner_tags"]):
        if i == 0:
            prefix = "B"
        else:
            prefix = "B" if example["ner_tags"][i-1] != tag else "I"

        if tag == "O":
            example["tags"].append(tag)
        else:
            tag = label_mapping[tag.split("-")[1]]
            example["tags"].append(f"{prefix}-{tag}")
    return example



def read_data(path):

    f = open(path, "r", encoding="utf-8")
    docs = f.read().split("\n\n")
    dataset = []

    for sentence in docs[1:]:# pass -DOCSTART-
        token_list = []
        tag_list = []

        sentence = [line.strip().split(" ") for line in sentence.split("\n") if len(line.strip().split(" "))==2]
        for token, tag in sentence:
            token_list.append(token)
            tag_list.append(tag)

        dataset.append({"tokens": token_list, "ner_tags": tag_list})
    
    dataset = pd.DataFrame(dataset, columns=["tokens", "ner_tags"])
    dataset = Dataset.from_pandas(dataset)
    return dataset

# load and shuffle
ener =  read_data("./datasets/raw/NameEntityRecognition/ener/edgar_all_4.csv").map(label_process).shuffle(0)


# # split
# train_len = int(0.8 * len(ener))
# train, test = ener.select([i for i in range(len(ener))][:train_len]), ener.select([i for i in range(len(ener))][train_len:])

# # train
# train_dataset = []
# for data in train:
#     sentence = []
#     for token, tag in zip(data["tokens"], data["tags"]):
#         sentence.append((token, tag))
#     train_dataset.append(sentence)

# # test
# test_dataset = []
# for data in test:
#     sentence = []
#     for token, tag in zip(data["tokens"], data["tags"]):
#         sentence.append((token, tag))
#     test_dataset.append(sentence)

# save_ner_data(train_dataset, "./datasets/process/NameEntityRecognition/ener", "train")
# save_ner_data(test_dataset, "./datasets/process/NameEntityRecognition/ener", "test")




# # split
test = ener

# test
test_dataset = []
for data in test:
    sentence = []
    for token, tag in zip(data["tokens"], data["tags"]):
        sentence.append((token, tag))
    test_dataset.append(sentence)

save_ner_data(test_dataset, "./datasets/process/NameEntityRecognition/ener", "test")
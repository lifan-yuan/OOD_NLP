import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_processing import set_seed, save_data
set_seed(0)

import os
import random
import numpy as np
import pandas as pd

import re

def text_process(text):
    text = re.sub(r'<re>|<er>|<le>|<el>', "", text)
    text = re.sub("\t", " ", text)
    text = re.sub(' +', ' ', text)
    return text

def label_process(label): 
    label = 0 if label == "pos" else 1
    return label

def read_data(path):
    dataset = pd.read_csv(path, sep=",", header=0)
    dataset = dataset[["abstract", "conclusion", "label_cat"]]
    dataset = dataset.rename(columns={"abstract":"premise", "conclusion":"hypothesis", "label_cat":"label"})
    dataset["premise"] = dataset["premise"].apply(text_process)
    dataset["hypothesis"] = dataset["hypothesis"].apply(text_process)
    dataset["label"] = dataset["label"].apply(label_process)
    dataset = [{"premise":data[0], "hypothesis":data[1], "label":data[2]}\
                     for data in dataset.values]
    return dataset

# load and shuffle
base_path = "./datasets/raw/NaturalLanguageInference/bio_nli"
dataset = {
    "train": read_data(f"{base_path}/train_df_blnc.csv"),
    "test": read_data(f"{base_path}/dev_df_blnc.csv")
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
save_data(train_dataset, "./datasets/process/NaturalLanguageInference/bio_nli", "train")
save_data(test_dataset, "./datasets/process/NaturalLanguageInference/bio_nli", "test")
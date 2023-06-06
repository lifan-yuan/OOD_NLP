import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_processing import set_seed, save_data
set_seed(0)

import os
import re
import random
import numpy as np
import pandas as pd

def text_process(text):
    text = re.sub("\t", " ", text)
    text = re.sub(" +", " ", text)
    return text

label_mapping = {"not_hate": 0, "implicit_hate": 1, "explicit_hate":1}
def label_process(label): 
    label = label_mapping[label]
    return label

def read_data(path):
    dataset = pd.read_csv(path, sep="\t", header=0)
    dataset = dataset.rename(columns={"post":"text", "class":"label"})
    dataset["text"] = dataset["text"].apply(text_process)
    dataset["label"] = dataset["label"].apply(label_process)
    dataset = [{"text":data[0], "label":data[1]} for data in dataset.values]
    return dataset

# load and shuffle
base_path = "./datasets/raw/ToxicDetection/implicit-hate/implicit-hate-corpus"
implicit_hate = read_data(f"{base_path}/implicit_hate_v1_stg1_posts.tsv")
random.shuffle(implicit_hate)

# split
test = implicit_hate

# test
test_dataset = []
for data in test:
    test_dataset.append((data["text"], data["label"]))

# save
save_data(test_dataset, "./datasets/process/ToxicDetection/implicit_hate", "test")
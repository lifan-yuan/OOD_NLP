import os
import random
import numpy as np
import pandas as pd
import torch

import emoji
import re

def set_seed(seed):
    print("set seed:", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def save_data(dataset, path, split):
    if len(dataset[0]) == 2:
        header = ["Text", "Label"]
    elif len(dataset[0]) == 3:
        header = ["Premise", "Hypothesis", "Label"]
    df = pd.DataFrame(dataset, columns=header)
    os.makedirs(path, exist_ok=True)
    df.to_csv(f"{path}/{split}.tsv", sep="\t", index=False, header=header)

def delete_emoji(data):
    response = re.sub('(:.*?:)', '', emoji.demojize(data))
    return response

def clean(text):
    text = re.sub(r"(//)?\s*@\S*?\s*(:| |$)", " ", text) # @username
    text = delete_emoji(text)
    text = re.sub(r"\[\S+\]", "", text)  # emoji
    text = re.sub(r"#\S+#", "", text)  # tag
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = re.sub(URL_REGEX, "", text)   # url
    text = text.replace("#", "")   # nonsense
    text = re.sub(r"\s+", " ", text) # space
    return text.strip("Subject").strip("Subject:").strip()


def save_ner_data(dataset, path, split):
    os.makedirs(path, exist_ok=True)
    f = open(f"{path}/{split}.tsv", "w")
    for sentence in dataset:
        for token, tag in sentence:
            f.write(token + "\t" + tag + "\n")
        f.write("\n")
    f.close()
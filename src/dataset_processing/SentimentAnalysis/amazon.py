import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_processing import set_seed, save_data
set_seed(0)

import os
import re
import json
import random
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

def text_process(text):
    text = re.sub("\t", " ", text)
    text = re.sub(" +", " ", text)
    return text

label_mapping = {1:0, 2:-1, 3:2, 4:-1, 5:1}
def label_process(label):
    label = label_mapping[label]
    return label

# load and shuffle
amazon = []
base_path = "./datasets/raw/SentimentAnalysis/amazon"
subsets = os.listdir(base_path)
subset_max_length = 20000

# For function "eval". Otherwise, 'NameError' rises.
true = True
false = False
for subset in subsets:
    f = open(f'{base_path}/{subset}','r')
    lines = f.readlines()
    random.shuffle(lines)
    # there may be some invalid samples, so we first take 2*subset_max_length samples, instead of subset_max_length.
    lines = [eval(line) for line in tqdm(lines[:int(2*subset_max_length)], desc=subset) if "reviewText" in line]
    lines = [{"text":line["reviewText"], "label":int(line["overall"])} for line in tqdm(lines[:subset_max_length], desc=subset)]

    amazon.extend(lines)

amazon = pd.DataFrame(amazon, columns=["text", "label"])
amazon["text"] = amazon["text"].apply(text_process)
amazon["label"] = amazon["label"].apply(label_process)
amazon = [{"text":data[0], "label":data[1]} for data in amazon.values if data[1] != -1]

random.shuffle(amazon)

# split
train_len = int(0.9 * len(amazon))
train, test = amazon[:train_len], amazon[train_len:]
labels = np.array([data["label"] for data in train])

# compute max_length
# max_length = max(
#                 np.sum(np.where(labels==0, np.ones((train_len,)), np.zeros((train_len,)))),
#                 np.sum(np.where(labels==1, np.ones((train_len,)), np.zeros((train_len,)))),
#                 np.sum(np.where(labels==2, np.ones((train_len,)), np.zeros((train_len,))))
#                 )
# print("max length:", max_length)
max_length = 10000
label_count = {0:0, 1:0, 2:0}

# train
train_dataset = []
for data in train:
    if label_count[data["label"]] < max_length:
        train_dataset.append((data["text"], data["label"]))
        label_count[data["label"]] += 1

# test
test_dataset = []
for data in test:
    test_dataset.append((data["text"], data["label"]))

# save
save_data(train_dataset, "./datasets/process/SentimentAnalysis/amazon", "train")
save_data(test_dataset, "./datasets/process/SentimentAnalysis/amazon", "test")


















# download

# urls = r"""
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/AMAZON_FASHION_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/All_Beauty_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Appliances_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Arts_Crafts_and_Sewing_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Automotive_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Books_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/CDs_and_Vinyl_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Cell_Phones_and_Accessories_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Clothing_Shoes_and_Jewelry_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Digital_Music_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Electronics_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Gift_Cards_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Grocery_and_Gourmet_Food_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Home_and_Kitchen_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Industrial_and_Scientific_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Kindle_Store_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Luxury_Beauty_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Magazine_Subscriptions_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Movies_and_TV_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Musical_Instruments_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Office_Products_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Patio_Lawn_and_Garden_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Pet_Supplies_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Prime_Pantry_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Software_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Sports_and_Outdoors_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Tools_and_Home_Improvement_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Toys_and_Games_5.json.gz
# https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz
# """.split("\n")

# urls = [url for url in urls if url != ""]
# import os

# os.system("cd /data/private/yuanlifan/oodbench_datasets/datasets/raw/Amazon")

# for url in urls:
#     os.system(f"wget {url} --no-check-certificate")
#     dataset = url.split("/")[-1]
#     os.system(f"gzip -d {dataset}")
#     dataset_name = dataset.strip(".json.gz")
#     new_name = dataset_name.strip("_5").lower()
#     os.system(f"mv {dataset_name}.json {new_name}.json")






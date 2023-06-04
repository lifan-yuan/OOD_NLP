import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_processing import set_seed, save_ner_data
set_seed(0)

import pandas as pd
from datasets import Dataset,concatenate_datasets

label_mapping = {
    'academicjournal': "product",  
    'album': "product",  
    'algorithm': "miscellaneous",  
    'astronomicalobject': "miscellaneous", 
    'award': "miscellaneous", 
    'band': "organization", 
    'book': "art",   
    'chemicalcompound': "miscellaneous", 
    'chemicalelement': "miscellaneous", 
    'conference': "event",  
    'country': "location", 
    'discipline': "miscellaneous",  
    'election': "event", 
    'enzyme': "miscellaneous", 
    'event': "event", 
    'field': "miscellaneous",  
    'literarygenre': "art", 
    'location': "location", 
    'magazine': "product",  
    'metrics': "miscellaneous",  
    'misc': "miscellaneous", 
    'musicalartist': "person", 
    'musicalinstrument': "product",  
    'musicgenre': "art", 
    'organisation': "organization", 
    'person': "person", 
    'poem': "art", 
    'politicalparty': "organization", 
    'politician': "person", 
    'product': "product", 
    'programlang': "miscellaneous",  
    'protein': "miscellaneous", 
    'researcher': "person", 
    'scientist': "person", 
    'song': "art", 
    'task': "miscellaneous",  
    'theory': "miscellaneous", 
    'university': "organization", 
    'writer': "person"
}

def label_process(example):
    example["tags"] = []
    for i, tag in enumerate(example["ner_tags"]):
        if tag == "O":
            example["tags"].append(tag)
        else:
            prefix, tag = tag.split("-")
            tag = label_mapping[tag]
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
    "train": {},
    "test": {},
}
basepath = "./datasets/raw/NameEntityRecognition/crossner"
for split in os.listdir(basepath):
    dataset["train"][split] = read_data(os.path.join(basepath, split, "train.txt"))
    dataset["test"][split] = read_data(os.path.join(basepath, split, "test.txt"))

dataset["train"] = concatenate_datasets([subset for split, subset in dataset["train"].items()])
dataset["test"] = concatenate_datasets([subset for split, subset in dataset["test"].items()])

dataset["train"] = dataset["train"].map(label_process).shuffle(0)
dataset["test"] = dataset["test"].map(label_process).shuffle(0)
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

save_ner_data(train_dataset, "./datasets/process/NameEntityRecognition/crossner", "train")
save_ner_data(test_dataset, "./datasets/process/NameEntityRecognition/crossner", "test")
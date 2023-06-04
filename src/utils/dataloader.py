import os
import datasets
import json, csv
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable

from openprompt.utils.logging import logger

from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor


class SentimentAnalysisProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["negative", "positive", "neutral"]
        self.label_mapping = {"negative":0, "positive":1, "neutral":2}

    def get_examples(self, data_dir, split):
        examples = []
        lines = pd.read_csv(os.path.join(data_dir, f"{split}.tsv"), sep="\t", header=0).values
        for idx, line in enumerate(lines):
            text_a = line[0]
            label = line[1]
            guid = "%s-%s" % (split, idx)
            
            if not isinstance(text_a, str):
                # print(line)
                continue

            try:
                example = InputExample(guid=guid, text_a=text_a, label=int(label))
            except:
                # print(line)
                continue
            examples.append(example)
        return examples


class ToxicDetectionProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["benign", "toxic"]
        self.label_mapping = {"benign":0, "toxic":1}

    def get_examples(self, data_dir, split):
        examples = []
        lines = pd.read_csv(os.path.join(data_dir, f"{split}.tsv"), sep="\t", header=0).values
        for idx, line in enumerate(lines):
            text_a = line[0]
            label = line[1]
            guid = "%s-%s" % (split, idx)
            
            if not isinstance(text_a, str):
                # print(line)
                continue

            try:
                example = InputExample(guid=guid, text_a=text_a, label=int(label))
            except:
                # print(line)
                continue
            examples.append(example)
        return examples


class NLIProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["Yes", "Maybe", "No"]
        self.label_mapping = {"Yes":0, "Maybe":1, "No": 2}

    def get_examples(self, data_dir, split):
        examples = []
        lines = pd.read_csv(os.path.join(data_dir, f"{split}.tsv"), sep="\t", header=0).values
        for idx, line in enumerate(lines):
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            guid = "%s-%s" % (split, idx)

            if not isinstance(text_a, str) or not isinstance(text_b, str):
                # print(line)
                continue

            try:
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=int(label))
            except:
                # print(line)
                continue
            examples.append(example)
        return examples



class QAProcessor(DataProcessor):
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        examples = []
        for line in open(os.path.join(data_dir, f"{split}.json"), "r"):
            examples.append(json.loads(line))     
        return examples



class NERProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        
        self.label_mapping = {  
                                "O": 0, 
                                "B-location": 1, "I-location": 2, 
                                "B-person": 3, "I-person": 4, 
                                "B-organization": 5, "I-organization": 6,
                                "B-miscellaneous": 7,"I-miscellaneous": 8,
                                "B-building": 9, "I-building": 10, 
                                "B-art": 11, "I-art": 12, 
                                "B-product": 13, "I-product": 14, 
                                "B-event": 15, "I-event": 16, 
                            }
        self.labels = list(self.label_mapping.keys())

    def get_examples(self, data_dir, split):
        f = open(os.path.join(data_dir, f"{split}.tsv"), "r", encoding="utf-8")
        docs = f.read().split("\n\n")
        examples = {'tokens':[], 'tags':[], "tag_ids":[]}

        for idx, sentence in enumerate(docs):
            guid = "%s-%s" % (split, idx)

            token_list = []
            tag_list = []
            sentence = [line.strip().split("\t") for line in sentence.split("\n") if len(line.strip().split("\t"))==2]
            
            token_list = [token for token, tag in sentence]
            tag_list = [tag for token, tag in sentence]
            tag_id_list = [self.label_mapping[tag] for tag in tag_list]

            if len(tag_list) > 0:
                examples["tokens"].append(token_list)
                examples["tags"].append(tag_list)
                examples["tag_ids"].append(tag_id_list)

        examples = pd.DataFrame(examples)
        examples = datasets.Dataset.from_pandas(examples)

        return examples


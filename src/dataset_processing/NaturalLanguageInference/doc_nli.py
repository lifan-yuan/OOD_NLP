import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_processing import set_seed, save_data
set_seed(0)

import re
import json
import random
import pandas as pd
import os
from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor
class NLIProcessor():
    
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
                print(line)
                # print("error")
                continue

            try:
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=int(label))
            except:
                print(line)
                continue
            examples.append(example)
        return examples

def text_process(text):
    text = re.sub("\t", " ", text)
    text = re.sub(" +", " ", text)
    return text

label_mapping = {"entailment":0, "not_entailment":1}
def read_data(path):
    dataset = []
    with open(path, "r") as f:
        lines = json.load(f)
        for line in lines:
            line["premise"] = text_process(line["premise"])
            line["hypothesis"] = text_process(line["hypothesis"])
            line["label"] = label_mapping[line["label"]]
            dataset.append(line)
    return dataset

# load and shuffle
base_path = "./datasets/raw/NaturalLanguageInference/doc_nli"
dataset = {
    # "train": read_data(f"{base_path}/train.json"),
    "test": read_data(f"{base_path}/test.json")
}
# random.shuffle(dataset["train"])
random.shuffle(dataset["test"])

# split
# train, test = dataset["train"], dataset["test"]
test = dataset["test"]

# train
# train_dataset = []
# for data in train:
#     train_dataset.append((data["premise"], data["hypothesis"], data["label"]))

# test
anli_test = NLIProcessor().get_examples("./datasets/process/NaturalLanguageInference/anli", "test")
anli_premise = {re.sub(r'[^\w\s]','',data.text_a) for data in anli_test}
anli_hypothesis = {re.sub(r'[^\w\s]','',data.text_b) for data in anli_test}
test_dataset = []
count = 0
import re
for data in test:
    if re.sub(r'[^\w\s]','',data["premise"]) in anli_premise or re.sub(r'[^\w\s]','',data["hypothesis"]) in anli_hypothesis:
        count += 1
        continue
    test_dataset.append((data["premise"], data["hypothesis"], data["label"]))

# # save
# save_data(train_dataset, "./datasets/process/NaturalLanguageInference/doc_nli", "train")
save_data(test_dataset, "./datasets/process/NaturalLanguageInference/doc_nli", "test")
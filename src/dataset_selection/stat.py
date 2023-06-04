import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json
import numpy as np
import pandas as pd

from src.utils import *
from src.utils.dataloader import *

def compute_stat(path, task_name, dataset_name):
    if task_name in ["SentimentAnalysis", "ToxicDetection"]:
        dataset_stat = {split:{"num_samples": "-", "avg_length": "-"} for split in ["train", "test"]}
        for split in ["train", "test"]:
            try:
                dataset = pd.read_csv(os.path.join(path, f"{split}.tsv"), sep="\t", header=0, names=["text", "label"])
                dataset_stat[split]["num_samples"] = len(dataset)
                dataset_stat[split]["avg_length"] = np.round(np.mean([len(text.split()) for text in dataset["text"] if isinstance(text, str)]), 2)
            except:
                continue
        dataset_statistics = [dataset_name, dataset_stat["train"]["num_samples"], dataset_stat["test"]["num_samples"], 
                              dataset_stat["train"]["avg_length"], dataset_stat["test"]["avg_length"]]
    elif task_name == "NaturalLanguageInference":
        dataset_stat = {split:{"num_samples": "-", "avg_length_p": "-", "avg_length_h":"-"} for split in ["train", "test"]}
        for split in ["train", "test"]:
            try:
                dataset = pd.read_csv(os.path.join(path, f"{split}.tsv"), sep="\t", header=0, names=["premise", "hypothesis", "label"])
                dataset_stat[split]["num_samples"] = len(dataset)
                dataset_stat[split]["avg_length_p"] = np.round(np.mean([len(text.split()) for text in dataset["premise"] if isinstance(text, str)]), 2)
                dataset_stat[split]["avg_length_h"] = np.round(np.mean([len(text.split()) for text in dataset["hypothesis"] if isinstance(text, str)]), 2)
            except:
                continue
        dataset_statistics = [dataset_name, dataset_stat["train"]["num_samples"], dataset_stat["test"]["num_samples"],
                              dataset_stat["train"]["avg_length_p"], dataset_stat["test"]["avg_length_p"], 
                              dataset_stat["train"]["avg_length_h"], dataset_stat["test"]["avg_length_h"]]
    elif task_name == "NameEntityRecognition":
        dataset_stat = {split:{"num_samples": "-", "avg_length": "-"} for split in ["train", "test"]}
        for split in ["train", "test"]:
            dataset = []
            try:
                f = open(os.path.join(path, f"{split}.tsv"), "r", encoding="utf-8")
            except:
                continue
            docs = f.read().split("\n\n")
            for sentence in docs:
                dataset.append([line.strip().split("\t")[0] for line in sentence.split("\n") if len(line.strip().split("\t"))==2])
            dataset_stat[split]["num_samples"] = len(dataset)
            dataset_stat[split]["avg_length"] = np.round(np.mean([len(token_list) for token_list in dataset]), 2)
        dataset_statistics = [dataset_name, dataset_stat["train"]["num_samples"], dataset_stat["test"]["num_samples"],
                              dataset_stat["train"]["avg_length"], dataset_stat["test"]["avg_length"]]
    elif task_name == "QuestionAnswering":
        dataset_stat = {split:{"num_samples": "-", "avg_length": "-"} for split in ["train", "test"]}
        for split in ["train", "test"]:
            dataset = []
            try:
                for line in open(os.path.join(path, f"{split}.json"), "r"):
                    dataset.append(json.loads(line))
            except:
                continue
            dataset_stat[split]["num_samples"] = len(dataset)
            dataset_stat[split]["avg_length"] = np.round(np.mean([len(data["context"].split()) for data in dataset]), 2)
        dataset_statistics = [dataset_name, dataset_stat["train"]["num_samples"], dataset_stat["test"]["num_samples"],
                              dataset_stat["train"]["avg_length"], dataset_stat["test"]["avg_length"]]
    elif task_name == "CommonSense":
        dataset_stat = {split:{"num_samples": "-", "avg_length_context": "-", "avg_length_option":"-"} for split in ["train", "test"]}
        for split in ["train", "test"]:
            try:
                dataset = pd.read_csv(os.path.join(path, f"{split}.tsv"), sep="\t", header=0, names=["context", "question", "option", "label"])
                dataset_stat[split]["num_samples"] = len(dataset)
                dataset_stat[split]["avg_length_context"] = np.round(np.mean([len(text.split()) for text in dataset["context"] if isinstance(text, str)]), 2)
                dataset_stat[split]["avg_length_option"] = np.round(np.mean([np.mean([len(text.split()) for text in eval(texts) if isinstance(text, str)]) for texts in dataset["option"] if isinstance(texts, str)]), 2)
            except:
                continue
        dataset_statistics = [dataset_name, dataset_stat["train"]["num_samples"], dataset_stat["test"]["num_samples"],
                              dataset_stat["train"]["avg_length_context"], dataset_stat["test"]["avg_length_context"], 
                              dataset_stat["train"]["avg_length_option"], dataset_stat["test"]["avg_length_option"]]
    print(dataset_statistics)
    return dataset_statistics

os.makedirs(f"./results/dataset_selection/statistics", exist_ok=True)
NAMES = {
    "SentimentAnalysis": ["Dataset", "train_num_sample", "test_num_sample", "train_avg_length", "test_avg_length"],
    "ToxicDetection": ["Dataset", "train_num_sample", "test_num_sample", "train_avg_length", "test_avg_length"], 
    "NaturalLanguageInference": ["Dataset", "train_num_sample", "test_num_sample", "train_avg_length_p", "test_avg_length_p", "train_avg_length_h", "test_avg_length_h"], 
    "NameEntityRecognition": ["Dataset", "train_num_sample", "test_num_sample", "train_avg_length", "test_avg_length"], 
    "QuestionAnswering": ["Dataset", "train_num_sample", "test_num_sample", "train_avg_length", "test_avg_length"],
    }
statistics = {
    "SentimentAnalysis": [],
    "ToxicDetection": [],
    "NaturalLanguageInference": [],
    "NameEntityRecognition": [],
    "QuestionAnswering": [],
}

for dataset_name, task_name in TASK.items():
    print(dataset_name)
    path = os.path.join(DATASET_PATH[task_name], dataset_name)
    dataset_statistics = compute_stat(path, task_name, dataset_name)
    statistics[task_name].append(dataset_statistics)
    print(statistics[task_name])

for task_name in statistics.keys():
    statistics[task_name] = pd.DataFrame(statistics[task_name], columns=NAMES[task_name])
    print(statistics[task_name])
    statistics[task_name].to_csv(f"./results/dataset_selection/statistics/{task_name}.tsv", sep="\t", index=False,
                        header=NAMES[task_name])

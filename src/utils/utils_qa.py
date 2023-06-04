import os
import pandas as pd
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


import random
import numpy as np
import torch



def wrap_template(raw_dataset, template):

    dataset = []
    for data in raw_dataset:
        src = deepcopy(template)

        context, question, answers = data["context"], data["question"], data["answers"]["text"]

        src = src.replace("{{question}}", question)
        src = src.replace("{{context}}", context)
        tgt = answers 

        dataset.append([src, tgt])

    return dataset


global tokenizer
def collate_fn(dataset, tokenizer):

        batch_src = []
        batch_tgt = []
        batch_size = len(dataset)

        for src, tgt in dataset:
            batch_src.append(src + " <extra_id_0>")
            batch_tgt.append(tgt)

        model_inputs = tokenizer(batch_src, max_length=768, padding=True, truncation=True, return_tensors="pt")
        
        tgt = [t[0] for t in batch_tgt]
        labels = tokenizer(text_target=tgt, max_length=20, padding=True, truncation=True, return_tensors="pt")["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["label"] = labels

        model_inputs["target_text"] = batch_tgt
        return model_inputs

import re
import string
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluation(prompt_model, test_dataloader, tokenizer, ood_name):

    prompt_model.eval()

    ground_truth_li = []
    predictions = []

    with torch.no_grad():
        em = 0
        f1 = 0
        count = 0
        for batch in test_dataloader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            tgt_text_li = batch["target_text"]
            outputs = prompt_model.generate(input_ids=input_ids, attention_mask=attention_mask)
            ground_truth_li.extend(tgt_text_li)
            for idx in range(outputs.shape[0]):
                predict = tokenizer.decode(outputs[idx], skip_special_tokens=True)
                context = tokenizer.decode(input_ids[idx], skip_special_tokens=True)
                if predict not in context:
                    count += 1
                predictions.append(predict.strip())

        for pred, ground_truth in zip(predictions, ground_truth_li):
            em += metric_max_over_ground_truths(exact_match_score, pred, ground_truth)
            f1 += metric_max_over_ground_truths(f1_score, pred, ground_truth)

    em = 100.0 * em / len(ground_truth_li)
    f1 = 100.0 * f1 / len(ground_truth_li)
    
    print('{} predictions not in the context, {}/{}={}'.format(count, count, len(predictions), count/len(predictions)))
    print('exact_match on {}: {}'.format(ood_name, em))
    print('f1 on {}: {}'.format(ood_name, f1))

    return em, f1



def eval(prompt_model, processor, tokenizer, ood_list, dataset_name, dataset_path, template, result_path):
    print("evaluation")
    
    dataset = {}
    for ood_name in ood_list:
        dataset[ood_name] = processor.get_examples(os.path.join(dataset_path, ood_name), "test")   
        dataset[ood_name] = wrap_template(dataset[ood_name], template)

            
    dataloader_dict = {}
    for ood_name in dataset.keys(): # including the test split
        if os.path.exists(f"./datasets/tokenize/QuestionAnswering/{ood_name}.pt"):
            print(f"load tokenized test dataset of {ood_name}")
            test_dataloader = torch.load(f"./datasets/tokenize/QuestionAnswering/{ood_name}.pt")
        else:
            test_dataloader = DataLoader(dataset[ood_name], shuffle=False, batch_size=32, collate_fn=collate_fn)
            batch_list = []
            for batch in test_dataloader:
                batch_list.append(batch)
            os.makedirs(f"./datasets/tokenize/QuestionAnswering", exist_ok=True)
            print(f"save tokenized test dataset of {ood_name}")
            torch.save(batch_list, f"./datasets/tokenize/QuestionAnswering/{ood_name}.pt")
        dataloader_dict[ood_name] = test_dataloader

    # evaluate performance
    print("Performance:")

    names = ["Dataset"]
    exact_match = ["Exact Match"]
    micro_f1 = ["F1"]
    for ood_name, test_dataloader in dataloader_dict.items():
        em, f1 = evaluation(prompt_model, test_dataloader, tokenizer, ood_name)
        names.append(ood_name)
        exact_match.append(em)
        micro_f1.append(f1)
    
    import pandas as pd 
    results = pd.DataFrame([exact_match, micro_f1], columns=names)
    results.to_csv(result_path, sep="\t", index=False)
    
    print("finish evaluation")






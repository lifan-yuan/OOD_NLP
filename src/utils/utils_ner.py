import os
import math
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from .dataloader import *
from openprompt.data_utils import InputExample, InputFeatures
from openprompt.utils import signature
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
from transformers import DataCollatorForTokenClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


import random
import numpy as np
import torch


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], max_length=256, padding=True, truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["tag_ids"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["label"] = labels
    
    return tokenized_inputs


from collections import Counter
def sampling_ner(dataset, num_classes, shots): # shots per tag

    dataset = dataset.shuffle()

    count = np.zeros((num_classes,), dtype=np.int64)
    sampled_dataset = []

    for i, data in enumerate(dataset):

        if data["tag_ids"].count(0) == len(data["tag_ids"]): # all "O"
            continue

        

        count_update = deepcopy(count)
        count_sentence = Counter(data["tag_ids"])

        required_tags = [tag_id for tag_id in range(num_classes) if count[tag_id] < shots]
        if all([count_sentence[tag_id] == 0 for tag_id in required_tags]): # no required tag in this sentence
            continue

        for tag_id in range(num_classes): # no "O"
            count_update[tag_id] += count_sentence[tag_id]
        
        # check if all elements are in [shots, 2*shots]
        num_entities = [item for item in count_update[1:]] # we do not add constrants to 0 (e.g., tag "O")
        if max(num_entities) > 2*shots: # invalid, should not update
            continue
        
        # update
        del count
        count = count_update
        sampled_dataset.append(i)

        if min(num_entities) >= shots: # enough
            break
            
    print(count)
    print(len(sampled_dataset))
    
    return dataset.select(sampled_dataset)



def evaluation(model, test_dataloader, ood_name, label_mapping):
    
    model.eval()

    references = []
    predictions = []
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            input_ids = batch["input_ids"].cuda()
            attention_masks = batch["attention_mask"].cuda()
            labels = batch["label"]

            tags = [[label_mapping[l] for l in label if l != -100] for label in labels]

            logits = model(input_ids=input_ids, attention_mask=attention_masks).logits
            probs = torch.nn.functional.softmax(logits, -1)

            # set the prob of a tag to 0 if the ood dataset does not have the specific tag
            assert probs.shape[-1] == 9 or probs.shape[-1] == 17
            if probs.shape[-1] == 17: # few-nerd tags
                if ood_name == "conll" or ood_name == "ener":
                    probs[..., 9:] = 0 # building, art, product, event
                elif ood_name == "wnut":
                    probs[..., 7:11] = 0 # miscellaneous, building
                    probs[..., 15:] = 0 # event
                elif ood_name == "crossner":
                    probs[..., 9:11] = 0 # building

            preds = probs.argmax(-1).detach().cpu().tolist()
            preds = [[label_mapping[p] for p, l in zip(pred, label) if l != -100] for pred, label in zip(preds, labels)]      

            references.extend(tags)
            predictions.extend(preds)

    metrics = datasets.load_metric("./utils/seqeval_metric.py")
    results = metrics.compute(predictions=predictions, references=references)

    model.train()
    print('f1 on {}: {}'.format(ood_name, results["overall_f1"]))
        
    return results["overall_precision"], results["overall_recall"], results["overall_f1"], results["overall_accuracy"]




def eval(model, processor, dataset_path, mytokenizer, result_path, task_name, ood_list, dataset_name, model_name, parameter=-1):
    print("evaluation")
    
    global tokenizer, soft_token_num
    tokenizer = mytokenizer
    soft_token_num = parameter
    dataset = {}
    for ood_name in ood_list:
        dataset[ood_name] = processor.get_examples(os.path.join(dataset_path, ood_name), "test")   
        dataset[ood_name] = dataset[ood_name].map(tokenize_and_align_labels, fn_kwargs={"tokenizer":tokenizer}, batched=True).remove_columns(["tokens", "tags", "tag_ids"])
        # break

    dataloader_dict = {}
    for ood_name in dataset.keys(): # including the test split
        if os.path.exists(f"./datasets/tokenize/NameEntityRecognition/{ood_name}.pt"):
            print(f"load tokenized test dataset of {ood_name}")
            test_dataloader = torch.load(f"./datasets/tokenize/NameEntityRecognition/{ood_name}.pt")
        else:
            # input_pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            data_collator = DataCollatorForTokenClassification(tokenizer)
            test_dataloader = DataLoader(dataset[ood_name], shuffle=False, batch_size=16, collate_fn=data_collator)
            batch_list = []
            for batch in test_dataloader:
                batch_list.append(batch)
            os.makedirs(f"./datasets/tokenize/NameEntityRecognition", exist_ok=True)
            print(f"save tokenized test dataset of {ood_name}")
            torch.save(batch_list, f"./datasets/tokenize/NameEntityRecognition/{ood_name}.pt")

        dataloader_dict[ood_name] = test_dataloader

    # evaluate performance
    print("Performance:")

    names = ["Dataset"]
    precision = ["Precision"]
    recall = ["Recall"]
    micro_f1 = ["F1"]
    accuracies = ["Acc"]
    for ood_name, test_dataloader in dataloader_dict.items():
        p, r, f1, acc = evaluation(model, test_dataloader, ood_name, processor.labels)
        names.append(ood_name)
        precision.append(100.00 * p)
        recall.append(100.00 * r)
        micro_f1.append(100.00 * f1)
        accuracies.append(100.00 * acc)

    import pandas as pd 
    results = pd.DataFrame([precision, recall, micro_f1, accuracies], columns=names)
    results.to_csv(result_path, sep="\t", index=False)
    
    print("finish evaluation")



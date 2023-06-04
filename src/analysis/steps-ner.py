import argparse
import math
import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.dataloader import *
from src.utils import *
from src.utils.utils_ner import *




def main(args):
    num_classes = args.num_classes
    model_name = args.model_name
    model_path = args.model_path
    task_name = args.task_name
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    scale = args.scale
    seed = args.seed
    set_seed(seed)

    ood_list = OOD_LIST[task_name]
    
    dataset = {}

    processor = PROCESSoR[task_name]()
    dataset['train'] = processor.get_examples(os.path.join(dataset_path, dataset_name), "train")
    dataset['test'] = processor.get_examples(os.path.join(dataset_path, dataset_name), "test")
    print(dataset_name)
    print(len(dataset['train']))

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    ################# train #################
    print("Train from scratch")

    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=num_classes).cuda()

    dataset['train'] = dataset['train'].map(tokenize_and_align_labels, fn_kwargs={"tokenizer":tokenizer}, batched=True).remove_columns(["tokens", "tags", "tag_ids"])
    train_dataloader = DataLoader(dataset['train'], shuffle=True, batch_size=16, collate_fn=data_collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model.train()
    train_length = len(train_dataloader)

    steps_dict = {}
    node_list = [10**i for i in range(-10, 0)]
    node_list.extend([i/10 for i in range(1, 10)])
    for node in node_list:
        if int(train_length * node) > 0:
            steps_dict[int(train_length * node)] = node

    for epoch in range(10):
        tot_loss = 0
        for step, batch in enumerate(train_dataloader):
            
            if (epoch == 0 and step in steps_dict.keys()) or (step == 0):
                node = epoch if step == 0 else steps_dict[step]
                print(node, "epochs")
                os.makedirs(f"./results/analysis/steps/{task_name}/{dataset_name}/{model_name}-{scale}/{node}-epochs", exist_ok=True)
                result_path = f"./results/analysis/steps/{task_name}/{dataset_name}/{model_name}-{scale}/{node}-epochs/{seed}.tsv"
                eval(model, processor, dataset_path, tokenizer, result_path, task_name, ood_list, dataset_name, model_name)

            input_ids = batch["input_ids"].cuda()
            attention_masks = batch["attention_mask"].cuda()
            labels = batch["label"].cuda()

            loss = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels).loss

            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 == 1:
                print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)

    print("finish training")

    
    epoch += 1
    os.makedirs(f"./results/analysis/steps/{task_name}/{dataset_name}/{model_name}-{scale}/{epoch}-epochs", exist_ok=True)
    result_path = f"./results/analysis/steps/{task_name}/{dataset_name}/{model_name}-{scale}/{epoch}-epochs/{seed}.tsv"
    eval(model, processor, dataset_path, tokenizer, result_path, task_name, ood_list, dataset_name, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--model_name', type=str, default="deberta")
    parser.add_argument('--dataset_name', type=str, default="conll")
    parser.add_argument('--scale', type=str, default="base")
    args = parser.parse_args()
    device = torch.device("cuda")
    
    args.task_name = TASK[args.dataset_name]
    args.model_path = MODEL_PATH[f"{args.model_name}-{args.scale}"]
    args.dataset_path = DATASET_PATH[args.task_name]
    args.num_classes = NUM_CLASSES[args.task_name]

    for i in range(args.repeats):
        set_seed(i)
        args.seed = i
        main(args)
    

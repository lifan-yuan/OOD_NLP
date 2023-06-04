import argparse
import math
import os
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
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
    shots = args.shots
    scale = args.scale
    seed = args.seed

    print(f"{shots=}")

    ood_list = OOD_LIST[task_name]
    

    sampled_dataset = deepcopy(dataset)
    if shots != -1 and shots != 0: # requires sampling
        sampled_dataset["train"] = sampling(sampled_dataset["train"], num_classes, shots)

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=num_classes).cuda()

    if shots != 0:
        print("Train from scratch")

        sampled_dataset['train'] = sampled_dataset['train'].map(tokenize_and_align_labels, fn_kwargs={"tokenizer":tokenizer}, batched=True).remove_columns(["tokens", "tags", "tag_ids"])
        train_dataloader = DataLoader(sampled_dataset['train'], shuffle=True, batch_size=16, collate_fn=data_collator)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        model.train()
        
        for epoch in range(10):
            tot_loss = 0
            for step, batch in enumerate(train_dataloader):

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

    os.makedirs(f"./results/analysis/shots/{task_name}/{dataset_name}/{model_name}-{scale}/{shots}-shots", exist_ok=True)
    result_path = f"./results/analysis/shots/{task_name}/{dataset_name}/{model_name}-{scale}/{shots}-shots/{seed}.tsv"
    eval(model, processor, dataset_path, tokenizer, result_path, task_name, ood_list, dataset_name, model_name)
    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=-1)
    parser.add_argument('--model_name', type=str, default="deberta")
    parser.add_argument('--dataset_name', type=str, default="fewnerd")
    parser.add_argument('--shots_list', type=int, default=[0], nargs="+")
    parser.add_argument('--scale', type=str, default="base")
    args = parser.parse_args()
    device = torch.device("cuda")
    
    args.task_name = TASK[args.dataset_name]
    args.model_path = MODEL_PATH[f"{args.model_name}-{args.scale}"]
    args.dataset_path = DATASET_PATH[args.task_name]
    args.num_classes = NUM_CLASSES[args.task_name]

    global dataset
    dataset = {}
    processor = PROCESSOR[args.task_name]()
    dataset['train'] = processor.get_examples(os.path.join(args.dataset_path, args.dataset_name), "train")
    print(args.dataset_name)
    print(len(dataset['train']))


    # automatically run few-shot settings
    if args.repeats == -1:
        shots_list = [0]
        shots_list.extend([2**i for i in range(25)])
        for shots in shots_list:
            args.shots = shots
            if shots * args.num_classes > len(dataset["train"]):
                print("Too much samples!")
                print("Exit!")
                exit()

            if shots <= 0:
                repeats = 1 # full data
            elif shots < 100:
                repeats = 5
            elif shots < 1000:
                repeats = 3
            else:
                repeats = 1
            
            for i in range(repeats):
                set_seed(i)
                args.seed = i
                main(args)

    else:
        repeats = args.repeats
        shots_list = args.shots_list
        for shots in shots_list:
            args.shots = shots
            if shots * args.num_classes > len(dataset["train"]):
                print("Too much samples!")
                print("Exit!")
                exit()

            for i in range(repeats):
                set_seed(i)
                args.seed = i
                main(args)

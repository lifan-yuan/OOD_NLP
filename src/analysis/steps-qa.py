import argparse
import math
import json
import os
import numpy as np
from tqdm import tqdm
import torch
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import datasets
from datasets import load_from_disk
import random
from copy import deepcopy
from transformers import T5Tokenizer, T5ForConditionalGeneration

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.dataloader import *
from src.utils import *
from src.utils.utils_qa import *

def main(args):
 
    model_name = args.model_name
    model_path = args.model_path
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    scale = args.scale

    seed = args.seed

    ood_list = OOD_LIST["QuestionAnswering"]
    
    with open(f"./prompts/QuestionAnswering/manual_template.txt", "r") as f:
            lines = f.readlines()
            template = "\n".join(lines)

    processor = QAProcessor()
    dataset = processor.get_examples(os.path.join(dataset_path, dataset_name), "train")        
    print(dataset_name)
    print(len(dataset))
    dataset = wrap_template(dataset, template)

    print("Train from scratch")
    
    global tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    prompt_model = T5ForConditionalGeneration.from_pretrained(model_path)
    prompt_model.cuda()    
    if scale == "large":
        prompt_model.parallelize()
        
    train_loader = DataLoader(dataset, shuffle=True, batch_size=16, collate_fn=lambda x: collate_fn(x, tokenizer))

    optimizer = torch.optim.AdamW(prompt_model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=5 * len(train_loader))
    
    loss_func = nn.CrossEntropyLoss()

    ############## train ##############
    prompt_model.train()
    train_length = len(train_loader)

    steps_dict = {}
    node_list = [10**i for i in range(-10, 0)]
    node_list.extend([i/10 for i in range(1, 10)])
    for node in node_list:
        if int(train_length * node) > 0:
            steps_dict[int(train_length * node)] = node

    for epoch in range(5):
        
        tot_loss = 0
        for step, inputs in enumerate(train_loader):

            if (epoch == 0 and step in steps_dict.keys()) or (step == 0):
                node = epoch if step == 0 else steps_dict[step]
                print(node, "epochs")
                os.makedirs(f"./results/analysis/steps/QuestionAnswering/{dataset_name}/{model_name}-{scale}/{node}-epochs", exist_ok=True)
                result_path = f"./results/analysis/steps/QuestionAnswering/{dataset_name}/{model_name}-{scale}/{node}-epochs/{seed}.tsv"
                eval(prompt_model, processor, tokenizer, ood_list, dataset_name, dataset_path, template, result_path)

            input_ids = inputs["input_ids"].cuda()
            attention_mask = inputs["attention_mask"].cuda()
            labels = inputs["label"].cuda()
            loss = prompt_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

            optimizer.zero_grad()
            loss.backward()

            tot_loss += loss.item()
            optimizer.step()

            if step % 100 == 1:
                print("Epoch {}, average loss: {}".format(epoch+1, tot_loss / (step+1)), flush=True)


    epoch += 1
    os.makedirs(f"./results/analysis/steps/QuestionAnswering/{dataset_name}/{model_name}-{scale}/{epoch}-epochs", exist_ok=True)
    result_path = f"./results/analysis/steps/QuestionAnswering/{dataset_name}/{model_name}-{scale}/{epoch}-epochs/{seed}.tsv"
    eval(prompt_model, processor, tokenizer, ood_list, dataset_name, dataset_path, template, result_path)
    print("finish all!")
    


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--model_name', type=str, default="t5")
    parser.add_argument('--dataset_name', type=str, default="squad")
    parser.add_argument('--scale', type=str, default="base")
    args = parser.parse_args()
    device = torch.device("cuda")
    
    args.model_path = MODEL_PATH[f"{args.model_name}-{args.scale}"]
    args.dataset_path = "./datasets/process/QuestionAnswering"

    for i in range(args.repeats):
        set_seed(i)
        args.seed = i
        main(args)
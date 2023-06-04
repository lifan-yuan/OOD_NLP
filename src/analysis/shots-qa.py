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
    shots = args.shots
    scale = args.scale
    incontext = args.incontext
    seed = args.seed

    print(f"{shots=}")

    ood_list = OOD_LIST["QuestionAnswering"]
    

    global dataset
    sampled_dataset = deepcopy(dataset)
    if shots != -1 and shots != 0:
        random.shuffle(sampled_dataset)
        sampled_dataset = sampled_dataset[:shots]
    
    global tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    prompt_model = T5ForConditionalGeneration.from_pretrained(model_path).cuda()
    prompt_model.parallelize()

    if shots != 0:
        print("Train from scratch")
        train_loader = DataLoader(sampled_dataset, shuffle=True, batch_size=16, collate_fn=lambda x: collate_fn(x, tokenizer))

        optimizer = torch.optim.AdamW(prompt_model.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=5 * len(train_loader))
        
        loss_func = nn.CrossEntropyLoss()

        ############## train ##############
        prompt_model.train()

        for epoch in range(5):
            
            tot_loss = 0
            for step, inputs in enumerate(train_loader):
                
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
    if not incontext:
        os.makedirs(f"./results/analysis/shots/QuestionAnswering/{dataset_name}/{model_name}-{scale}/{shots}-shots", exist_ok=True)
        result_path = f"./results/analysis/shots/QuestionAnswering/{dataset_name}/{model_name}-{scale}/{shots}-shots/{seed}.tsv"
    else:
        os.makedirs(f"./results/analysis/shots/QuestionAnswering/{dataset_name}/{model_name}-{scale}/in-context", exist_ok=True)
        result_path = f"./results/analysis/shots/QuestionAnswering/{dataset_name}/{model_name}-{scale}/in-context/{seed}.tsv"
    eval(prompt_model, processor, tokenizer, ood_list, dataset_name, dataset_path, template, result_path)
    


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=-1)
    parser.add_argument('--model_name', type=str, default="t5")
    parser.add_argument('--dataset_name', type=str, default="squad")
    parser.add_argument('--shots_list', type=int, default=[0], nargs="+")
    parser.add_argument('--scale', type=str, default="base")
    parser.add_argument('--incontext', action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    
    args.model_path = MODEL_PATH[f"{args.model_name}-{args.scale}"]
    args.dataset_path = "./datasets/process/QuestionAnswering"

    template_path = "./prompts/QuestionAnswering/manual_template.txt" if not args.incontext \
                    else "./prompts/QuestionAnswering/incontext_template.txt"
    with open(template_path, "r") as f:
            lines = f.readlines()
            template = "\n".join(lines)

    global dataset
    processor = QAProcessor()
    dataset = processor.get_examples(os.path.join(args.dataset_path, args.dataset_name), "train")   
    print(args.dataset_name)
    print(len(dataset))

    dataset = wrap_template(dataset, template)



    shots_list = args.shots_list
    for shots in shots_list:
        args.shots = shots
        if shots > len(dataset):
            print("Too much samples!")
            print("Exit!")
            exit()

        if shots <= 0:
            repeats = 1 # full data
        else:
            repeats = 5
        
        for i in range(repeats):
            set_seed(i)
            args.seed = i
            main(args)
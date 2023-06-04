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
from src.utils import *
from src.utils.utils_qa import *
from src.utils.dataloader import QAProcessor
from src.evaluations.methods import *

def main(args):
 
    model_name = args.model_name
    model_path = args.model_path
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    scale = args.scale
    method = args.method
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

    ood_model_path = f"./model_cache/QuestionAnswering/{dataset_name}/{model_name}-{scale}/{method}/{seed}"
    ################# load #################    
    if os.path.exists(ood_model_path):
        print("Load plm from cache")
        tokenizer = T5Tokenizer.from_pretrained(ood_model_path)
        prompt_model = T5ForConditionalGeneration.from_pretrained(ood_model_path)
        prompt_model.cuda()
    ################# train #################   
    else:
        print("Train from scratch")
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        prompt_model = T5ForConditionalGeneration.from_pretrained(model_path)
        prompt_model.cuda()
        prompt_model.parallelize()

        train_loader = DataLoader(dataset, shuffle=True, batch_size=16, collate_fn=lambda x: collate_fn(x, tokenizer))

        optimizer = torch.optim.AdamW(prompt_model.parameters(), lr=2e-5)
        
        if method == "focal_loss":
            loss_func = FocalLoss()
        else:
            label_smoothing = 0.1 if method == "label_smoothing" else 0 
            loss_func = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if method == "freelb":
            freelb = FreeLB(base_model=model_name)  

        prompt_model.train()

        for epoch in range(5):
            
            tot_loss = 0
            for step, inputs in enumerate(train_loader):
                input_ids = inputs["input_ids"].cuda()
                attention_mask = inputs["attention_mask"].cuda()
                labels = inputs["label"].cuda()
                if method == "freelb":
                    loss = freelb.attack(prompt_model, inputs)
                else:
                    logits = prompt_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits
                    loss = loss_func(logits.view(-1, logits.size(-1)), labels.view(-1))
                    loss.backward()

                tot_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()

                if step % 100 == 1:
                    print("Epoch {}, average loss: {}".format(epoch+1, tot_loss / (step+1)), flush=True)

        prompt_model.save_pretrained(ood_model_path)
        tokenizer.save_pretrained(ood_model_path)
        print("save model")
        print("finish training")
        
    os.makedirs(f"./results/evaluations/method/QuestionAnswering/{dataset_name}/{model_name}-{scale}/{method}", exist_ok=True)
    result_path = f"./results/evaluations/method/QuestionAnswering/{dataset_name}/{model_name}-{scale}/{method}/{seed}.tsv"
    eval(prompt_model, processor, tokenizer, ood_list, dataset_name, dataset_path, template, result_path)

                
        
        


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--model_name', type=str, default="t5")
    parser.add_argument('--dataset_name', type=str, default="squad")
    parser.add_argument('--scale', type=str, default="base")
    parser.add_argument('--method', type=str, default="vanilla", 
                        choices=["zero-shot", "vanilla", "freelb", "focal_loss", "label_smoothing"])
    args = parser.parse_args()
    device = torch.device("cuda")
    
    args.model_path = MODEL_PATH[f"{args.model_name}-{args.scale}"]
    args.dataset_path = "./datasets/process/QuestionAnswering"


    for i in range(args.repeats):
        set_seed(i)
        args.seed = i
        main(args)
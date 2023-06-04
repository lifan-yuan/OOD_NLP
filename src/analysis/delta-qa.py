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
from opendelta import AutoDeltaModel, AdapterModel, SoftPromptModel

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
    method = args.method
    parameter = args.parameter
    scale = args.scale
    seed = args.seed

    ood_list = OOD_LIST["QuestionAnswering"]
    
    template_path = "./prompts/QuestionAnswering/manual_template.txt"

    
    with open(template_path, "r") as f:
        lines = f.readlines()
        template = "\n".join(lines)

    processor = QAProcessor()
    dataset = processor.get_examples(os.path.join(dataset_path, dataset_name), "train")        
    print(dataset_name)
    print(len(dataset))
    dataset = wrap_template(dataset, template)

    ood_model_path = f"./model_cache/delta/QuestionAnswering/{dataset_name}/{model_name}-{scale}/{method}-{parameter}/{seed}"
    ################# load #################    
    if os.path.exists(ood_model_path):
        print("Load plm from cache")
        tokenizer = T5Tokenizer.from_pretrained(ood_model_path)
        prompt_model = T5ForConditionalGeneration.from_pretrained(ood_model_path)
        delta_model = AutoDeltaModel.from_finetuned(ood_model_path, backbone_model=prompt_model, parameter=parameter)
        delta_model.freeze_module(exclude=["deltas", "layernorm_embedding"], set_state_dict=True)
        prompt_model.cuda()
    ################# train #################   
    else:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        prompt_model = T5ForConditionalGeneration.from_pretrained(model_path)

        #########################################################
        delta_model = AdapterModel(backbone_model=prompt_model, bottleneck_dim=parameter)
        delta_model.freeze_module(exclude=["deltas", "layernorm_embedding"], set_state_dict=True)
        delta_model.log()
        prompt_model.cuda()    
        if scale == "large":
            prompt_model.parallelize()
        #########################################################


        train_loader = DataLoader(dataset, shuffle=True, batch_size=16, collate_fn=lambda x: collate_fn(x, tokenizer))
        
        optimizer = torch.optim.AdamW(prompt_model.parameters(), lr=2e-5)

        loss_func = nn.CrossEntropyLoss()

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
        delta_model.save_finetuned(ood_model_path)
        print("save model")
        print("finish training")

    os.makedirs(f"./results/analysis/delta/QuestionAnswering/{dataset_name}/{model_name}-{scale}/{method}/{parameter}-parameter", exist_ok=True)
    result_path = f"./results/analysis/delta/QuestionAnswering/{dataset_name}/{model_name}-{scale}/{method}/{parameter}-parameter/{seed}.tsv"
    eval(prompt_model, processor, tokenizer, ood_list, dataset_name, dataset_path, template, result_path)

                
        
        


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--model_name', type=str, default="t5")
    parser.add_argument('--dataset_name', type=str, default="squad")
    parser.add_argument('--method', type=str, default="soft")
    parser.add_argument('--parameter', type=int, default=1)
    parser.add_argument('--scale', type=str, default="base")
    parser.add_argument
    args = parser.parse_args()
    device = torch.device("cuda")
    
    args.model_path = MODEL_PATH[f"{args.model_name}-{args.scale}"]
    args.dataset_path = "./datasets/process/QuestionAnswering"

    for i in range(args.repeats):
        set_seed(i)
        args.seed = i
        main(args)
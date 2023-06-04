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


from opendelta import AutoDeltaModel, AdapterModel


def main(args):
    num_classes = args.num_classes
    model_name = args.model_name
    model_path = args.model_path
    task_name = args.task_name
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    method = args.method
    parameter = args.parameter
    scale = args.scale
    seed = args.seed


    ood_list = OOD_LIST[task_name]
    
    dataset = {}

    processor = PROCESSOR[task_name]()
    dataset['train'] = processor.get_examples(os.path.join(dataset_path, dataset_name), "train")
    print(dataset_name)
    print(len(dataset['train']))

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    ood_model_path = f"./model_cache/analysis/delta/{task_name}/{dataset_name}/{model_name}-{scale}/{method}-{parameter}/{seed}"
    ################# load #################
    if os.path.exists(ood_model_path):
        print("Load plm from cache")
        model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=num_classes).cuda()
        model.classifier.load_state_dict(torch.load(os.path.join(ood_model_path, "classifier.pt")))
        delta_model = AutoDeltaModel.from_finetuned(ood_model_path, backbone_model=model.base_model, parameter=parameter)
        delta_model.freeze_module(module=model.base_model, exclude=["deltas", "layernorm_embedding"], set_state_dict=True)
    ################# train #################
    else:
        print("Train from scratch")

        model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=num_classes).cuda()
        delta_model = AdapterModel(backbone_model=model.base_model, bottleneck_dim=parameter)
        delta_model.freeze_module(module=model.base_model, exclude=["deltas", "layernorm_embedding"], set_state_dict=True)
        delta_model.log()
        
        dataset['train'] = dataset['train'].map(tokenize_and_align_labels, fn_kwargs={"tokenizer":tokenizer}, batched=True).remove_columns(["tokens", "tags", "tag_ids"])
        train_dataloader = DataLoader(dataset['train'], shuffle=True, batch_size=16, collate_fn=data_collator)

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
        delta_model.save_finetuned(ood_model_path)
        torch.save(model.classifier.state_dict(), os.path.join(ood_model_path, "classifier.pt"))

        


    os.makedirs(f"./results/analysis/delta/{task_name}/{dataset_name}/{model_name}-{scale}/{method}/{parameter}-parameter", exist_ok=True)
    result_path = f"./results/analysis/delta/{task_name}/{dataset_name}/{model_name}-{scale}/{method}/{parameter}-parameter/{seed}.tsv"
    if method != "soft":
        parameter = -1
    eval(model, processor, dataset_path, tokenizer, result_path, task_name, ood_list, dataset_name, model_name, parameter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--model_name', type=str, default="deberta")
    parser.add_argument('--dataset_name', type=str, default="fewnerd")
    parser.add_argument('--method', type=str, default="adapter")
    parser.add_argument('--parameter', type=int, default=1)
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
    

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
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import *
from src.utils.utils_ner import *
from src.utils.dataloader import *
from src.evaluations.methods import *



def main(args):
    num_classes = args.num_classes
    model_name = args.model_name
    model_path = args.model_path
    task_name = args.task_name
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    scale = args.scale
    method = args.method
    seed = args.seed
    

    ood_list = OOD_LIST[task_name]
    
    dataset = {}

    processor = PROCESSOR[task_name]()
    dataset['train'] = processor.get_examples(os.path.join(dataset_path, dataset_name), "train")
    print(dataset_name)
    print(len(dataset['train']))

    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)


    ood_model_path = f"./model_cache/{task_name}/{dataset_name}/{model_name}-{scale}/{method}/{seed}"
    ################# load #################
    if os.path.exists(os.path.join(ood_model_path, "pytorch_model.bin")):
        print("Load plm from cache")
        model = AutoModelForTokenClassification.from_pretrained(ood_model_path, num_labels=num_classes).cuda()
    ################# train #################
    else:
        print("Train from scratch")
        
        print(model_path, num_classes)
        model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=num_classes).cuda()

        train_dataset = dataset['train'].map(tokenize_and_align_labels, fn_kwargs={"tokenizer":tokenizer}, batched=True).remove_columns(["tokens", "tags", "tag_ids"])
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=data_collator)

        if method == "focal_loss":
            loss_func = FocalLoss()
        else:
            label_smoothing = 0.1 if method == "label_smoothing" else 0 
            loss_func = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if method == "freelb":
            freelb = FreeLB(base_model=model_name)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        model.train()
        train_length = len(train_dataloader)
        for epoch in range(10):
            tot_loss = 0
            for step, inputs in enumerate(train_dataloader):

                inputs = {k: v.cuda() for k, v in inputs.items()}

                if method == "freelb":
                    loss = freelb.attack(model, inputs)
                else:
                    input_ids = inputs["input_ids"].cuda()
                    attention_masks = inputs["attention_mask"].cuda()
                    labels = inputs["label"].cuda()
                    logits = model(input_ids=input_ids, attention_mask=attention_masks).logits
                    loss = loss_func(logits.view(-1, num_classes), labels.view(-1))
                    loss.backward()
                
                tot_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()

                if step % 100 == 1:
                    print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)

        print("finish training")
        try:
            model.save_pretrained(ood_model_path)
        except:
            model.model.save_pretrained(ood_model_path)
            model = model.model

    os.makedirs(f"./results/evaluations/method/{task_name}/{dataset_name}/{model_name}-{scale}/{method}", exist_ok=True)
    result_path = f"./results/evaluations/method/{task_name}/{dataset_name}/{model_name}-{scale}/{method}/{seed}.tsv"
    eval(model, processor, dataset_path, tokenizer, result_path, task_name, ood_list, dataset_name, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--model_name', type=str, default="deberta")
    parser.add_argument('--dataset_name', type=str, default="fewnerd")
    parser.add_argument('--scale', type=str, default="base")
    parser.add_argument('--method', type=str, default="vanilla", 
                        choices=["zero-shot", "vanilla", "freelb", "focal_loss", "label_smoothing"])
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
    

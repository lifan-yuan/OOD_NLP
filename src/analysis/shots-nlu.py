import argparse
import math
import os
import pandas as pd
from tqdm import tqdm
import torch
from openprompt.data_utils.utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer
from openprompt import PromptForClassification
from openprompt.data_utils.data_sampler import FewShotSampler
from transformers import get_linear_schedule_with_warmup
from openprompt import PromptDataLoader
from transformers import T5ForConditionalGeneration, AutoModelForSequenceClassification
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.dataloader import *
from src.utils import *
from src.utils.utils_nlu import *




def main(args):
    num_classes = args.num_classes
    model_name = args.model_name
    model_path = args.model_path
    task_name = args.task_name
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    shots = args.shots
    scale = args.scale
    incontext = args.incontext
    seed = args.seed

    print(f"{shots=}")

    ood_list = OOD_LIST[task_name]
    

    sampled_dataset = deepcopy(dataset)
    if shots != -1 and shots != 0: # requires sampling
        sampled_dataset["train"] = sampling(sampled_dataset["train"], shots)

    if model_name == "t0":
        plm, tokenizer, model_config, WrapperClass = load_plm("t5", model_path)
    else:
        plm, tokenizer, model_config, WrapperClass = load_plm(model_name.split("-")[0], model_path)
    
    template_path = f"./prompts/{task_name}/manual_template.txt" if not incontext \
                    else f"./prompts/{task_name}/incontext_template.txt"

    mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(template_path, choice=0)
    myverbalizer = ManualVerbalizer(tokenizer, num_classes=num_classes).from_file(f"./prompts/{task_name}/manual_verbalizer.txt")
    prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False).cuda()
    # prompt_model.parallelize()
    if shots != 0:
        print("Train from scratch")
        
        train_dataloader = PromptDataLoader(dataset=sampled_dataset["train"], template=mytemplate, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=768, decoder_max_length=3,
                batch_size=16, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                truncate_method="tail")
        
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-5)
        
        loss_func = nn.CrossEntropyLoss()

        prompt_model.train()

        for epoch in range(10):
            tot_loss = 0
            for step, inputs in enumerate(train_dataloader):
                inputs = inputs.cuda()
                logits = prompt_model(inputs)
                labels = inputs['label']
                loss = loss_func(logits, labels)
                loss.backward()
                tot_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                if step % 100 == 1:
                    print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)
    
    if not incontext:
        os.makedirs(f"./results/analysis/shots/{task_name}/{dataset_name}/{model_name}-{scale}/{shots}-shots", exist_ok=True)
        result_path = f"./results/analysis/shots/{task_name}/{dataset_name}/{model_name}-{scale}/{shots}-shots/{seed}.tsv"
    else:
        os.makedirs(f"./results/analysis/shots/{task_name}/{dataset_name}/{model_name}-{scale}/in-context", exist_ok=True)
        result_path = f"./results/analysis/shots/{task_name}/{dataset_name}/{model_name}-{scale}/in-context/{seed}.tsv"
    eval(prompt_model, processor, dataset_path, mytemplate, tokenizer, WrapperClass, result_path, 
            task_name, ood_list, dataset_name, model_name)
    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=-1)
    parser.add_argument('--model_name', type=str, default="t5")
    parser.add_argument('--dataset_name', type=str, default="amazon")
    parser.add_argument('--shots_list', type=int, default=[0], nargs="+")
    parser.add_argument('--scale', type=str, default="base")
    parser.add_argument('--incontext', action="store_true")
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

    # shots_list = args.shots_list
    # if args.repeats == 0:
    #     shots_list = [0]
    #     shots_list.extend([2**i for i in range(25)])
    # for shots in shots_list:
    #     args.shots = shots
    #     if shots * args.num_classes > len(dataset["train"]):
    #         print("Too much samples!")
    #         print("Exit!")
    #         exit()

    #     if shots <= 0:
    #         repeats = 1 # full data
    #     else:
    #         repeats = 5
        
    #     for i in range(repeats):
    #         set_seed(i*1000)
    #         args.seed = i
    #         main(args)
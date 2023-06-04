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
    scale = args.scale
    seed = args.seed
    
    ood_list = OOD_LIST[task_name]
    
    dataset = {}

    processor = PROCESSOR[task_name]()
    dataset['train'] = processor.get_examples(os.path.join(dataset_path, dataset_name), "train")
    print(dataset_name)
    print(len(dataset['train']))


    print("Train from scratch")
    plm, tokenizer, model_config, WrapperClass = load_plm(model_name.split("-")[0], model_path)
    mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(f"./prompts/{task_name}/manual_template.txt", choice=0)
    myverbalizer = ManualVerbalizer(tokenizer, num_classes=num_classes).from_file(f"./prompts/{task_name}/manual_verbalizer.txt")
    prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False).cuda()
    prompt_model.parallelize()

    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=768, decoder_max_length=3,
            batch_size=16, shuffle=True, teacher_forcing=False, predict_eos_token=False,
            truncate_method="tail")
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    
    loss_func = nn.CrossEntropyLoss()

    prompt_model.train()
    train_length = len(train_dataloader)

    steps_dict = {}
    node_list = [10**i for i in range(-10, 0)]
    node_list.extend([i/10 for i in range(1, 10)])
    for node in node_list:
        if int(train_length * node) > 0:
            steps_dict[int(train_length * node)] = node

    for epoch in range(10):
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):

            accumulated_step = epoch*train_length + step

            if (epoch == 0 and step in steps_dict.keys()) or (step == 0):
                node = epoch if step == 0 else steps_dict[step]
                print(node, "epochs")
                os.makedirs(f"./results/analysis/steps/{task_name}/{dataset_name}/{model_name}-{scale}/{node}-epochs", exist_ok=True)
                result_path = f"./results/analysis/steps/{task_name}/{dataset_name}/{model_name}-{scale}/{node}-epochs/{seed}.tsv"
                eval(prompt_model, processor, dataset_path, mytemplate, tokenizer, WrapperClass, result_path, 
                        task_name, ood_list, dataset_name, model_name)


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


    epoch += 1
    os.makedirs(f"./results/analysis/steps/{task_name}/{dataset_name}/{model_name}-{scale}/{epoch}-epochs", exist_ok=True)
    result_path = f"./results/analysis/steps/{task_name}/{dataset_name}/{model_name}-{scale}/{epoch}-epochs/{seed}.tsv"
    eval(prompt_model, processor, dataset_path, mytemplate, tokenizer, WrapperClass, result_path, 
            task_name, ood_list, dataset_name, model_name)
    print("finish all!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--model_name', type=str, default="t5")
    parser.add_argument('--dataset_name', type=str, default="hsol")
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
    

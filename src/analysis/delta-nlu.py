import argparse
import math
import os
import pandas as pd
from tqdm import tqdm
import torch
from openprompt.data_utils.utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, SoftTemplate
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

from opendelta import AutoDeltaModel, AdapterModel




def load_delta(plm, method, parameter):
    if method == "adapter":
        delta_model = AdapterModel(backbone_model=plm, bottleneck_dim=parameter)
    elif method == "soft":
        delta_model = SoftPromptModel(backbone_model=plm, soft_token_num=parameter)
    else:
        delta_model = None
    return plm, delta_model



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
    dataset['test'] = processor.get_examples(os.path.join(dataset_path, dataset_name), "test")  
    print(dataset_name)
    print(len(dataset['train']))

    ood_model_path = f"./model_cache/delta/{task_name}/{dataset_name}/{model_name}-{scale}/{method}-{parameter}/{seed}"
    print(ood_model_path)
    ################# load #################
    if os.path.exists(ood_model_path):
        print("Load plm from cache")
        plm, tokenizer, model_config, WrapperClass = load_plm(model_name.split("-")[0], model_path)
        delta_model = AutoDeltaModel.from_finetuned(ood_model_path, backbone_model=plm, parameter=parameter)
        delta_model.freeze_module(exclude=["deltas", "layernorm_embedding"], set_state_dict=True)
        mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(f"./prompts/{task_name}/manual_template.txt", choice=0)
        myverbalizer = ManualVerbalizer(tokenizer, num_classes=num_classes).from_file(f"./prompts/{task_name}/manual_verbalizer.txt")
        prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False).cuda()
    ################# train #################
    else:
        print("Train from scratch")
        plm, tokenizer, model_config, WrapperClass = load_plm(model_name, model_path)
        delta_model = AdapterModel(backbone_model=plm, bottleneck_dim=parameter)
        mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(f"./prompts/{task_name}/manual_template.txt", choice=0)
        myverbalizer = ManualVerbalizer(tokenizer, num_classes=num_classes).from_file(f"./prompts/{task_name}/manual_verbalizer.txt")

        delta_model.freeze_module(exclude=["deltas", "layernorm_embedding"], set_state_dict=True)
        delta_model.log()

        prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False).cuda()
        if scale == "large":
            prompt_model.parallelize()

        train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=768, decoder_max_length=3,
            batch_size=16,shuffle=True, teacher_forcing=False, predict_eos_token=False,
            truncate_method="tail")
        
        loss_func = torch.nn.CrossEntropyLoss()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5)
        
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
                if step %100 == 1:
                    print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)
        delta_model.save_finetuned(ood_model_path)
        print("save model")
        print("finish training")



    os.makedirs(f"./results/analysis/delta/{task_name}/{dataset_name}/{model_name}-{scale}/{method}/{parameter}-parameter", exist_ok=True)
    result_path = f"./results/analysis/delta/{task_name}/{dataset_name}/{model_name}-{scale}/{method}/{parameter}-parameter/{seed}.tsv"
    eval(prompt_model, processor, dataset_path, mytemplate, tokenizer, WrapperClass, result_path, 
            task_name, ood_list, dataset_name, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--model_name', type=str, default="t5")
    parser.add_argument('--dataset_name', type=str, default="amazon")
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
    

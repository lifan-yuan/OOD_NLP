import argparse
import math
import pandas as pd
from tqdm import tqdm
import torch
from openprompt.data_utils.utils import InputExample, InputFeatures
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptForClassification
from openprompt.data_utils.data_sampler import FewShotSampler
from transformers import get_linear_schedule_with_warmup
from openprompt import PromptDataLoader
from transformers import T5ForConditionalGeneration, AutoModelForSequenceClassification
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import *
from src.utils.utils_nlu import *
from src.utils.dataloader import *
from src.evaluations.methods import *


OOD_LIST = {
        "SentimentAnalysis": ["amazon", "dsc", "dynasent", "imdb","semeval", "sst5", "yelp"],
        "ToxicDetection": ["abuse_analyzer", "adv_civil","civil_comments", "ethos", "hate_speech", "hsol","implicit_hate", "olid", "toxigen"],
        "NaturalLanguageInference": ["anli", "bio_nli", "cb", "contract_nli", "doc_nli", "mnli", "qnli", "snli", "wanli"],
        "NameEntityRecognition": ["conll", "crossner", "fewnerd", "wnut"],
}


def main(args):
    num_classes = args.num_classes
    model_name = args.model_name
    model_path = args.model_path
    task_name = args.task_name
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    method = args.method
    scale = args.scale
    seed = args.seed
    set_seed(seed)

    ood_list = OOD_LIST[task_name]
    
    dataset = {}

    processor = PROCESSOR[task_name]()
    dataset['train'] = processor.get_examples(os.path.join(dataset_path, dataset_name), "train")
    print(dataset_name)
    print(len(dataset['train']))

    
    


    ood_model_path = f"./model_cache/{task_name}/{dataset_name}/{model_name}-{scale}/{method}/{seed}"
    ################# load #################
    if os.path.exists(ood_model_path):
        print("Load plm from cache")
        plm, tokenizer, model_config, WrapperClass = load_plm(model_name.split("-")[0], ood_model_path)
        mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(f"./prompts/{task_name}/manual_template.txt", choice=0)
        myverbalizer = ManualVerbalizer(tokenizer, num_classes=num_classes).from_file(f"./prompts/{task_name}/manual_verbalizer.txt")
        prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False).cuda()
    ################# train #################
    else:
        print("Train from scratch")
        plm, tokenizer, model_config, WrapperClass = load_plm(model_name.split("-")[0], model_path)
        mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(f"./prompts/{task_name}/manual_template.txt", choice=0)
        myverbalizer = ManualVerbalizer(tokenizer, num_classes=num_classes).from_file(f"./prompts/{task_name}/manual_verbalizer.txt")
        prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False).cuda()
        prompt_model.parallelize()
        train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                batch_size=16, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                truncate_method="tail")
        

        if method == "eda":
            dataset["train"] = eda_augmentation(dataset["train"])

        # loss function
        if method == "shallow_ensemble":
            loss_func = LearnedMixin(penalty=0.03, dim=model_config.hidden_size)
            prompt_model = ShallowEnsembledModel(prompt_model, loss_func).cuda()
            prompt_model.parallelize()
            # capture bias
            if not os.path.exists(f"./results/evaluations/method/{task_name}/{dataset_name}/{model_name}-{scale}/shallow_ensemble/bias.pt"):
                shallow_model = ShallowModelForNLU()
                biases = shallow_model.capture_bias(dataset["train"])
                print("save bias!")
                os.makedirs(f"./results/evaluations/method/{task_name}/{dataset_name}/{model_name}-{scale}/shallow_ensemble", exist_ok=True)
                torch.save(biases, f"./results/evaluations/method/{task_name}/{dataset_name}/{model_name}-{scale}/shallow_ensemble/bias.pt")
            else:
                print("load bias!")
                biases = torch.load(f"./results/evaluations/method/{task_name}/{dataset_name}/{model_name}-{scale}/shallow_ensemble/bias.pt")
            # prepare dataloader
            train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                batch_size=1, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                truncate_method="tail")
            
            train_dataset = []
            import logging
            logging.disable(logging.WARNING)
            for input, bias in zip(train_dataloader, biases):
                train_dataset.append(InputFeatures( input_ids=input["input_ids"][0], 
                                        attention_mask=input["attention_mask"][0],
                                        decoder_input_ids=input["decoder_input_ids"][0],
                                        label=input["label"][0],
                                        loss_ids=input["loss_ids"][0],
                                        bias=torch.tensor(bias)))
            InputFeatures.add_keys("bias")
            
            from torch.utils.data import DataLoader
            train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=InputFeatures.collate_fct)
        elif method == "freelb":
            freelb = FreeLB(base_model=model_name)  
        elif method == "focal_loss": # gamma: 2:90/43/51/74, 10:90/43/50/74, 1:91/44/50/75
            loss_func = FocalLoss()
        else:
            label_smoothing = 0.1 if method == "label_smoothing" else 0 # 0.1: 90/44/50/74, 0.2:90/44/50/74
            loss_func = nn.CrossEntropyLoss(label_smoothing=label_smoothing)


        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-5)

        prompt_model.train()
        
        for epoch in range(10):
            tot_loss = 0
            for step, inputs in enumerate(train_dataloader):

                inputs = inputs.cuda()

                
                if method == "shallow_ensemble":
                    labels = inputs["label"].cuda()
                    bias = inputs["bias"].cuda()
                    logits, loss = prompt_model(inputs, labels, bias)
                    loss.backward()
                elif method == "freelb":
                    loss = freelb.attack(prompt_model, inputs)
                else:
                    logits = prompt_model(inputs)
                    labels = inputs['label']
                    loss = loss_func(logits, labels)
                    loss.backward()

                tot_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                if step % 100 == 1:
                    print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)
        
        try:
            prompt_model.plm.save_pretrained(ood_model_path)
        except:
            prompt_model.model.plm.save_pretrained(ood_model_path)
            prompt_model = prompt_model.model
            del InputFeatures.all_keys[-1]
        tokenizer.save_pretrained(ood_model_path)
        model_config.save_pretrained(ood_model_path)

        print("save model")
        print("finish training")

    os.makedirs(f"./results/evaluations/method/{task_name}/{dataset_name}/{model_name}-{scale}/{method}", exist_ok=True)
    result_path = f"./results/evaluations/method/{task_name}/{dataset_name}/{model_name}-{scale}/{method}/{seed}.tsv"
    eval(prompt_model, processor, dataset_path, mytemplate, tokenizer, WrapperClass, result_path, 
            task_name, ood_list, dataset_name, model_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--model_name', type=str, default="t5")
    parser.add_argument('--scale', type=str, default="large")
    parser.add_argument('--dataset_name', type=str, default="hsol")
    parser.add_argument('--method', type=str, default="vanilla", 
                        choices=["zero-shot", "vanilla", "freelb", "focal_loss", "fl", "label_smoothing", "ls", 
                                "shallow_ensemble", "se"])
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
    

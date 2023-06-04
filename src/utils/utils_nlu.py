import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from .dataloader import *
from openprompt import PromptDataLoader
from tqdm import tqdm




import random
import numpy as np
import torch


def sampling(dataset, shots):
    subsets = {}
    sampled_dataset = []
    for data in dataset:
        if data.label not in subsets.keys():
            subsets[data.label] = []
        subsets[data.label].append(data)

    for label, subset in subsets.items():
        random.shuffle(subset)
        shots = min(shots, len(subset))
        sampled_dataset.extend(subset[:shots])
    
    random.shuffle(sampled_dataset)

    return sampled_dataset
    
def evaluation(test_dataloader, prompt_model, task_name, dataset_name, model_name, ood_name):

    prompt_model.eval()

    allprobs = []
    allpreds = []
    alllabels = []

    with torch.no_grad():
        for step, inputs in enumerate(test_dataloader):
            inputs = inputs.cuda()
            logits = prompt_model(inputs)
            probs = F.softmax(logits, dim=-1)

            if (task_name == "SentimentAnalysis" and ood_name in ["dsc", "imdb"]):
                    probs = probs[:, :2]
            if (task_name == "NaturalLanguageInference" and ood_name in ["bio_nli", "doc_nli", "qnli"]):
                    probs = torch.stack([probs[:,0], probs[:,1]+probs[:,2]], dim=1)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allprobs.extend([prob.max().item() for prob in probs])
            allpreds.extend(torch.argmax(probs, dim=-1).cpu().tolist())

    prompt_model.train()

    acc = 100 * sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    print('acc on {}: {}'.format(ood_name, acc))
    
    return acc




def eval(prompt_model, processor, dataset_path, mytemplate, tokenizer, WrapperClass, result_path, 
            task_name, ood_list, dataset_name, model_name):
    print("evaluation")
    
    dataset = {}
    for ood_name in ood_list:
        dataset[ood_name] = processor.get_examples(os.path.join(dataset_path, ood_name), "test")   
      
            
    dataloader_dict = {}
    for ood_name in dataset.keys(): # including the test split
        if os.path.exists(f"./datasets/tokenize/{task_name}/{ood_name}.pt"):
            print(f"load tokenized test dataset of {ood_name}")
            test_dataloader = torch.load(f"./datasets/tokenize/{task_name}/{ood_name}.pt")
        else:
            test_dataloader = PromptDataLoader(dataset=dataset[ood_name], template=mytemplate, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=768, decoder_max_length=3,
                batch_size=2,shuffle=False, teacher_forcing=False, predict_eos_token=False,
                truncate_method="head")
            batch_list = []
            for batch in test_dataloader:
                batch_list.append(batch)
            os.makedirs(f"./datasets/tokenize/{task_name}", exist_ok=True)
            print(f"save tokenized test dataset of {ood_name}")
            torch.save(batch_list, f"./datasets/tokenize/{task_name}/{ood_name}.pt")
        dataloader_dict[ood_name] = test_dataloader

    # evaluate performance
    print("Performance:")
    
    names = ["Dataset"]
    accuracies = ["Acc"]
    for ood_name, test_dataloader in dataloader_dict.items():
        acc = evaluation(test_dataloader, prompt_model, task_name, dataset_name, model_name, ood_name)
        names.append(ood_name)
        accuracies.append(acc)

    results = pd.DataFrame([accuracies], columns=names)
    results.to_csv(result_path, sep="\t", index=False, header=names)
    print("finish evaluation")


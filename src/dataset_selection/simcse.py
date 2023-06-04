import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

import json, csv
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from tqdm import tqdm

from src.utils import *
from src.utils.dataloader import *

simcse_embeddings = {}
centroids = {}

# sup_ckpt = "princeton-nlp/sup-simcse-roberta-large"
# unsup_ckpt = "princeton-nlp/unsup-simcse-roberta-large"
sup_ckpt = "/data/private/yuanlifan/.cache/transformers/models--princeton-nlp--sup-simcse-roberta-large/snapshots/96d164d9950b72f4ce179cb1eb3414de0910953f"
unsup_ckpt = "/data/private/yuanlifan/.cache/transformers/models--princeton-nlp--unsup-simcse-roberta-large/snapshots/d3f863b476c59b0673264042f159cea15842e265"
tokenizer_unsup = AutoTokenizer.from_pretrained(unsup_ckpt)
model_unsup = AutoModel.from_pretrained(unsup_ckpt).cuda()

tokenizer_sup = AutoTokenizer.from_pretrained(sup_ckpt)
model_sup = AutoModel.from_pretrained(sup_ckpt).cuda()

# compute embeddings and centroids
for dataset_name, task_name in TASK.items():
    if os.path.exists(f"./results/dataset_selection/embeddings/{task_name}/{dataset_name}.npy"):
        embeddings = np.load(f"./results/dataset_selection/embeddings/{task_name}/{dataset_name}.npy")
    else:
        dataset_path = DATASET_PATH[task_name]
        processor = PROCESSOR[task_name]()
        dataset = processor.get_examples(os.path.join(dataset_path, dataset_name), "test")
        print(dataset[0])
        texts = []
        if task_name in ["SentimentAnalysis", "ToxicDetection", "NaturalLanguageInference", "CommonSense"]:
            for data in dataset:
                texts.append(data.text_a)
        elif task_name == "NameEntityRecognition":
            for data in dataset:
                texts.append(data[0])
        elif task_name == "QuestionAnswering":
            for data in dataset:
                texts.append(data["context"])
        texts = list(set(texts))

        data_loader = DataLoader(texts, shuffle=False, batch_size=16)

        embeddings = []

        if task_name == "NaturalLanguageInference":
            tokenizer = tokenizer_unsup
            model = model_unsup
        else:
            tokenizer = tokenizer_sup
            model = model_sup

        for batch in tqdm(data_loader, desc=dataset_name):
            # Tokenize input texts
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()
            # Get the embeddings
            with torch.no_grad():
                embedding = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.detach().cpu().numpy()
                embeddings.extend(embedding)

        
        embeddings = np.array(embeddings)
        os.makedirs(f"./results/dataset_selection/embeddings/{task_name}", exist_ok=True)
        np.save(f"./results/dataset_selection/embeddings/{task_name}/{dataset_name}.npy", embeddings)
    
    simcse_embeddings[dataset_name] = embeddings
    centroids[dataset_name] = np.mean(embeddings, axis=0)

# compute cosine distance
cosine_similarity = {}
names = {}
for task_name in PROCESSOR.keys():
    cosine_similarity[task_name] = []
    names[task_name] = ["Dataset"]

for dataset_name, task_name in TASK.items():
    names[task_name].append(dataset_name)
    cos_sim = [dataset_name]
    for ood_name, ood_task in TASK.items():
        if task_name != ood_task:
            continue
        # Cosine similarities are in [-1, 1]. Higher means more similar     
        cos_sim.append(np.round((1 - cosine(centroids[dataset_name], centroids[ood_name]))*100, 2))
    cosine_similarity[task_name].append(cos_sim)

for task_name, simcse in cosine_similarity.items():
    results = pd.DataFrame(simcse, columns=names[task_name])
    os.makedirs(f"./results/dataset_selection/simcse", exist_ok=True)
    results.to_csv(f"./results/dataset_selection/simcse/{task_name}.tsv", sep="\t", index=False)

    
    





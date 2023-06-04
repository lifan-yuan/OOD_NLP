from .dataloader import *

TASK = {
    "amazon": "SentimentAnalysis",
    "dsc": "SentimentAnalysis",
    "dynasent": "SentimentAnalysis",
    "imdb": "SentimentAnalysis",
    "semeval": "SentimentAnalysis",
    "sst5": "SentimentAnalysis",
    "yelp": "SentimentAnalysis",

    "abuse_analyzer": "ToxicDetection",
    "adv_civil": "ToxicDetection",
    "civil_comments": "ToxicDetection",
    "hate_speech": "ToxicDetection",
    "hsol": "ToxicDetection",
    "implicit_hate": "ToxicDetection",
    "olid": "ToxicDetection",
    "toxigen": "ToxicDetection",

    "anli": "NaturalLanguageInference",
    "bio_nli": "NaturalLanguageInference",
    "cb": "NaturalLanguageInference",
    "contract_nli": "NaturalLanguageInference",
    "doc_nli": "NaturalLanguageInference",
    "mnli": "NaturalLanguageInference",
    "snli": "NaturalLanguageInference",
    "wanli": "NaturalLanguageInference",

    "conll": "NameEntityRecognition",
    "crossner": "NameEntityRecognition",
    "ener": "NameEntityRecognition",
    "fewnerd": "NameEntityRecognition",
    "wnut": "NameEntityRecognition",

    "advqa": "QuestionAnswering",
    "hotpotqa": "QuestionAnswering",
    "naturalquestions": "QuestionAnswering",
    "newsqa": "QuestionAnswering",
    "searchqa": "QuestionAnswering",
    "squad": "QuestionAnswering",
    "squadshifts": "QuestionAnswering",
    "triviaqa": "QuestionAnswering",

}

PROCESSOR = {
    "SentimentAnalysis": SentimentAnalysisProcessor,
    "ToxicDetection": ToxicDetectionProcessor,
    "NaturalLanguageInference": NLIProcessor,
    "NameEntityRecognition": NERProcessor,
    "QuestionAnswering": QAProcessor,
}

DATASET_PATH = {
    "SentimentAnalysis": "./datasets/process/SentimentAnalysis",
    "ToxicDetection": "./datasets/process/ToxicDetection",
    "NaturalLanguageInference": "./datasets/process/NaturalLanguageInference",
    "NameEntityRecognition": "./datasets/process/NameEntityRecognition",
    "QuestionAnswering": "./datasets/process/QuestionAnswering",
}

DATASET_LIST = {
    "SentimentAnalysis": ["amazon", "dynasent", "semeval", "sst5"],
    "ToxicDetection": ["adv_civil","civil_comments", "implicit_hate", "toxigen"],
    "NaturalLanguageInference": ["anli", "contract_nli", "mnli", "wanli"],
    "NameEntityRecognition": ["conll", "ener", "fewnerd", "wnut"],
    "QuestionAnswering": ["advqa", "newsqa", "searchqa", "squad"]
}

OOD_LIST = {
    "SentimentAnalysis": ["amazon", "dynasent", "semeval", "sst5"],
    "ToxicDetection": ["adv_civil","civil_comments", "implicit_hate", "toxigen"],
    "NaturalLanguageInference": ["anli", "contract_nli", "mnli", "wanli"],
    "NameEntityRecognition": ["conll", "ener", "fewnerd", "wnut"],
    "QuestionAnswering": ["advqa", "newsqa", "searchqa", "squad"]
}

VERBALIZER = {
    "SentimentAnalysis": {
        "amazon": ["negative", "positive", "neutral"],
        "dynasent": ["negative", "positive", "neutral"],
        "semeval": ["negative", "positive", "neutral"],
        "sst5": ["negative", "positive", "neutral"],
    },
    "ToxicDetection": {
        "adv_civil": ["benign", "toxic"],
        "civil_comments": ["benign", "toxic"],
        "implicit_hate": ["benign", "toxic"],
        "toxigen": ["benign", "toxic"],
    },
    "NaturalLanguageInference": {
        "anli": ["entailment", "neutral", "contradiction"],
        "bio_nli": ["yes", "no"],
        "cb": ["entailment", "neutral", "contradiction"],
        "contract_nli": ["entailment", "neutral", "contradiction"],
        "doc_nli": ["yes", "no"],
        "mnli": ["entailment", "neutral", "contradiction"],
        "wanli": ["entailment", "neutral", "contradiction"],
    },
}

MAX_TOKENS = {
    "SentimentAnalysis": 1,
    "ToxicDetection": 1,
    "NaturalLanguageInference": 1,
    "NameEntityRecognition": 50,
    "QuestionAnswering": 5
}

MODEL_PATH = {
    "t5": "./model_cache/t5-base",
    "t5-small": "./model_cache/t5-small",
    "t5-base": "./model_cache/t5-base",
    "t5-large": "./model_cache/t5-large",
    "t5-3b": "./model_cache/t5-3b",
    "t0-3b": "./model_cache/t0-3b",

    "bert": "bert-base-cased",
    "deberta": "microsoft/deberta-v3-base",
    "deberta-small": "microsoft/deberta-v3-small",
    "deberta-base": "microsoft/deberta-v3-base",
    "deberta-large": "microsoft/deberta-v3-large",
    
}

NUM_CLASSES = {
    "SentimentAnalysis": 3,
    "ToxicDetection": 2,
    "NaturalLanguageInference": 3,
    "NameEntityRecognition": 17,
}

import random
import numpy as np
import torch
def set_seed(seed):
    print("set seed:", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
import openai

openai.api_key = open("api.key").readlines()[0].strip()

import argparse

import os
from tqdm import tqdm
from time import sleep
import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils_llm import *


def main(wrapped_dataset, labels, dataset_name):

    if os.path.exists(os.path.join("llm_temp_output", task, dataset_name, model_name, f"{setting}.tsv")):
        df = pd.read_csv(os.path.join("llm_temp_output", task, dataset_name, model_name, f"{setting}.tsv"), sep='\t')
        prediction_list = df["prediction"].astype(str).tolist()
        if task in ["SentimentAnalysis", "ToxicDetection", "NaturalLanguageInference"]:
          reference_list = df["reference"].tolist()
        elif task == "NameEntityRecognition":
          reference_list = [eval(ref) for ref in df["reference"].astype(str).tolist()]
        elif task == "QuestionAnswering":
          reference_list = [eval(ref) for ref in df["reference"].astype(str).tolist()]
        else:
          print(task, "error!")
    else:
        prediction_list = []
        reference_list = []

    total = len(wrapped_dataset)
    if len(prediction_list) < total:
        for i, (input, label) in tqdm(enumerate(zip(wrapped_dataset, labels)), total=total):

            if i % 1000 == 0:
                print(dataset_name, "Evaluating {}/{}".format(i, total))
                # save outputs in case of being interupted
                if i > 0:
                    df = pd.DataFrame({"prediction": prediction_list, "reference": reference_list})
                    df.to_csv(os.path.join("llm_temp_output", task, dataset_name, model_name, f"{setting}.tsv"), sep="\t", index=None)

            if i < len(prediction_list): # start from the outputs saved in last run
              continue
            # query with try-catch
            while True:
                try:
                    result = evaluation(input, max_tokens, model_name, 
                                        generator if model_name == "llama" else None).strip("\n")
                except openai.error.RateLimitError as e:
                    # print(e)
                    if str(e).startswith("Rate limit reached"):
                        sleep(1)
                        continue
                else:
                    break
            prediction_list.append(result)
            reference_list.append(label)

        # save outputs
        df = pd.DataFrame({"prediction": prediction_list, "reference": reference_list})
        df.to_csv(os.path.join("llm_temp_output", task, dataset_name, model_name, f"{setting}.tsv"), sep="\t", index=None)

    if task in ["SentimentAnalysis", "ToxicDetection", "NaturalLanguageInference"]:
        verbalizer = VERBALIZER[task][dataset_name] 
        from copy import deepcopy
        predictions = []
        references = []
        count = 0
        for pred, ref in zip(prediction_list, reference_list):
            try:
                predictions.append(verbalizer.index(pred.strip().lower()))
                references.append(ref)
            except ValueError:
                try:
                  predictions.append([verbalizer[i].startswith(pred.strip().lower()) for i in range(len(verbalizer))].index(1))
                  references.append(ref)
                except ValueError:
                  count += 1
                  continue
        print("{}/{} format errors!".format(count, len(prediction_list)))
    elif task == "NameEntityRecognition":
        predictions = [extract_entity_from_gpt_output(pred) for pred in prediction_list]
        references = reference_list
    elif task == "QuestionAnswering":
        predictions = prediction_list
        references = reference_list
    else:
        raise NotImplementedError

    results = compute_metric(task, predictions, references)
    return results






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="turbo", choices=["turbo", "davinci3", "llama"])
    parser.add_argument('--setting', type=str, default="zero-shot", choices=["zero-shot", "in-context", "ood-in-context"])
    args = parser.parse_args()

    model_name = args.model_name
    setting = args.setting

    if model_name == "llama":
        from transformers import pipeline, LlamaForCausalLM, LlamaTokenizer

        model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
        tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        generator = pipeline(task="text-generation", 
                            model=model, 
                            tokenizer=tokenizer,
                            device=0)

    for task in ["SentimentAnalysis", "ToxicDetection", "NaturalLanguageInference", "NameEntityRecognition", "QuestionAnswering"]:

        processor = PROCESSOR[task]()
        max_tokens = MAX_TOKENS[task]
        for dataset_name in DATASET_LIST[task]:
            print(dataset_name)

            if setting == "ood-in-context" and dataset_name not in PROMPT[setting].keys():
                continue

            if task in ["SentimentAnalysis", "ToxicDetection", "NaturalLanguageInference"]:
                names = ["Dataset"]
                accuracies = ["Acc"]
            elif task == "NameEntityRecognition":
                names = ["Dataset"]
                micro_f1 = ["F1"]
            elif task == "QuestionAnswering":
                names = ["Dataset"]
                exact_match = ["Exact Match"]
                micro_f1 = ["F1"]

            dataset_path = os.path.join("datasets", "process", task, dataset_name)
            dataset = processor.get_examples(dataset_path, "test")

            os.makedirs(os.path.join("llm_results", task, dataset_name, model_name, setting), exist_ok=True)
            os.makedirs(os.path.join("llm_temp_output", task, dataset_name, model_name), exist_ok=True)
            wrapped_dataset, labels = wrap_dataset(dataset, setting, task, dataset_name)

            result_path =os.path.join("llm_results", task, dataset_name, model_name, setting, "0.tsv")
            result = main(wrapped_dataset, labels, dataset_name)
            print(dataset_name, result)

            if task in ["SentimentAnalysis", "ToxicDetection", "NaturalLanguageInference"]:
                names.append(dataset_name)
                accuracies.append(result)
            elif task == "NameEntityRecognition":
                names.append(dataset_name)
                micro_f1.append(result)
            elif task == "QuestionAnswering":
                names.append(dataset_name)
                exact_match.append(result["exact_match"])
                micro_f1.append(result["f1"])
            try:
              if task in ["SentimentAnalysis", "ToxicDetection", "NaturalLanguageInference"]:
                  results = pd.DataFrame([accuracies], columns=names)
                  results.to_csv(result_path, sep="\t", index=False)
              elif task == "NameEntityRecognition":
                  results = pd.DataFrame([micro_f1], columns=names)
                  results.to_csv(result_path, sep="\t", index=False)
              elif task == "QuestionAnswering":
                  results = pd.DataFrame([exact_match, micro_f1], columns=names)
                  results.to_csv(result_path, sep="\t", index=False)
            except OSError:
              print("OSError! Fail to write tsv file!")
        print("finish task:", task)
    print("finish all tasks")
from .dataloader import *
from .utils_qa import exact_match_score, f1_score, metric_max_over_ground_truths
import re

# extract the entities from the GPT output
def extract_entity_from_gpt_output(output):
    entity_dict = {}
    for entity_type in output.strip().strip(".").split(' | '):
        try:
            entity_type = entity_type.strip().split(': ')
            tag, entities = entity_type if len(entity_type) == 2 else (entity_type[0], ": ".join(entity_type[1:]))

            tag = tag.strip().lower()
            for entity in entities.split(', '):
                entity = entity.strip()
                if entity != 'None':
                    if tag in entity_dict:
                        entity_dict[tag].append(entity)
                    else:
                        entity_dict[tag] = [entity]
        except:
            continue

    entity_list = []
    for tag, entities in entity_dict.items():
        for entity in entities:
            entity_list.append((entity, tag))
    return entity_list



# extract the entities from the BIO scheme
def extract_entity_from_bio_scheme(tokens, tags):
    """
    Extracts entity-tag pairs from a list of tokens and their corresponding BIO tags.
    Returns a list of (entity, tag) tuples.
    """
    entities = []
    entity = ""
    tag = ""
    for i in range(len(tokens)):
        if tags[i] == "O":
            # Current token is not part of an entity
            if entity != "":
                # End of previous entity, append to entities list
                entities.append((entity.strip(), tag))
                entity = ""
                tag = ""
        elif tags[i][0] == "B":
            # Beginning of a new entity
            if entity != "":
                # End of previous entity, append to entities list
                entities.append((entity.strip(), tag))
            entity = tokens[i]
            tag = tags[i][2:]
        elif tags[i][0] == "I":
            # Continuation of current entity
            if entity != "":
                entity += " " + tokens[i]
            else:
                entity = tokens[i]
            tag = tags[i][2:]
    if entity != "":
        # End of last entity, append to entities list
        entities.append((entity.strip(), tag))
    return entities


from collections import defaultdict
def evaluate_ner(predictions, references):
    entity_types = set([item[1] for sublist in references for item in sublist])
    scores = defaultdict(dict)
    for entity_type in entity_types:
        # Convert prediction and ground-truth to sets of tuples for the current entity type
        prediction = {tuple(item) for sublist in predictions for item in sublist if item[1] == entity_type}
        ground_truth = {tuple(item) for sublist in references for item in sublist if item[1] == entity_type}

        # Calculate true positives, false positives, and false negatives for the current entity type
        tp = len(prediction.intersection(ground_truth))
        fp = len(prediction - ground_truth)
        fn = len(ground_truth - prediction)

        # Calculate precision, recall, and F1-score for the current entity type
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        # Store the scores for the current entity type
        scores[entity_type]['precision'] = precision
        scores[entity_type]['recall'] = recall
        scores[entity_type]['f1_score'] = f1_score

    # Calculate the average F1-score across all entity types
    avg_f1_score = sum([scores[entity_type]['f1_score'] for entity_type in entity_types]) / len(entity_types)

    # Return the scores as a dictionary
    scores['avg'] = {'precision': 0, 'recall': 0, 'f1_score': avg_f1_score}
    return scores




with open("./prompts/llm_instructions.txt", "r") as f:
    PROMPT = eval("\n".join(f.readlines()))

def wrap_dataset(dataset, setting, task, dataset_name, max_length = 1024):
    if setting == "ood-in-context":
        prompt = PROMPT[setting][dataset_name]
    else:
        prompt = PROMPT[setting][task] if task != "NameEntityRecognition" else PROMPT[setting][task][dataset_name]
    prompt = re.sub(r" +", " ", prompt)
    prompt = prompt.replace("\n\n", "\n").replace("\n ", "\n").strip()
    
    if task == "SentimentAnalysis" or task == "ToxicDetection":
        wrapped_dataset = [prompt.format(" ".join(sample.text_a.split(" ")[:max_length])) for sample in dataset]
        labels = [sample.label for sample in dataset]
    elif task == "NaturalLanguageInference":
        wrapped_dataset = [prompt.format(" ".join(sample.text_a.split(" ")[:max_length]), " ".join(sample.text_b.split(" ")[:max_length])) for sample in dataset]
        labels = [sample.label for sample in dataset]
    elif task == "NameEntityRecognition":
        wrapped_dataset = [prompt.format(" ".join(sample[:max_length])) for sample in dataset["tokens"]]
        labels = [extract_entity_from_bio_scheme(tokens, tags) for tokens, tags in zip(dataset["tokens"], dataset["tags"])]
    elif task == "QuestionAnswering":
        wrapped_dataset = [prompt.format(" ".join(sample["context"].split(" ")[:max_length]), sample["question"]) for sample in dataset]
        labels = [sample["answers"]["text"] for sample in dataset]
    return wrapped_dataset, labels

def compute_metric(task, predictions, references):
    if task in ["SentimentAnalysis", "ToxicDetection", "NaturalLanguageInference"]:
        results = 100.00 * sum([1 if pred == ref else 0 for pred, ref in zip(predictions, references)]) / len(references)
    elif task == "NameEntityRecognition":
        from datasets import load_metric
        results = 100.00 * evaluate_ner(predictions, references)["avg"]["f1_score"]
    elif task == "QuestionAnswering":
        em, f1 = 0, 0
        for pred, ground_truth in zip(predictions, references):
            em += metric_max_over_ground_truths(exact_match_score, pred, ground_truth)
            f1 += metric_max_over_ground_truths(f1_score, pred, ground_truth)
        results = {"exact_match": 100.00 * em / len(references), "f1": 100.00 * f1 / len(references)}
    return results

def evaluation(prompt, max_tokens, model_name="turbo", generator=None):
    if model_name == "turbo":
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=max_tokens
        )
        response = response["choices"][0]["message"]["content"].strip("\n").strip()
        return response
    elif model_name == "davinci3": 
        response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0,
                    )
        response = response["choices"][0]["text"].strip("\n").strip()
        return response
    elif model_name == "llama":
        response = generator(prompt, num_return_sequences=1, return_full_text=False, handle_long_generation="hole",
                            temperature=0, max_new_tokens=max_tokens, do_sample=False)
        response = response[0]["generated_text"].strip("\n").strip()
        return response
    else:
        raise NotImplementedError



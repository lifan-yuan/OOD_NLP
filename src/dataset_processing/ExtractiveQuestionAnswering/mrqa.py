import os
import re
import json
from tqdm import tqdm
from datasets import load_from_disk, Dataset


for split in ["train", "test"]:
    for dataset_name in tqdm(["HotpotQA", "NaturalQuestions", "NewsQA", "SearchQA", "TriviaQA"]):
        with open(f"./datasets/raw/QuestionAnswering/{split}/{dataset_name}.jsonl", 'rb') as f:
            examples = []
            for i, line in enumerate(f):
                if i == 0: # header
                    continue
                examples.append(json.loads(line))

        dataset_name = dataset_name.lower()
        os.makedirs(f"./datasets/process/QuestionAnswering/{dataset_name}", exist_ok=True)
        with open(f"./datasets/process/QuestionAnswering/{dataset_name}/{split}.json", "w", newline='\n') as f:
            for example in examples:
                title = ""
                id = example["id"]
                context = example["context"]
                for qas in example["qas"]:
                    # qid = qas["qid"]
                    question = qas["question"]
                    qas["answers"].extend([qas["detected_answers"][i]["text"] for i in range(len(qas["detected_answers"]))])
                    answers_text_list = list(set(qas["answers"]))
                    answers_answer_start_list = [qas["detected_answers"][i]["text"] for i in range(len(qas["detected_answers"]))]
                    # answers_text = max(answers_text_list, key=answers_text_list.count)
                    # answers_answer_start = max(answers_answer_start_list, key=answers_answer_start_list.count)
                    json.dump({"title":title, "context":context, "id":id, "question":question, 
                                "answers":{"text":answers_text_list, "answer_start":answers_answer_start_list}}, f)
                    f.write("\n")

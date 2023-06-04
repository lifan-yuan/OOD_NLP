import json
from datasets import Dataset

for split in ["train", "dev"]:
# for dataset_name in ["1_dbidaf"]:
    with open(f"./datasets/raw/QuestionAnswering/adversarialQA/combined/{split}.json", "r") as f:
        dataset = json.load(f)

        print(dataset.keys())
        print(len(dataset["data"]))
        print(dataset["data"][0].keys())
        print(len(dataset["data"][0]["paragraphs"]))
        print(len(dataset["data"][0]["paragraphs"][0]["qas"]))
    if split == "dev":
        split = "test"
    with open(f"./datasets/process/QuestionAnswering/advqa/{split}.json", "w") as f:
        for data in dataset["data"]:
            title = data["title"]
            for paragraph in data["paragraphs"]:
                context = paragraph["context"]
                for qas in paragraph["qas"]:
                    id = qas["id"]
                    question, answers = qas["question"], qas["answers"][0]
                    answers_text, answers_answer_start = [answers["text"]], [answers["answer_start"]]
                    json.dump({"id":id, "title":title, "context":context, "question":question, 
                                "answers":{"text":answers_text, "answer_start":answers_answer_start}}, f)
                    f.write("\n")


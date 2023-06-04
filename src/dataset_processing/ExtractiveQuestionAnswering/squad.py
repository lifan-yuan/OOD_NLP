import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_processing import set_seed, save_data, clean
set_seed(0)

from datasets import load_from_disk


# load and shuffle
squad = load_from_disk("./datasets/raw/QuestionAnswering/squad").shuffle(0)

# split 
train, test = squad["train"], squad["validation"]

# save
base_path = "./datasets/process/QuestionAnswering/squad"
os.makedirs(base_path, exist_ok=True)
print(train)
print(test)

train.to_json(os.path.join(base_path, "train.json"))
test.to_json(os.path.join(base_path, "test.json"))
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.utils_processing import set_seed, save_data, clean
set_seed(0)

from datasets import load_from_disk


# load and shuffle
squadshifts_amazon = load_from_disk("./datasets/raw/QuestionAnswering/squadshifts/amazon")["test"]
squadshifts_nyt = load_from_disk("./datasets/raw/QuestionAnswering/squadshifts/nyt")["test"]
squadshifts_reddit = load_from_disk("./datasets/raw/QuestionAnswering/squadshifts/reddit")["test"]

from datasets import concatenate_datasets
squadshifts = concatenate_datasets([squadshifts_amazon, squadshifts_nyt, squadshifts_reddit])
# split 
test = squadshifts

# save
base_path = "./datasets/process/QuestionAnswering/squadshifts"
os.makedirs(base_path, exist_ok=True)

test.to_json(os.path.join(base_path, "test.json"))
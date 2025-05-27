import random
import json
from datasets import load_dataset
from tqdm import tqdm


random.seed(12)

dataset = load_dataset("json", data_files="data/db2/*.jsonl", split="train").to_list()

num_samples = len(dataset)

train_size = int(0.8 * num_samples)
test_size = num_samples - train_size

train_ds = random.sample(dataset, train_size)
test_ds = [item for item in dataset if item not in train_ds]

train_ds = sorted(train_ds, key=lambda x: x["date"])
test_ds = sorted(test_ds, key=lambda x: x["date"])

with open("data/db2/train_matches.jsonl", "w") as f:
    for sample in tqdm(train_ds):
        sample["date"] = sample["date"].strftime("%Y-%m-%d")
        f.write(json.dumps(sample) + "\n")
        
with open("data/db2/test_matches.jsonl", "w") as f:
    for sample in tqdm(test_ds):
        sample["date"] = sample["date"].strftime("%Y-%m-%d")
        f.write(json.dumps(sample) + "\n")
from datasets import load_dataset
import numpy
from transformers import AutoTokenizer
from collections import Counter
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
# ds_name = "ag_news"
# dataset = load_dataset(ds_name, cache_dir="./cache_data")
dataset = load_dataset("glue", "mrpc", cache_dir="./cache_data")
# dataset = load_dataset("yelp_polarity", cache_dir="./cache_data")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir="./cache")
sample_split = dataset["train"]
length_count = Counter()
lengths = []
for data in tqdm(sample_split, total=len(sample_split)):
    # text = data["text"]
    # tokens = tokenizer(text)["input_ids"]
    #tokens = tokenizer(data["premise"], data["hypothesis"])["input_ids"]
    tokens = tokenizer(data["sentence1"], data["sentence2"])["input_ids"]
    length_count[len(tokens)] += 1
    lengths.append(2 ** len(tokens) if len(tokens) < 11 else 2**11)
    if len(lengths) == 100000:
        break
print(numpy.mean(lengths))
# plt.hist(lengths)
# plt.show()





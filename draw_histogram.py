import math
import os
from torch.utils.data import DataLoader
import torch
import random
import argparse
import json
import numpy as np
from torch import nn, optim
import loguru
from tqdm import tqdm
from datasets import Dataset
from create_dataset import (
    output_dir as dataset_dir,
    model_cache_dir
)
from matplotlib import pyplot as plt
from run import collate_fn

alL_train_datasets = dict()
all_test_datasets = dict()
output_dir = os.path.join("visualization", "value_histogram")
os.makedirs(output_dir, exist_ok=True)
for explainer in ["svs", "lig", "lime"]:
    print(f"plot values for {explainer}")
    train_dataset, test_dataset = torch.load(os.path.join(dataset_dir, f"data_{explainer}.pkl"))
    train_dataset, test_dataset = Dataset.from_dict(train_dataset), Dataset.from_dict(test_dataset)
    alL_train_datasets[explainer] = train_dataset
    all_test_datasets[explainer] = test_dataset
    # train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn)
    # for data in train_dataloader:
    all_vals = []
    for data in train_dataset:
        _vals = data["output"]
        all_vals.extend(_vals)
    plt.xlabel(f"{explainer} values")
    plt.hist(all_vals, bins=5)
    plt.title(f"histogram for {explainer} values")
    plt.savefig(os.path.join(output_dir, f"train_histogram_{explainer}.pdf"))
    plt.clf()
    all_val_log = [math.log(abs(x) + 1e-7, 10) for x in all_vals]
    plt.hist(all_val_log, bins=5)
    plt.title(f"histogram for log(abs({explainer})) values")
    plt.savefig(os.path.join(output_dir, f"train_log_histogram_{explainer}.pdf"))
    plt.clf()

        # print(_vals)


import torch
import os
from torch.utils.data import DataLoader
from datasets import Dataset, concatenate_datasets
from typing import *
from tqdm import tqdm

def collate_fn(features):
    ret = {}
    for k in features[0]:
        if k not in ["output_rank", "ft_label"]:
            ret[k] = torch.tensor([feature[k] for feature in features])
        else:
            ret[k] = torch.LongTensor([feature[k] for feature in features])
    return ret

def sort_by_file_size(paths):
    tgts = [(x, os.path.getsize(x)) for x in paths]
    tgts.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in tgts]

def get_zero_baselines(datasets: List[Dataset], target_model, tokenizer, args, device="cuda"):
    buf = []
    for dataset in datasets:
        dataloader = DataLoader(dataset, batch_size=20, collate_fn=collate_fn)
        remove_columns = ["output", "output_rank", "ft_label", "prediction_dist", "special_tokens_mask", "id"]
        zero_baselines = []
        for batch in tqdm(dataloader, total=len(dataloader), desc="adding baseline"):
            new_batch = dict()
            for k in batch:
                if k not in remove_columns:
                    # remove irrelevant columns for bert.forward()
                    new_batch[k] = batch[k].to(device)
            new_batch['input_ids'] = new_batch["input_ids"] * 0 + tokenizer.pad_token_id
            target_model_zero_output = target_model(**new_batch)[0].data.cpu()
            zero_baselines.append(target_model_zero_output)
        ret = torch.cat(zero_baselines, dim=0)
        _ds = Dataset.from_dict({
            "zero_baseline": ret
        })
        buf.append(concatenate_datasets([dataset, _ds], axis=1))
    return buf




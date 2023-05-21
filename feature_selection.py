import json
import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr
import os
import glob
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, load_from_disk
from utils import sort_by_file_size


#task_name = "yelp_polarity"
task_name = "mnli"
if task_name == "yelp_polarity":
    model_name = "textattack/bert-base-uncased-yelp-polarity"
    dataset = load_from_disk("thermostat/experiments/thermostat/datasets/yelp_polarity")
    candidates = ["kernelshap-3600", "kernelshap-3600-sample200", "kernelshap-500-sample2000",
                  "kernelshap-500-sample8000", "svs-3600", "lime", "lime-200"]
elif task_name == "mnli":
    # original textattack model has the problem of label mismatch, fixed by myself,
    # see issues at: https://github.com/QData/TextAttack/issues/684.
    # The fixed accuracy_mm is 84.44% and is 7% before the fix applied.
    model_name = "chromeNLP/textattack_bert_base_MNLI_fixed"
    dataset = load_from_disk("thermostat/experiments/thermostat/datasets/multi_nli")
    candidates = ["kernelshap-2000", "kernelshap-2000-sample200", "kernelshap-2000-sample2000", "kernelshap-2000-sample8000", "lime-2000", "lime-2000-sample200", "svs-2000"]
else:
    raise NotImplementedError

# model_name = "textattack/bert-base-uncased-imdb"
# model_name = "textattack/bert-base-uncased-MNLI"
# model_name = "textattack/bert-base-uncased-yelp-polarity"
model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)


def textattack_mnli_label_mapping(label):
    label_mapping_from_model_to_gt = {
        0: 2,
        1: 0,
        2: 1
    }
    return label_mapping_from_model_to_gt[label]


if model_name == "textattack/bert-base-uncased-MNLI":
    label_mapping = textattack_mnli_label_mapping
else:
    label_mapping = lambda x: x
# for explainer in ["kernelshap-2000-sample2000",]:
def get_eraser_performance(attribution, model, input_data, tokenizer, label):
    top_indexes = list(attribution.argsort()[::-1])
    num_actual_tokens = input_data["attention_mask"].sum().item()
    res = []
    res.append(int(model(**{k: v.cuda() for k, v in input_data.items()})[0].argmax().item() == label))
    for topP in [0.01, 0.05, 0.10, 0.20, 0.50]:
        num_token_masked = int(num_actual_tokens * topP)
        # num_token_masked = topP
        token_masked = torch.LongTensor(top_indexes[: num_token_masked])
        _input_ids = input_data['input_ids'].clone()
        _input_ids[0][token_masked] = tokenizer.pad_token_id
        _output = model(
            input_ids=_input_ids.cuda(),
            attention_mask=input_data["attention_mask"].cuda(),
            token_type_ids=input_data["token_type_ids"].cuda()
        )[0]
        global label_mapping
        res.append(1 if label_mapping(_output.argmax().item()) == label else 0)
    return res


for explainer in candidates:
    if task_name == "mnli":
        path = f"path/to/thermostat/experiments/thermostat/multi_nli/bert/{explainer}/"
    elif "yelp" in task_name:
        path = f"path/to/thermostat/experiments/thermostat/yelp_polarity/bert/{explainer}/"
    print("NOW evaluating:", explainer)
    seed_dirs = glob.glob(path + "seed_*")
    all_correlations = []
    all_ps = []
    all_mask_check = 0
    seed_dir0 = os.path.join(path, seed_dirs[0])
    seed_file0 = sort_by_file_size(glob.glob(os.path.join(seed_dir0, "*.jsonl")))[0]
    seed_file_path0 = os.path.join(seed_dir0, seed_file0)
    print(seed_file_path0)
    all_eraser_res = []
    for _seed_dir1 in seed_dirs[1: ]:
        seed_dir1 = os.path.join(path, _seed_dir1)
        seed_file1 = sort_by_file_size(glob.glob(os.path.join(seed_dir1, "*.jsonl")))[0]
        seed_file_path1 = os.path.join(seed_dir1, seed_file1)
        print(seed_file_path1)
        topks = {1: [], 5: [], 10: [], 20: []}
        all_eraser_res0 = []
        all_eraser_res1 = []
        try:
            with open(seed_file_path0, "r", encoding='utf-8') as f_in0, open(seed_file_path1, "r", encoding='utf-8') as f_in1:
                buf0, buf1 = f_in0.readlines(), f_in1.readlines()
                buf0, buf1 = buf0[:500], buf1[:500]
                id_counter = 0
                acc_num = 0
                for line0, line1 in tqdm(zip(buf0, buf1), total=len(buf0)):
                    obj0, obj1 = json.loads(line0), json.loads(line1)
                    attr0, attr1 = obj0["attributions"], obj1["attributions"]
                    attr0, attr1 = np.array(attr0), np.array(attr1)
                    assert (obj0["dataset"].get("start", 0) + obj0["index_running"]) == (
                            obj1["dataset"].get("start", 0) + obj1["index_running"])
                    data_id = obj0["dataset"].get("start", 0) + obj0["index_running"]
                    assert ((torch.LongTensor(obj0["input_ids"])) == (torch.LongTensor(obj1["input_ids"]))).all()
                    assert obj0["label"] == obj1['label']
                    if ((attr0 == 0) != (attr1 == 0)).any():
                        all_mask_check += 1

                    in0, in1 = obj0["input_ids"], obj1["input_ids"]
                    acc_num += 1 if label_mapping(torch.tensor(obj0['predictions']).argmax().item()) == obj0['label'] else 0
                    assert in0 == in1
                    postfix = sum(np.array(in0) == 0)
                    if postfix > 0:
                        attr0_pruned = attr0[:-postfix]
                        attr1_pruned = attr1[:-postfix]
                        in0_pruned = in0[:-postfix]
                        in1_pruned = in1[:-postfix]
                    sort0 = attr0.argsort()
                    sort1 = attr1.argsort()

                    real_instance = dataset[data_id]
                    if "nli" in path:
                        _input = tokenizer(real_instance["premise"], real_instance["hypothesis"],
                                           truncation=obj0['model']['tokenization']["truncation"],
                                           max_length=obj0['model']['tokenization']['max_length'],
                                           padding=obj0['model']["tokenization"]["padding"],
                                           return_tensors="pt"
                                           )
                    else:
                        _input = tokenizer(real_instance["text"],
                                           truncation=obj0['model']['tokenization']["truncation"],
                                           max_length=obj0['model']['tokenization']['max_length'],
                                           padding=obj0['model']["tokenization"]["padding"],
                                           return_tensors="pt"
                                           )
                    # assert (torch.LongTensor(_input["input_ids"].tolist()) == torch.LongTensor(obj0["input_ids"])).all()
                    assert obj0["label"] == obj1["label"]
                    # assert (torch.LongTensor(_input["input_ids"].tolist()) == torch.LongTensor(in0_pruned)).all()

                    res0 = get_eraser_performance(attr0, model, _input, tokenizer, obj0['label'])
                    res1 = get_eraser_performance(attr1, model, _input, tokenizer, obj1['label'])
                    all_eraser_res0.append(res0)
                    all_eraser_res1.append(res1)

                    _spearman, _pval = spearmanr(attr0_pruned, attr1_pruned)
                    all_correlations.append(_spearman)
                    all_ps.append(_pval)
                    for key in topks:
                        topk_intersection = set(sort0[::-1][:key].tolist()) & set(sort1[::-1][:key].tolist())
                        topk_intersection = [in0[x] for x in sorted(topk_intersection)]
                        _topk = len(topk_intersection)
                        topks[key].append(_topk)
                    id_counter += 1
            print("acc:", acc_num / len(buf0))
            print(
                f"spearman correlation: {np.mean(all_correlations)} ({np.std(all_correlations)}, {np.min(all_correlations)}, {np.max(all_correlations)})", )
            print(f"spearman ps: {np.mean(all_ps)} ({np.std(all_ps)})", )
            print(f"mask mismatch rate: {all_mask_check / len(all_ps)}")
            for key in topks:
                print(f"top{key}: {np.mean(topks[key])}")
            # print('feat selection res:')
            if len(all_eraser_res) == 0:
                all_eraser_res.append(torch.tensor(all_eraser_res0).float().mean(dim=0))
            all_eraser_res.append(torch.tensor(all_eraser_res1).float().mean(dim=0))
        except AssertionError as e:
            print(e)
            print("assertion error, skip ...")
    print('feat selection res:')
    print("sample0")
    print(all_eraser_res[0])
    all_eraser_res = torch.stack(all_eraser_res, dim=0)
    print("mean")
    print(all_eraser_res.mean(dim=0))
    print("std")
    print(all_eraser_res.std(dim=0))

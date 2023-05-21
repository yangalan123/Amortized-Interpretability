import json
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
import os
import glob
def sort_by_file_size(paths):
    tgts = [(x, os.path.getsize(x)) for x in paths]
    tgts.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in tgts]

# feel free to add other explainer you want to compare
for explainer in ["kernelshap-2000-sample200",]:
    # path = f"path/to/thermostat/experiments/thermostat/yelp_polarity/bert/{explainer}/"
    path = f"path/to/thermostat/experiments/thermostat/multi_nli/bert/{explainer}/"
    # path2 = f"path/to/thermostat/experiments/thermostat/yelp_polarity/bert/svs-3600/"
    path2 = f"path/to/thermostat/experiments/thermostat/multi_nli/bert/svs-2000/"
    print("NOW evaluating:", explainer)
    seed_dirs = glob.glob(path + "seed_*")
    seed_dirs2 = glob.glob(path2 + "seed_*")
    if len(seed_dirs) < 2:
        print("not enough seed dirs for {}".format(path))
        exit()
    seed_aggr_spearman = []
    seed_aggr_l2 = []
    seed_topks = {1: [], 5: [], 10: [], 20: []}

    count = 0
    mse_count = 0
    for i in range(len(seed_dirs)):
        for j in range(len(seed_dirs2)):

            all_correlations = []
            all_ps = []
            all_l2 = []
            all_mask_check = 0
            seed_dir0 = os.path.join(path, seed_dirs[i])
            seed_dir1 = os.path.join(path2, seed_dirs2[j])
            seed_file0 = sort_by_file_size(glob.glob(os.path.join(seed_dir0, "*.jsonl")))[0]
            seed_file1 = sort_by_file_size(glob.glob(os.path.join(seed_dir1, "*.jsonl")))[0]
            seed_file_path0 = os.path.join(seed_dir0, seed_file0)
            seed_file_path1 = os.path.join(seed_dir1, seed_file1)
            # print(seed_file_path0)
            # print(seed_file_path1)
            topks = {1:[], 5:[], 10:[], 20:[]}
            with open(seed_file_path0, "r", encoding='utf-8') as f_in0, open(seed_file_path1, "r", encoding='utf-8') as f_in1:
                buf0, buf1 = f_in0.readlines(), f_in1.readlines()
                #assert len(buf0) == len(buf1), f"{len(buf0)}, {len(buf1)}"
                for line0, line1 in tqdm(zip(buf0, buf1), total=len(buf0)):
                    obj0, obj1 = json.loads(line0), json.loads(line1)
                    attr0, attr1 = obj0["attributions"], obj1["attributions"]
                    in0, in1 = obj0["input_ids"], obj1["input_ids"]
                    attr0, attr1 = np.array(attr0), np.array(attr1)
                    assert in0 == in1
                    if ((attr0 == 0) != (attr1 == 0)).any():
                        all_mask_check += 1
                    postfix = sum(np.array(in0) == 0)
                    #assert postfix < len(attr0)
                    if postfix > 0:
                        attr0 = attr0[:-postfix]
                        attr1 = attr1[:-postfix]
                    assert len(attr0) > 0 and len(attr0) == len(attr1), f"{len(attr0)}"
                    count += 1
                    #print(attr0)
                    #print(attr1)
                    #print(len(attr0), postfix)
                    _spearman, _pval = spearmanr(attr0, attr1)
                    all_correlations.append(_spearman)
                    all_ps.append(_pval)
                    mse = mean_squared_error(attr0, attr1)
                    if mse > 1e2:
                       mse_count += 1
                    else:
                        all_l2.append(mean_squared_error(attr0, attr1))
                    sort0 = attr0.argsort()
                    sort1 = attr1.argsort()
                    for key in topks:
                        _topk = len(set(sort0[::-1][:key].tolist()) & set(sort1[::-1][:key].tolist()))
                        topks[key].append(_topk)
            # print(f"spearman correlation: {np.mean(all_correlations)} ({np.std(all_correlations)}, {np.min(all_correlations)}, {np.max(all_correlations)})", )
            # print(f"spearman ps: {np.mean(all_ps)} ({np.std(all_ps)})", )
            # print(f"mask mismatch rate: {all_mask_check / len(all_ps)}")
            # for key in topks:
            #     print(f"top{key}: {np.mean(topks[key])}")
            seed_aggr_spearman.append(np.mean(all_correlations))
            seed_aggr_l2.append(np.mean(all_l2))
            for key in seed_topks:
                seed_topks[key].append(np.mean(topks[key]))
    print(f"spearman correlation: {np.mean(seed_aggr_spearman)} ({np.std(seed_aggr_spearman)}, {np.min(seed_aggr_spearman)}, {np.max(seed_aggr_spearman)})", )
    print(f"MSE correlation: {np.mean(seed_aggr_l2)} ({np.std(seed_aggr_l2)}, {np.min(seed_aggr_l2)}, {np.max(seed_aggr_l2)})", )
    print(f"ignored MSE pairs: {mse_count} / {count}, {mse_count / count}")
    for key in seed_topks:
        print(f"top{key}: {np.mean(seed_topks[key])} ({np.std(seed_topks[key])})")







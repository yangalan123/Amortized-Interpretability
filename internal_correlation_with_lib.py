import json
import thermostat
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr
import os
import glob
data_cache_dir = "./datasets/imdb"
data = thermostat.load(f"imdb-bert-svs", cache_dir=data_cache_dir)
for explainer in ["kernelshap-3600-sample200", "kernelshap-3600"]:
    count = 0
    path = f"path/to/thermostat/experiments/thermostat/multi_nli/bert/{explainer}/"
    seed_dirs = glob.glob(path + "seed_*")
    if len(seed_dirs) > 2:
        seed_dirs = seed_dirs[:2]
    all_correlations_1_ref = []
    all_correlations_2_ref = []
    all_correlations = []
    all_ps = []
    all_mask_check = 0
    seed_dir0 = os.path.join(path, seed_dirs[0])
    seed_dir1 = os.path.join(path, seed_dirs[1])
    seed_file0 = glob.glob(os.path.join(seed_dir0, "*.jsonl"))[0]
    seed_file1 = glob.glob(os.path.join(seed_dir1, "*.jsonl"))[0]
    seed_file_path0 = os.path.join(seed_dir0, seed_file0)
    seed_file_path1 = os.path.join(seed_dir1, seed_file1)
    print(seed_file_path0)
    print(seed_file_path1)
    with open(seed_file_path0, "r", encoding='utf-8') as f_in0, open(seed_file_path1, "r", encoding='utf-8') as f_in1:
        buf0, buf1 = f_in0.readlines(), f_in1.readlines()
        assert len(buf0) == len(buf1), f"{len(buf0)}, {len(buf1)}"
        for line0, line1 in tqdm(zip(buf0, buf1), total=len(buf0)):
            obj0, obj1 = json.loads(line0), json.loads(line1)
            ref = data[count].attributions
            if count == 0:
                print(data[count])
            attr0, attr1 = obj0["attributions"], obj1["attributions"]
            attr0, attr1 = np.array(attr0), np.array(attr1)
            if ((attr0 == 0) != (attr1 == 0)).any():
                all_mask_check += 1
            _spearman, _pval = spearmanr(attr0, ref)
            all_correlations_1_ref.append(_spearman)
            _spearman, _pval = spearmanr(attr1, ref)
            all_correlations_2_ref.append(_spearman)
            _spearman, _pval = spearmanr(attr1, attr0)
            all_correlations.append(_spearman)
            count += 1
    print(f"spearman correlation: {np.mean(all_correlations)} ({np.std(all_correlations)}, {np.min(all_correlations)}, {np.max(all_correlations)})", )
    print(f"spearman correlation: {np.mean(all_correlations_1_ref)} ({np.std(all_correlations_1_ref)}, {np.min(all_correlations_1_ref)}, {np.max(all_correlations_1_ref)})", )
    print(f"spearman correlation: {np.mean(all_correlations_2_ref)} ({np.std(all_correlations_2_ref)}, {np.min(all_correlations_2_ref)}, {np.max(all_correlations_2_ref)})", )
    print(f"mask mismatch rate: {all_mask_check / len(all_correlations)}")






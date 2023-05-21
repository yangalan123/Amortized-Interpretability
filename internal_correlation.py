import json
import pickle

from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import glob
from matplotlib import container


def sort_by_file_size(paths):
    tgts = [(x, os.path.getsize(x)) for x in paths]
    tgts.sort(key=lambda x: x[1], reverse=True)
    return [x for x in tgts]


np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
plt.rcParams.update({'font.size': 16})
plt.rcParams["figure.figsize"] = (10, 6)
cmap = ["red", "blue", "orange", "purple", "cyan", "green", "lime", "#bb86fc"]
markers = [".", "v", "*", "o", "s", "d", "P", "p"]
method_length_spearman_decomp = {}
task_name = "mnli"
candidates = ["svs-2000", "kernelshap-2000", "kernelshap-2000-sample200", "kernelshap-2000-sample2000", "kernelshap-2000-sample8000", "lime-2000", "lime-2000-sample200"]
keys = ["svs-25", "kernelshap-25", "kernelshap-200", "kernelshap-2000", "kernelshap-8000", "lime-25", "lime-200"]
for explainer_i, explainer in enumerate(candidates):
    if task_name == "yelp":
        path = f"path/to/thermostat/experiments/thermostat/yelp_polarity/bert/{explainer}/"
    else:
        path = f"path/to/thermostat/experiments/thermostat/multi_nli/bert/{explainer}/"
    print("NOW evaluating:", explainer)
    seed_dirs = glob.glob(path + "seed_*")
    seed_aggr_spearman = []
    seed_aggr_l2 = []
    seed_topks = {1: [], 5: [], 10: [], 20: []}
    method_length_spearman_decomp[explainer] = {}

    for i in range(len(seed_dirs)):
        for j in range(i + 1, len(seed_dirs)):

            all_correlations = []
            all_ps = []
            all_l2 = []
            all_ill_condition = 0
            all_length_decomp = {}
            all_mask_check = 0
            seed_dir0 = os.path.join(path, seed_dirs[i])
            seed_dir1 = os.path.join(path, seed_dirs[j])
            seed_file0, seed_file0_size = sort_by_file_size(glob.glob(os.path.join(seed_dir0, "*.jsonl")))[0]
            seed_file1, seed_file1_size = sort_by_file_size(glob.glob(os.path.join(seed_dir1, "*.jsonl")))[0]
            if seed_file0_size > 2 * seed_file1_size or seed_file1_size > 2 * seed_file0_size:
                print(seed_file0, seed_file0_size)
                print(seed_file1, seed_file1_size)
                continue
            seed_file_path0 = os.path.join(seed_dir0, seed_file0)
            seed_file_path1 = os.path.join(seed_dir1, seed_file1)
            topks = {1: [], 5: [], 10: [], 20: []}
            with open(seed_file_path0, "r", encoding='utf-8') as f_in0, open(seed_file_path1, "r",
                                                                             encoding='utf-8') as f_in1:
                buf0, buf1 = f_in0.readlines(), f_in1.readlines()
                # assert len(buf0) == len(buf1), f"{len(buf0)}, {len(buf1)}"
                for line0, line1 in tqdm(zip(buf0, buf1), total=min(len(buf0), len(buf1))):
                    obj0, obj1 = json.loads(line0), json.loads(line1)
                    attr0, attr1 = obj0["attributions"], obj1["attributions"]
                    in0, in1 = obj0["input_ids"], obj1["input_ids"]
                    attr0, attr1 = np.array(attr0), np.array(attr1)
                    assert in0 == in1
                    if ((attr0 == 0) != (attr1 == 0)).any():
                        all_mask_check += 1
                    postfix = sum(np.array(in0) == 0)
                    # assert postfix < len(attr0)
                    if postfix > 0:
                        attr0 = attr0[:-postfix]
                        attr1 = attr1[:-postfix]
                    assert len(attr0) > 0 and len(attr0) == len(attr1), f"{len(attr0)}"
                    _mse = ((attr0 - attr1)**2).sum() / len(attr0)
                    if _mse > 1e3:
                        # due to ill-conditioned matrix and float error, kernelshap can give bad values sometimes
                        all_ill_condition += 1
                        continue
                    _spearman, _pval = spearmanr(attr0, attr1)
                    if len(attr0) not in all_length_decomp:
                        all_length_decomp[len(attr0)] = []
                    all_length_decomp[len(attr0)].append(_spearman)
                    all_correlations.append(_spearman)
                    all_ps.append(_pval)
                    all_l2.append(_mse)
                    sort0 = attr0.argsort()
                    sort1 = attr1.argsort()
                    for key in topks:
                        _topk = len(set(sort0[::-1][:key].tolist()) & set(sort1[::-1][:key].tolist()))
                        topks[key].append(_topk)
            if all_ill_condition > 0:
                print("find ill_condition: ", all_ill_condition, 100 * all_ill_condition / min(len(buf0), len(buf1)))
            seed_aggr_spearman.append(np.mean(all_correlations))
            seed_aggr_l2.append(np.mean(all_l2))
            for length in all_length_decomp:
                if length not in method_length_spearman_decomp[explainer]:
                    method_length_spearman_decomp[explainer][length] = []
                method_length_spearman_decomp[explainer][length].append(np.mean(all_length_decomp[length]))

            for key in seed_topks:
                seed_topks[key].append(np.mean(topks[key]))
    print(
        f"spearman correlation: {np.mean(seed_aggr_spearman)} ({np.std(seed_aggr_spearman)}, {np.min(seed_aggr_spearman)}, {np.max(seed_aggr_spearman)})", )
    print(
        f"MSE correlation: {np.mean(seed_aggr_l2)} ({np.std(seed_aggr_l2)}, {np.min(seed_aggr_l2)}, {np.max(seed_aggr_l2)})", )
    for key in seed_topks:
        print(f"top{key}: {np.mean(seed_topks[key])} ({np.std(seed_topks[key])})")
    print(f"${'{:.2f}'.format(np.mean(seed_aggr_spearman))} (\pm {'{:.2f}'.format(np.std(seed_aggr_spearman))})$ & "
          f"${'{:.2f}'.format(np.mean(seed_topks[5]))} (\pm {'{:.2f}'.format(np.std(seed_topks[5]))})$ & "
          f"${'{:.2f}'.format(np.mean(seed_topks[10]))} (\pm {'{:.2f}'.format(np.std(seed_topks[10]))})$ & "
          f"${'{:.2f}'.format(np.mean(seed_aggr_l2))} (\pm {'{:.2f}'.format(np.std(seed_aggr_l2))})$ & "
          )
    if "lime" not in explainer:
        xs = list(method_length_spearman_decomp[explainer].keys())
        xs.sort()
        if task_name == "mnli":
            xs = [x for x in xs if x < 80 and x > 5]
        ys = [np.mean(method_length_spearman_decomp[explainer][x]) for x in xs]
        yerr = [np.std(method_length_spearman_decomp[explainer][x]) for x in xs]
        plt.plot(xs, ys, color=cmap[explainer_i],
                     marker=markers[explainer_i], label=keys[explainer_i].replace("kernelshap", "ks") if "kernelshap" in keys[explainer_i] else keys[explainer_i])

handles, labels = plt.gca().get_legend_handles_labels()
handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 1, box.height])

plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel("Spearman's Correlation", fontsize=22)
plt.xlabel("#(Tokens) Per Instance", fontsize=22)
target_dir = os.path.join("visualization", "internal_correlation", task_name)
os.makedirs(target_dir, exist_ok=True)
target_fp = os.path.join(target_dir, "internal_correlation_w_length_decomp_wo_errorbar.pdf")
plt.tight_layout()

plt.savefig(target_fp)
pickle.dump(method_length_spearman_decomp, open(os.path.join(target_dir, "dump.pkl"), "wb"))

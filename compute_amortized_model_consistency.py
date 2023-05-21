import glob
import numpy as np
from scipy.stats import spearmanr
import torch
from tqdm import trange
from tqdm import tqdm

for prop in [0.1, 0.3, 0.5, 0.7, 1.0]:
    print(f"eval {prop}")
    all_spearman = []
    all_spearman_1 = []
    all_spearman_2 = []
    gt_spearmans = []
    diffs = []

    # test_outputs path, change this to your own path, for example:
    seed_path_format = "/path/to/amortized_model_formal/multi_nli/lr_5e-05-epoch_30/seed_{}_prop_{}/model_svs_norm_False_discrete_False/test_outputs.pkl"
    # seed_path_format = "/path/to/amortized_model_formal/yelp_polarity/lr_5e-05-epoch_30/seed_{}_prop_{}/model_svs_norm_False_discrete_False/test_outputs.pkl"
    seeds = [0, 1, 2]
    seed_gt_spearmans = []
    seed_all_spearmans = []
    seed_l2_delta = []
    for seed_1 in tqdm(range(len(seeds)), position=0, leave=True):
        for seed_2 in tqdm(range(seed_1 + 1, len(seeds)), position=0, leave=True):
            seed_path1 = seed_path_format.format(seeds[seed_1], prop)
            seed_path2 = seed_path_format.format(seeds[seed_2], prop)
            output_pred1, output_ref1, output_attn1, output_in1 = torch.load(seed_path1)
            output_pred2, output_ref2, output_attn2, output_in2 = torch.load(seed_path2)

            for i in range(len(output_ref1)):
                assert (output_attn1[i] == output_attn2[i]).all()
                assert (output_in1[i] == output_in2[i]).all()
                sp, p = spearmanr(output_ref1[i], output_ref2[i])
                gt_spearmans.append(sp)
                sp, p = spearmanr(output_pred1[i], output_pred2[i])
                all_spearman.append(sp)
                all_spearman_1.append(spearmanr(output_pred1[i], output_ref1[i])[0])
                all_spearman_2.append(spearmanr(output_pred2[i], output_ref2[i])[0])
                diffs.append(np.linalg.norm(output_pred1[i] - output_pred2[i]))
            seed_gt_spearmans.append(np.mean(gt_spearmans))
            seed_all_spearmans.append(np.mean(all_spearman))
            seed_l2_delta.append(np.mean(diffs))
    print("gt spearman: ", np.mean(seed_gt_spearmans), np.std(seed_gt_spearmans))
    print("all spearman: ", np.mean(seed_all_spearmans), np.std(seed_all_spearmans))
    print("l2_delta: ", np.mean(seed_l2_delta), np.std(seed_l2_delta))

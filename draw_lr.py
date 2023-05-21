from matplotlib import pyplot as plt
import numpy as np
import glob
import os

# intersections = [22, 28, 25, 27, 27]
# kendall = [0.0707, 0.0281, -0.07, 0.003, 0.0261]
# spearman = [0.1046, 0.0423, -0.05, 0.005, 0.0389]
# proportions = [0.1, 0.3, 0.5, 0.7, 1]
# model_name = "svs"


#intersections = [24, 29, 24, 24, 25]
#kendall = [0.0064, 0.0119, 0.00580, -0.00196, -0.0119]
#spearman = [0.0096, 0.0177, 0.00866, -0.002954, -0.0180]
proportions = [0.1, 0.3, 0.5, 0.7, 1]
# proportions = [0.1, 0.3, 0.5]
proportions_str = ["0.1", "0.3", "0.5", "0.7", "1.0"]
plt.rcParams.update({'font.size': 16})
# plt.rcParams["figure.figsize"] = (10, 6)
# proportions_str = ["0.1", "0.3", "0.5"]
# output_dir = "visualization/learning_curve"
# output_dir = "visualization/learning_curve_mnli"
# task_name = "mnli"
task_name = "yelp"
# output_dir = "visualization/learning_curve_yelp_with_fastshap_baseline"
output_dir = f"visualization/learning_curve_{task_name}_with_fastshap_baseline"
os.makedirs(output_dir, exist_ok=True)


# for model_name in ["svs", "lig", "lime"]:
for model_name in ["svs", ]:
    all_intersections = dict()
    all_kendal = dict()
    all_spearman = dict()
    for target_eval in ["svs",]:
        intersections = []
        kendall = []
        spearman = []

        for prop_str in proportions_str:
            if task_name == "mnli":
                record_dir = f"path/to/amortized_model_formal/multi_nli/lr_5e-05-epoch_30/seed_*_prop_{prop_str}/model_{model_name}_norm_False_discrete_False"
                fastshap_baseline = 0.23
            else:
                assert task_name == "yelp"
                record_dir = f"path/to/amortized_model_formal/yelp_polarity/lr_5e-05-epoch_30/seed_*_prop_{prop_str}/model_{model_name}_norm_False_discrete_False"
                fastshap_baseline = 0.18
            logs = glob.glob(os.path.join(record_dir, f"test_log_no_pad_{target_eval}.txt"))
            _intersections = []
            _spearmans = []
            _kendals = []
            for logfn in logs:
                with open(logfn, "r", encoding='utf-8') as f_in:
                    for line in f_in:
                        # if "loss at epoch" in line:
                        #     _num = float(line.strip().split(":")[-1].strip())
                        #     loss.append(_num)
                        if "intersection: " in line:
                            _num = float(line.strip().split("intersection: ")[-1])
                            _intersections.append(_num)
                        if "spearman:" in line:
                            _num = float(line.strip().split("correlation=")[1].split(",")[0])
                            _spearmans.append(_num)
                        if "kendaltau:" in line:
                            _num = float(line.strip().split("correlation=")[1].split(",")[0])
                            _kendals.append(_num)
                            break
            intersections.append(_intersections)
            spearman.append(_spearmans)
            kendall.append(_kendals)

        for ys, yname, color in zip([intersections, kendall, spearman], ["intersections", "kendall", "spearman"], ["r", "g", "b"]):
            print(f"plotting {yname} for base-{model_name}-eval-{target_eval}")
            arr_ys = np.array(ys)
            plt.errorbar(range(len(proportions)), np.mean(arr_ys, axis=1), yerr=np.std(arr_ys, axis=1), capsize=3, fmt='o-', color=color, label="ours")
            plt.xticks(range(len(proportions)), [f"{int(x * 100)}%" for x in proportions])
            plt.xlabel("proportion of data used", fontsize=22)
            plt.ylabel(yname.capitalize() + " w/ SVS-25", fontsize=22)
            if yname == "spearman":
                plt.axhline(y=fastshap_baseline, label="fastshap", linestyle="--")
            # plt.title(f"{model_name}_{yname}")
                plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"base_{model_name}_target_{target_eval}_{yname}_{task_name}.pdf"))
            plt.clf()

        all_intersections[target_eval] = intersections
        all_kendal[target_eval] = kendall
        all_spearman[target_eval] = spearman

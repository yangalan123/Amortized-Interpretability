from matplotlib import pyplot as plt
import glob
import os
import numpy as np

# intersections = [22, 28, 25, 27, 27]
# kendall = [0.0707, 0.0281, -0.07, 0.003, 0.0261]
# spearman = [0.1046, 0.0423, -0.05, 0.005, 0.0389]
# proportions = [0.1, 0.3, 0.5, 0.7, 1]
# model_name = "svs"


#intersections = [24, 29, 24, 24, 25]
#kendall = [0.0064, 0.0119, 0.00580, -0.00196, -0.0119]
#spearman = [0.0096, 0.0177, 0.00866, -0.002954, -0.0180]
seed_dirs = glob.glob("amortized_model_debug/seed_*")
model_name = "svs"
output_dir = "visualization/learning_curve_debug_discrete"
os.makedirs(output_dir, exist_ok=True)

intersections = []
kendalls = []
spearmans = []
losses = []
for seed_dir in seed_dirs:
    intersection = []
    spearman = []
    kendall = []
    loss = []
    record_dir = f"{seed_dir}/model_{model_name}_norm_False_discrete_True"
    logs = glob.glob(os.path.join(record_dir, "log*txt"))
    if len(logs) > 1:
        file_sizes = [(os.path.getsize(_path), _path) for _path in logs]
        file_sizes.sort(key=lambda x: x[0], reverse=True)
        logfn = file_sizes[0][1]
    else:
        logfn = logs[0]
    with open(logfn, "r", encoding='utf-8') as f_in:
        for line in f_in:
            if "loss at epoch" in line:
                _num = float(line.strip().split(":")[-1].strip())
                loss.append(_num)
            if "intersection: " in line:
                _num = float(line.strip().split("intersection: ")[-1])
                intersection.append(_num)
            if "Spearman" in line:
                _num = float(line.strip().split("correlation=")[1].split(",")[0])
                spearman.append(_num)
            if "kendal" in line:
                _num = float(line.strip().split("correlation=")[1].split(",")[0])
                kendall.append(_num)
    losses.append(loss)
    intersections.append(intersection)
    spearmans.append(spearman)
    kendalls.append(kendall)

losses = np.array(losses)
intersections = np.array(intersections)
spearmans = np.array(spearmans)
kendalls = np.array(kendalls)

for ys, yname, color in zip([losses, intersections, kendalls, spearmans], ["losses", "intersections", "kendall", "spearman"], ["black", "black", "black", "black"]):
    plt.errorbar(range(len(losses[0])), np.mean(ys, axis=0), yerr=np.std(ys, axis=0), fmt="-o", color=color, capthick=5, ecolor='g', capsize=3)
    plt.xlabel("epoch")
    plt.ylabel(yname)
    plt.title(f"{model_name}_{yname}")
    plt.savefig(os.path.join(output_dir, f"{model_name}_{yname}.pdf"))
    plt.clf()


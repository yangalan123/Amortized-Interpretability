import matplotlib.pyplot as plt
import random
import os
import csv
from matplotlib import container
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

random.seed(1)
plt.rcParams.update({'font.size': 16})
plt.rcParams["figure.figsize"] = (10, 6)
filename = "feature_selection_yelp.csv"
# filename = "feature_selection_mnli.csv"
with open(filename, 'r', encoding='utf-8') as f_in:
# with open("feature_selection_mnli.csv", 'r', encoding='utf-8') as f_in:
    reader = csv.DictReader(f_in)
    res = dict()
    for line in reader:
        for key in line:
            if "Ratio" not in key:
                if key not in res:
                    res[key] = []
                res[key].append(float(line[key]))
    # xs = [f"Top {x}% Mask" for x in [1, 5, 10, 20, 50]]
    # xs = [f"{x}%" for x in [1, 5, 10, 20, 50]]
    xs = [f"{x}%" for x in [1, 5, 10, 20]]
    target_dir = "visualization"
    os.makedirs(target_dir, exist_ok=True)
    # cmap = get_cmap(len(res))
    cmap = ["red", "blue", "orange", "purple", "cyan", "green", "lime", "#bb86fc"]
    markers = [".", "v", "*", "o", "s", "d", "P", "p"]
    # keys = sorted(res.keys())
    # cmap = ["red", "#ef9a9a", "#e57373", "#ef5350", "#f44336", "#ba68c8", "#9c27b0", "#7cb342"]
    if "mnli" in filename:
        plt.axhline(y=33.33, xmin=0, xmax=4, ls="--", color="pink", label="random")
        plt.axhline(y=84.65, xmin=0, xmax=4, ls="--", color="brown", label="0% mask")
    elif "yelp" in filename:
        plt.axhline(y=50.00, xmin=0, xmax=4, ls="--", color="pink", label="random")
        plt.axhline(y=97.42, xmin=0, xmax=4, ls="--", color="brown", label="0% mask")
    # keys = ["svs-25", "kernelshap-25", "kernelshap-200", "kernelshap-2000", "kernelshap-8000", "lime-25", "lime-200", "AmortizedModel"]
    keys = ["svs-25", "kernelshap-25", "kernelshap-200", "kernelshap-2000", "kernelshap-8000", "AmortizedModel"]
    # keys = ["svs-25", "kernelshap-25", "kernelshap-200", "kernelshap-2000", "kernelshap-8000"]
    for i, key in enumerate(keys):
        print(key)
        _label = key if "kernelshap" not in key else key.replace("kernelshap", "ks")
        _label = "Our Model" if "Amortized" in _label else _label
        # plt.plot(range(len(xs)), res[key][:len(xs)], label=key.lower(), color=cmap[i], marker=markers[i])
        plt.errorbar(range(len(xs)), res[key + " (mean)"][:len(xs)], yerr=res[key + " (std)"][: len(xs)], color=cmap[i], capthick=3, ecolor='black', capsize=5, marker=markers[i], label=_label)
    # plt.plot(range(len(xs)), [33.33, ] * len(xs), color='pink', ls="--", label="random")
    # plt.plot(range(len(xs)), [84.65, ] * len(xs), color='brown', ls='--', label='0% mask')
    plt.xticks(range(len(xs)), xs)
    # get handles
    handles, labels = plt.gca().get_legend_handles_labels()
    # remove the errorbars
    # handles = [h[0] for h in handles]
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    # ax = plt.gca()
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                  box.width, box.height * 0.9])
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 1, box.height])

    # Put a legend below current axis
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #           fancybox=True, shadow=True, ncol=5)
    # plt.legend(handles, labels, loc="lower left")
    # plt.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=4)
    plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel("Accuracy", fontsize=22)
    plt.xlabel("Top K% Mask", fontsize=22)
    # target_fp = os.path.join(target_dir, "feature_selection_mnli_wo_amortized.pdf")
    # target_fp = os.path.join(target_dir, "feature_selection_mnli.pdf")
    # target_fp = os.path.join(target_dir, "feature_selection_yelp_wo_amortized_w_errorbar.pdf")
    if "yelp" in filename:
        target_fp = os.path.join(target_dir, "feature_selection_yelp_w_amortized_w_errorbar.pdf")
    elif "mnli" in filename:
        target_fp = os.path.join(target_dir, "feature_selection_mnli_w_amortized_w_errorbar.pdf")
    plt.tight_layout()

    plt.savefig(target_fp)
    plt.show()



        # print(line)
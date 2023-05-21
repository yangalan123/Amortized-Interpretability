import os
import numpy as np
from collections import Counter

import thermostat
from tqdm import trange
from scipy.stats import spearmanr

if __name__ == '__main__':
    data_cache_dir = "./datasets/yelp_polarity_vis"
    data1 = thermostat.load("yelp_polarity_amortized_model_output", data_cache_dir=data_cache_dir)
    data2 = thermostat.load("yelp_polarity_amortized_model_reference", data_cache_dir=data_cache_dir)
    tag1 = "Amortized Model"
    tag2 = "SVS-25"
    output_dir = "visualization/error_analysis_htmls"
    os.makedirs(output_dir, exist_ok=True)
    vocabulary = Counter()
    bad_vocab = Counter()
    vocab_to_spearman = dict()
    for i in trange(min(len(data1), len(data2))):
        instance1 = data1[i]
        words = [x[0] for x in instance1.explanation]
        for word in words:
            vocabulary[word] += 1

    for i in trange(min(len(data1), len(data2))):
        instance1 = data1[i]
        instance2 = data2[i]
        attr1 = [x[1] for x in instance1.explanation]
        attr2 = [x[1] for x in instance2.explanation]
        words = [x[0] for x in instance1.explanation]
        word_set = set(words)
        assert len(attr1) == len(attr2)
        s, p = spearmanr(attr1, attr2)
        for word in word_set:
            if word not in vocab_to_spearman:
                vocab_to_spearman[word] = []
            vocab_to_spearman[word].append(s)
        if s < 0.5:
            for word in word_set:
                bad_vocab[word] += 1
    word_freqs = vocabulary.most_common()
    _counter = 0
    word_counts = sum([x[1] for x in word_freqs])
    for word in reversed(word_freqs):
        if word[1] <= 5:
            continue
        if _counter >= 20:
            break
        print(word, np.mean(vocab_to_spearman[word[0]]))
        _counter += 1

    for word in bad_vocab.most_common()[:30]:
        print(word[0], word[1], word[1]/min(len(data1), len(data2)))

        # hm1 = instance1.heatmap
        # html1 = hm1.render()
        # hm2 = instance2.heatmap
        # html2 = hm2.render()
        # f1 = open(os.path.join(output_dir, f"output_{i}.html"), "w", encoding='utf-8')
        # # f1.write('<p style="font-size: 1.5em; ">Seed = 1</p>\n')
        # f1.write(f'<p style="font-size: 1.5em; ">spearman: {s:.2f} ({p:.2f})</p>\n')
        # f1.write(f'<p style="font-size: 1.5em; ">{tag1}</p>\n')
        # f1.write(html1 + "\n")
        # f1.write(f'<p style="font-size: 1.5em; ">{tag2}</p>\n')
        # f1.write(html2)
        # f1.close()

    # print(html1)
    # html1 = instance1.render()
    # print(html1)
    # html2 = instance2.render()


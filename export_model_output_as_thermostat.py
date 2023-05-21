import torch
import copy
import random
import argparse
import json
import numpy as np
from torch import nn, optim
import loguru
from tqdm import tqdm
from scipy.stats import spearmanr, kendalltau
from amortized_model import AmortizedModel
from create_dataset import (
    output_dir as dataset_dir,
    model_cache_dir
)
import os
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import DataCollatorForTokenClassification, AutoModelForSequenceClassification, PreTrainedTokenizer, \
    AutoTokenizer
from config import Args, GetParser
from utils import collate_fn, get_zero_baselines
from metrics import get_eraser_metrics


# example_output = '{"dataset": {"batch_size": 1, "columns": ["input_ids", "attention_mask", "special_tokens_mask", "token_type_ids", "labels"], "end": 3600, "name": "yelp_polarity", "root_dir": "./experiments/thermostat/datasets", "split": "test", "label_names": ["1", "2"], "version": "1.0.0"}, "model": {"mode_load": "hf", "name": "textattack/bert-base-uncased-yelp-polarity", "path_model": null, "tokenization": {"max_length": 512, "padding": "max_length", "return_tensors": "np", "special_tokens_mask": true, "truncation": true}, "tokenizer": "PreTrainedTokenizerFast(name_or_path='textattack/bert-base-uncased-yelp-polarity', vocab_size=30522, model_max_len=512, is_fast=True, padding_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})"}, "explainer": {"internal_batch_size": 1, "n_samples": 25, "name": "KernelShap"}, "batch": 0, "instance": 0, "index_running": 0, "input_ids": [101, 10043, 2000, 2060, 4391, 1010, 1045, 2031, 5717, 10821, 2055, 1996, 2326, 2030, 1996, 7597, 1012, 1045, 2031, 2042, 2893, 12824, 2326, 2182, 2005, 1996, 2627, 1019, 2086, 2085, 1010, 1998, 4102, 2000, 2026, 3325, 2007, 3182, 2066, 27233, 3337, 1010, 2122, 4364, 2024, 5281, 1998, 2113, 2054, 2027, 1005, 2128, 2725, 1012, 1032, 6583, 4877, 2080, 1010, 2023, 2003, 2028, 2173, 2008, 1045, 2079, 2025, 2514, 2066, 1045, 2572, 2108, 2579, 5056, 1997, 1010, 2074, 2138, 1997, 2026, 5907, 1012, 2060, 8285, 9760, 2031, 2042, 12536, 2005, 3007, 6026, 2006, 2026, 18173, 1997, 3765, 1010, 1998, 2031, 8631, 2026, 2924, 4070, 4318, 1012, 2021, 2182, 1010, 2026, 2326, 1998, 2346, 6325, 2038, 2035, 2042, 2092, 4541, 1011, 1998, 2292, 2039, 2000, 2033, 2000, 5630, 1012, 1032, 16660, 2094, 2027, 2074, 10601, 1996, 3403, 2282, 1012, 2009, 3504, 1037, 2843, 2488, 2084, 2009, 2106, 1999, 3025, 2086, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "label": 1, "attributions": [-0.15166577696800232, 0.009686493314802647, 0.022048579528927803, -0.010080059990286827, 0.012372241355478764, -0.014180195517838001, 0.022048499435186386, 0.016043271869421005, 0.022048546001315117, -0.00825815461575985, -0.01712176762521267, 0.012372172437608242, -0.007291753310710192, 0.01604328118264675, 0.018203958868980408, -0.025593385100364685, -0.010080037638545036, -0.0021892308723181486, 0.012372259981930256, 0.012372216209769249, 0.023971816524863243, 0.016043292358517647, 0.031862590461969376, -0.028381315991282463, 0.003578625852242112, 0.012372217141091824, -0.019756361842155457, 0.002530973171815276, 0.03623899817466736, -0.010080034844577312, 0.012372217141091824, -0.014180171303451061, 0.009154818020761013, 0.003578625852242112, -0.007291719317436218, 0.012372217141091824, 0.008152471855282784, 0.016043270006775856, 0.0010390281677246094, -0.017149394378066063, 0.0023846065159887075, -0.004741252399981022, 0.012198690325021744, -0.011731443926692009, -0.0012864458840340376, -0.0015238537453114986, 0.017966555431485176, 0.003578625852242112, -0.006065146531909704, 0.016748590394854546, 0.008152471855282784, 0.0023846065159887075, -0.002189238090068102, 0.01796814799308777, 0.003578625852242112, 0.012372217141091824, 0.016043270006775856, 0.014295504428446293, -0.10412800312042236, 0.016043270006775856, -0.0015238537453114986, 0.03623899817466736, 0.013531191274523735, -0.007291719317436218, 0.02204854227602482, 0.016043270006775856, 0.02204854227602482, -0.007291719317436218, -0.004587118048220873, -0.010080034844577312, 0.003578625852242112, 0.035075053572654724, -0.038211312144994736, 0.009154818020761013, 0.003578625852242112, 0.008152471855282784, 0.009686610661447048, 0.012372217141091824, 0.004481421783566475, 0.016043270006775856, 0.016043270006775856, 0.012372217141091824, -0.0012864458840340376, 0.012198690325021744, 0.016748590394854546, 0.016748590394854546, 0.02204854227602482, 0.012372217141091824, -0.002189238090068102, -0.012710582464933395, 0.003578625852242112, 0.01545447576791048, -0.002189238090068102, -0.029719743877649307, 0.012372217141091824, 0.02204854227602482, 0.019564373418688774, -0.002189238090068102, 0.015076815150678158, 0.012904006987810135, 0.016043270006775856, 0.016748590394854546, 0.03186262771487236, 0.003854867070913315, 0.003578625852242112, 0.012372217141091824, 0.018671875819563866, 0.03623899817466736, 0.02204854227602482, 0.03186262771487236, 0.02204854227602482, -0.004278442356735468, -0.004741252399981022, -0.008890857920050621, 0.003578625852242112, 0.003578625852242112, 0.017966555431485176, 0.03186262771487236, -0.010080034844577312, -0.019756361842155457, 0.023971829563379288, 0.012372217141091824, 0.003578625852242112, 0.0040879095904529095, 0.013531191274523735, 0.018203964456915855, 0.009154818020761013, -0.0015238537453114986, 0.003578625852242112, -0.01281975582242012, -0.004741252399981022, 0.016043270006775856, -0.014180171303451061, 0.003578625852242112, 0.03237045556306839, -0.015182516537606716, 0.016043270006775856, 0.02204854227602482, -0.010080034844577312, -0.0012864458840340376, 0.012372217141091824, 0.02204854227602482, 0.013531191274523735, 0.003578625852242112, 0.014935438521206379, 0.018671875819563866, 0.013531191274523735, 0.03623899817466736, 0.016748590394854546, 0.020859969779849052, 0.012372217141091824, 0.03186262771487236, 0.02204854227602482, -0.008412305265665054, 0.03186262771487236, 0.02212294563651085, -0.002189238090068102, 0.018671875819563866, 0.014935438521206379, -0.0015238537453114986, 0.003578625852242112, 0.003578625852242112, 0.003578625852242112, 0.03237045556306839, 0.016043270006775856, -0.004503846634179354, 0.03186262771487236, 0.004481421783566475, 0.016748255118727684, 0.016043270006775856, 0.02204854227602482, 0.012198690325021744, 0.015076815150678158, -0.032937146723270416, -0.0015238537453114986, 0.016043270006775856, 0.02204854227602482, 0.003578625852242112, -0.0015238537453114986, -0.0015238537453114986, -0.017121760174632072, 0.003578625852242112, 0.003578625852242112, 0.002530973171815276, 0.03186262771487236, -0.03287728875875473, 0.0023846065159887075, 0.03623899817466736, 0.012372217141091824, -0.0012864458840340376, 0.008152471855282784, 0.012372217141091824, -0.04084590822458267, 0.02204854227602482, -0.008412305265665054, 0.008152471855282784, -0.045401785522699356, 0.007830922491848469, 0.003578625852242112, 0.0031351549550890923, 0.012372217141091824, 0.009662647731602192, -0.024617265909910202, -0.028381265699863434, 0.009154818020761013, 0.02204854227602482, 0.003578625852242112, 0.016043270006775856, 0.012372217141091824, 0.016043270006775856, -0.04084590822458267, -0.010080034844577312, 0.003854867070913315, 0.016043270006775856, 0.013531191274523735, 0.003176276572048664, 0.02204854227602482, 0.019055737182497978, -0.0053684343583881855, -0.015182516537606716, -0.007291719317436218, 0.02656267210841179, -0.008475658483803272, 0.02204854227602482, -0.00043386375182308257, 0.012372217141091824, 0.016748590394854546, 0.004481421783566475, 0.003854867070913315, 0.008152471855282784, 0.017966555431485176, 0.001068122568540275, 0.003578625852242112, -0.010509118437767029, 1.0285876669513527e-05, 0.03623899817466736, -0.011833010241389275, -4.1121522372122854e-05, 0.016748590394854546, 0.002530973171815276, 0.003578625852242112, 0.012372217141091824, 0.022021381184458733, 0.02204854227602482, -0.010509118437767029, 0.023971829563379288, 0.02915305830538273, 0.009154818020761013, 0.03623899817466736, 0.02204854227602482, -0.06154629588127136, 0.03186262771487236, 0.004481421783566475, 0.002530973171815276, -0.004035932011902332, 0.02204854227602482, 0.016043270006775856, -0.0015238537453114986, 0.016043270006775856, -0.0006864278111606836, -0.009803798981010914, -0.00388179998844862, 0.03186262771487236, 0.003578625852242112, 0.012372217141091824, 0.023345274850726128, 0.012372217141091824, -0.02237599529325962, 0.0084234569221735, -0.018704941496253014, 0.016748590394854546, -0.01588624157011509, 0.008152471855282784, 0.016748590394854546, 0.009662647731602192, 0.023971829563379288, 0.016748590394854546, 0.02204854227602482, -0.01839991845190525, -0.010080034844577312, -0.0015238537453114986, 0.016748590394854546, 0.012198690325021744, 0.016748590394854546, 0.016043270006775856, 0.012372217141091824, 0.012372217141091824, 0.02204854227602482, 0.02204854227602482, 0.008152471855282784, 0.02204854227602482, 0.016043270006775856, 0.02204854227602482, -0.004741252399981022, 0.03623899817466736, -0.0015222595538944006, 7.39634851925075e-05, 0.02204854227602482, 0.02204854227602482, 0.016043270006775856, 0.017966555431485176, 0.03623899817466736, 0.001264021499082446, -0.0015238537453114986, -0.0015238537453114986, 0.02204854227602482, 0.012372217141091824, -0.002189238090068102, 0.02204854227602482, -0.0012864458840340376, -0.00043386375182308257, 0.016748590394854546, 0.02204854227602482, -0.002189238090068102, 0.003578625852242112, 0.016748590394854546, 0.016748590394854546, -0.0012864458840340376, 0.01796814799308777, 0.009154818020761013, 0.02204854227602482, 0.02204854227602482, 0.016043270006775856, 0.014824969694018364, -0.007291719317436218, 0.016748590394854546, 0.008527638390660286, -0.019756361842155457, -0.010509118437767029, 0.002530973171815276, 0.004481421783566475, -0.01945282518863678, -0.0012864458840340376, 0.0023846065159887075, 0.012372217141091824, -0.002189238090068102, 0.016043270006775856, -0.002189238090068102, -0.012561912648379803, 0.016043270006775856, 0.02656267210841179, -0.02269398421049118, -0.02237599529325962, 0.016043270006775856, -0.007291719317436218, 0.012198690325021744, 0.009154818020761013, 0.016043270006775856, -0.0015238537453114986, -0.002189238090068102, 0.016748590394854546, 0.03237045556306839, -0.03827592730522156, 0.0023846065159887075, -0.010080034844577312, 0.012372217141091824, 0.03623899817466736, 0.02204854227602482, 0.003854867070913315, 0.017966555431485176, 0.013531191274523735, 0.012372217141091824, 0.016043270006775856, 0.0023846065159887075, 0.009154818020761013, -0.024617265909910202, 0.03186262771487236, 0.03623899817466736, -0.002189238090068102, -0.010509118437767029, -0.010509118437767029, -0.009803798981010914, 0.02656267210841179, 0.02204854227602482, 0.02204854227602482, -0.012477915734052658, -0.010080034844577312, 0.03186262771487236, 0.02204854227602482, -0.007291719317436218, -0.025430934503674507, 0.003578625852242112, 0.03186262771487236, 0.012372217141091824, 0.0006368431495502591, 0.009154818020761013, 0.016043270006775856, -0.029719743877649307, -0.007291719317436218, 0.016043270006775856, -0.0015238537453114986, -0.028381265699863434, 0.008152471855282784, -0.0012864458840340376, -0.0031351549550890923, 0.016043270006775856, 0.0023846065159887075, 0.016748590394854546, -0.0012864458840340376, 0.012372217141091824, 0.03186262771487236, -0.017121760174632072, 0.016748590394854546, -0.02237599529325962, -0.004503846634179354, 0.02656267210841179, 0.03186262771487236, 0.012198690325021744, 0.008152471855282784, 0.016043270006775856, 0.014295504428446293, 0.016043270006775856, 0.007932491600513458, -0.0015238537453114986, 0.015584642998874187, -0.003452391130849719, 0.03186262771487236, -0.00043386375182308257, 0.003854867070913315, 0.012372217141091824, 0.008152471855282784, 0.004481421783566475, 0.008152471855282784, -0.00043386375182308257, 0.02204854227602482, -0.0012864458840340376, -0.0012864458840340376, -0.002189238090068102, -0.012344200164079666, 0.017966555431485176, 0.01947673410177231, 0.016043270006775856, -0.016788210719823837, 0.016043270006775856, 0.009154818020761013, 0.02204854227602482, 0.02656267210841179, 0.0023846065159887075, -0.010509118437767029, 0.03186262771487236, -0.004278442356735468, 0.016748590394854546, 0.008338750340044498, 0.009154818020761013, 0.02204854227602482, -0.0015222595538944006, 0.02915305830538273, 0.02204854227602482, 0.009296262636780739, -0.010080034844577312, 0.012372217141091824, -0.022973762825131416, -0.002189238090068102, 0.016043270006775856, 0.02204854227602482, 0.001264021499082446, -0.038211312144994736, -0.007291719317436218, 0.005778150167316198, 0.008152471855282784, 0.003578625852242112, 0.012198690325021744, -0.029719743877649307, 0.016043270006775856, -0.007291719317436218, 0.02204854227602482, -0.015182516537606716, 0.016043270006775856, -0.004035932011902332, 0.012372217141091824, 0.013531191274523735, -0.0012864458840340376, 0.016748590394854546, 0.012372217141091824, 0.02204854227602482, 0.017966555431485176, 0.008152471855282784, 0.03186262771487236, 0.03186262771487236, -0.010080034844577312, -0.0002659528108779341, 0.02204854227602482, 0.03186262771487236, 0.016043270006775856, 0.009662647731602192, 0.003578625852242112, 0.008527638390660286, 0.004481421783566475, 0.001304063480347395, 0.016748590394854546, -0.002189238090068102, 0.007729377131909132, -0.020859969779849052, -0.0012864458840340376, 0.001264021499082446, 0.008440319448709488, 0.012372217141091824, 1.4129986573903504e-16, 0.03186262771487236, 0.016043270006775856, 0.002530973171815276, -7.236459887000114e-17, -0.010509118437767029, -0.02237599529325962, 0.02204854227602482, -0.007291719317436218, 0.03186262771487236, 0.003578625852242112, -0.0012864458840340376, -0.0015238537453114986, 0.008440319448709488, 0.023971829563379288], "predictions": [-4.52530574798584, 4.283736705780029]}'
def get_example_output():
    # Please change the filepath to your own path
    filepath = "/path/to/thermostat/experiments/thermostat/yelp_polarity/bert/kernelshap-3600/seed_1/[date].KernelShap.jsonl"
    with open(filepath, "r", encoding='utf-8') as f_in:
        for line in f_in:
            obj = json.loads(line.strip())
            return obj

def running_step(dataloader, model, K, optimizer=None, is_train=False, save=False, args=None):
    def get_top_k(_output):
        _rank_output = [(x, i) for i, x in enumerate(_output)]
        _rank_output.sort(key=lambda x: x[0], reverse=True)
        _rank_output = [x[1] for x in _rank_output][:K]
        return _rank_output

    # def dropout(_input):
    #     _rand = torch.rand_like(_input.float())
    #     _mask = _rand >= 0.5
    #     return _mask.long() * _input

    all_loss = 0
    all_outputs = []
    all_aux_outputs = []
    all_refs = []
    all_attn = []
    all_ins = []
    count_elements = 0
    spearman = []
    kendals = []
    intersection = []
    # dropout = nn.Dropout(inplace=True)
    desc = "testing"
    if is_train:
        assert optimizer is not None
        optimizer.zero_grad()
        desc = 'training'
    for batch in tqdm(dataloader, desc=desc):
        # if is_train:
        # # add masking like FASTSHAP
        #     dropout(batch["attention_mask"])
        if hasattr(model, "multitask") and model.multitask:
            main_output, main_loss, aux_output, aux_loss = model(batch)
            output = main_output
            loss = main_loss
            all_aux_outputs.extend((aux_output.argmax(dim=-1) == batch["ft_label"].cuda()).detach().cpu().tolist())
        else:
            output, loss = model(batch)
        if is_train:
            if not hasattr(args, "discrete") or not args.discrete:
                if len(all_aux_outputs) == 0:
                    loss = loss
                else:
                    loss = torch.sqrt(loss) + aux_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # recording purposes
        all_loss += loss.item()
        # # do not count [CLS]
        # batch["attention_mask"][:, 0] = 0

        attn_mask = batch["attention_mask"].cuda()
        batch["output"] = batch["output"].cuda()
        for _ind in range(len(output)):
            _output = output[_ind][attn_mask[_ind] > 0].detach().cpu().numpy()
            _ref = batch["output"][_ind][attn_mask[_ind] > 0].detach().cpu().numpy()
            all_attn.append(attn_mask.detach().cpu().numpy())
            all_ins.append(batch['input_ids'].detach().cpu().numpy())
            _rank_output = get_top_k(_output)
            _rank_ref = get_top_k(_ref)
            intersect_num = len(set(_rank_ref) & set(_rank_output))
            _spearman, p_val = spearmanr(_output, _ref, axis=0)
            _kendal, kp_val = kendalltau(_output, _ref)
            spearman.append(_spearman)
            kendals.append(_kendal)
            intersection.append(intersect_num)
            all_outputs.append(_output)
            all_refs.append(_ref)
            if not is_train:
                all_aux_outputs.append(output[_ind].detach().cpu().numpy())

        count_elements += batch["attention_mask"].sum().item()
    if save and args is not None:
        torch.save([all_outputs, all_refs, all_attn, all_ins],
                   os.path.join(os.path.dirname(args.save_path),
                                os.path.basename(args.save_path).strip(".pt"),
                                "test_outputs_output_verified.pkl")
                   )
    return all_loss, all_outputs, all_refs, count_elements, spearman, kendals, intersection, all_aux_outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Amortized Model Arguments Parser")
    parser = GetParser(parser)
    global_args = parser.parse_args()
    logger = loguru.logger
    # assert global_args.train_bsz == 1 and global_args.test_bsz == 1, "currently only support batch_size == 1"

    torch.manual_seed(global_args.seed)
    random.seed(global_args.seed)
    target_model = AutoModelForSequenceClassification.from_pretrained(global_args.target_model).cuda()
    tokenizer = AutoTokenizer.from_pretrained(global_args.target_model)
    if global_args.target_model == "textattack/bert-base-uncased-MNLI":
        label_mapping_dict = {
            0: 2,
            1: 0,
            2: 1
        }
        label_mapping = lambda x: label_mapping_dict[x]
    else:
        label_mapping = None
    K = global_args.topk
    alL_train_datasets = dict()
    all_valid_datasets = dict()
    all_test_datasets = dict()
    explainers = global_args.explainer
    if "," in explainers:
        explainers = explainers.split(",")
    else:
        explainers = [explainers, ]
    for explainer in explainers:
        train_dataset, valid_dataset, test_dataset = torch.load(os.path.join(dataset_dir, f"data_{explainer}.pkl"))
        train_dataset, valid_dataset, test_dataset = Dataset.from_dict(train_dataset), Dataset.from_dict(
            valid_dataset), Dataset.from_dict(test_dataset)
        alL_train_datasets[explainer] = train_dataset
        all_valid_datasets[explainer] = valid_dataset
        all_test_datasets[explainer] = test_dataset
    for proportion in [1.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
        for explainer in explainers:
            args = Args(seed=global_args.seed, explainer=explainer, proportion=str(proportion),
                        epochs=global_args.epoch,
                        batch_size=global_args.train_bsz, normalization=global_args.normalization,
                        task_name=global_args.task,
                        discretization=global_args.discrete,
                        lr=global_args.lr, neuralsort=global_args.neuralsort,
                        multitask=True if hasattr(global_args, "multitask") and global_args.multitask else False,
                        suf_reg=global_args.suf_reg if hasattr(global_args, "suf_reg") and global_args.suf_reg else False,
                        storage_root=global_args.storage_root,
                        )
            train_dataset, valid_dataset, test_dataset = alL_train_datasets[explainer], all_valid_datasets[explainer], \
                                                         all_test_datasets[explainer]
            if proportion < 1:
                id_fn = os.path.join(os.path.dirname(args.save_path),
                                     os.path.basename(args.save_path).strip(".pt"),
                                     "training_ids.pkl")
                if not os.path.exists(id_fn):
                    sample_ids = random.sample(range(len(train_dataset)), int(proportion * len(train_dataset)))
                    os.makedirs(
                        os.path.join(os.path.dirname(args.save_path),
                                     os.path.basename(args.save_path).strip(".pt"),
                                     ),
                        exist_ok=True
                    )
                    torch.save(sample_ids,
                               os.path.join(os.path.dirname(args.save_path),
                                            os.path.basename(args.save_path).strip(".pt"),
                                            "training_ids.pkl")
                               )
                else:
                    sample_ids = torch.load(id_fn)
                train_dataset = train_dataset.select(sample_ids)
            train_dataset, valid_dataset, test_dataset = get_zero_baselines([train_dataset, valid_dataset, test_dataset], target_model, tokenizer, args)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                          collate_fn=collate_fn)
            valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
            if args.fastshap or args.suf_reg:
                model = AmortizedModel(global_args.amortized_model, cache_dir=model_cache_dir, args=args,
                                       target_model=target_model, tokenizer=tokenizer).cuda()
            else:
                model = AmortizedModel(global_args.amortized_model, cache_dir=model_cache_dir, args=args).cuda()

            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            # handler_id = logger.add(os.path.join(os.path.dirname(args.save_path), "log_{time}.txt"))
            log_dir = os.path.join(os.path.dirname(args.save_path), os.path.basename(args.save_path).strip(".pt"))
            handler_id = logger.add(os.path.join(log_dir, "output_verify_no_pad_log_{time}.txt"))
            logger.info(json.dumps(vars(args), indent=4))
            try:
                model = torch.load(args.save_path)
            except:
                os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
                best_valid_spearman = -999999
                for epoch_i in range(args.epochs):
                    training_loss, all_outputs, all_refs, count_elements, spearman, kendals, intersection, all_aux_output = running_step(
                        train_dataloader, model, K, optimizer, is_train=True)
                    logger.info(f"training loss at epoch {epoch_i}: {training_loss / len(train_dataloader)}")
                    logger.info(f"training spearman (micro-avg): {np.mean(spearman)}")
                    logger.info(f"training top-{K} intersection: {np.mean(intersection)}")

                    all_outputs = np.concatenate(all_outputs)
                    all_refs = np.concatenate(all_refs)
                    logger.info(f"training spearman: {spearmanr(all_outputs, all_refs)}")
                    logger.info(f"training kendaltau: {kendalltau(all_outputs, all_refs)}")
                    if len(all_aux_output) > 0:
                        logger.info(f"training aux acc: {np.mean(all_aux_output)}")

                    if (epoch_i) % args.validation_period == 0:
                        with torch.no_grad():
                            valid_loss, valid_all_outputs, valid_all_refs, valid_count_elements, valid_spearman, valid_kendals, valid_intersection, all_valid_aux_output = running_step(
                                valid_dataloader, model, K, optimizer, is_train=False)
                            logger.info(f"Validating at epoch-{epoch_i}")
                            valid_all_outputs = np.concatenate(valid_all_outputs)
                            valid_all_refs = np.concatenate(valid_all_refs)
                            valid_macro_spearman = spearmanr(valid_all_outputs, valid_all_refs)
                            valid_macro_kendal = kendalltau(valid_all_outputs, valid_all_refs)
                            logger.info(f"validation spearman: {valid_macro_spearman}")
                            logger.info(f"validation kendaltau: {valid_macro_kendal}")
                            micro_spearman = np.mean(valid_spearman)
                            micro_kendal = np.mean(valid_kendals)
                            logger.info(f"validation micro spearman: {micro_spearman}")
                            logger.info(f"validation micro kendal: {micro_kendal}")
                            if len(all_valid_aux_output) > 0:
                                logger.info(f"validation aux acc: {np.mean(all_valid_aux_output)}")
                            if valid_macro_spearman.correlation > best_valid_spearman:
                                best_valid_spearman = valid_macro_spearman.correlation
                                logger.info(
                                    f"best validation spearman at {epoch_i}: {valid_macro_spearman.correlation}, save checkpoint here")
                                torch.save(model, args.save_path)

            with torch.no_grad():
                model = model.eval()
                for test_explainer in explainers:
                    handler_id_test = logger.add(
                        os.path.join(os.path.dirname(args.save_path), os.path.basename(args.save_path).strip(".pt"),
                                     f"test_log_no_pad_{test_explainer}_output_verify.txt"))
                    test_dataset = all_test_datasets[test_explainer]
                    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                                 collate_fn=collate_fn)
                    logger.info(f"doing testing for {test_explainer}")
                    test_loss, all_outputs, all_refs, count_elements, spearman, kendals, intersection, all_test_aux_output = running_step(
                        test_dataloader, model, K, optimizer, is_train=False, save=True, args=args)

                    logger.info(f"testing spearman (micro-avg): {np.mean(spearman)}")
                    logger.info(f"testing kendal (micro-avg): {np.mean(kendals)}")
                    logger.info(f"testing top-{K} intersection: {np.mean(intersection)}")
                    logger.info(f"testing RMSE: {np.sqrt(test_loss / count_elements)}")
                    backup_all_outputs = copy.deepcopy(all_outputs)
                    all_outputs = np.concatenate(all_outputs)
                    all_refs = np.concatenate(all_refs)
                    logger.info(f"testing spearman: {spearmanr(all_outputs, all_refs)}")
                    logger.info(f"testing kendaltau: {kendalltau(all_outputs, all_refs)}")
                    if len(all_test_aux_output) > 0:
                        logger.info(f"testing aux acc: {np.mean(all_test_aux_output)}")
                    example = get_example_output()
                    example["end"] = len(test_dataloader)
                    example['explainer']['name'] = "AmortizedModelBERT"
                    counter_id = 0
                    all_examples_out = list()
                    all_examples_out_ref = list()
                    for batch in test_dataloader:
                        input_ids = batch['input_ids']
                        attn_mask = batch['attention_mask']
                        labels = batch['ft_label']
                        #print(input_ids.shape)
                        #print(attn_mask.shape)
                        assert len(input_ids[0]) == len(attn_mask[0])
                        # assert len(input_ids[0]) == len(all_outputs[counter_id])
                        for batch_i in range(len(input_ids)):
                            assert len(input_ids[batch_i][attn_mask[batch_i] > 0]) == len(backup_all_outputs[counter_id])
                            assert len(all_test_aux_output[counter_id]) == len(input_ids[batch_i])
                            new_example = copy.deepcopy(example)
                            new_example['batch'] = counter_id
                            new_example['index_running'] = counter_id
                            new_example['input_ids'] = batch['input_ids'][batch_i].tolist()
                            # new_example['attributions'] = list(all_outputs[counter_id] + [1e-6, ]* len())
                            new_example['attributions'] = [float(x) for x in list(all_test_aux_output[counter_id])]
                            new_example['label'] = int(labels[batch_i])
                            if "prediction_dist" in batch:
                                new_example["predictions"] = [float(x) for x in batch['prediction_dist'][batch_i]]
                            all_examples_out.append(new_example)
                            new_example_ref = copy.deepcopy(new_example)
                            new_example_ref["attributions"] = batch["output"][batch_i].cpu().tolist()
                            assert len(new_example_ref['attributions']) == len(new_example['attributions'])
                            all_examples_out_ref.append(new_example_ref)



                            counter_id += 1


                    # change it to the path of your thermostat
                    example_out_dir = f'/path/to/thermostat/experiments/thermostat/yelp_polarity/bert/AmortizedModel/seed_{args.seed}'
                    os.makedirs(example_out_dir, exist_ok=True)
                    with open(os.path.join(example_out_dir, "output.jsonl"), "w", encoding='utf-8') as f_out:
                        for line in all_examples_out:
                            for key in line.keys():
                                if torch.is_tensor(line[key]):
                                    line[key] = [float(x) for x in line[key].tolist()]
                            f_out.write(json.dumps(line) + "\n")
                    with open(os.path.join(example_out_dir, "ref.jsonl"), "w", encoding='utf-8') as f_out:
                        for line in all_examples_out_ref:
                            for key in line.keys():
                                if torch.is_tensor(line[key]):
                                    line[key] = [float(x) for x in line[key].tolist()]
                            f_out.write(json.dumps(line) + "\n")


                    try:
                        stat_dict = torch.load(os.path.join(log_dir, f"eraser_stat_dict_{test_explainer}.pt"))
                    except:
                        test_dataloader = DataLoader(test_dataset, batch_size=1,
                                                     collate_fn=collate_fn)
                        stat_dict = get_eraser_metrics(test_dataloader, target_model, amortized_model=model,
                                                       tokenizer=tokenizer, label_mapping=label_mapping)
                        torch.save(stat_dict, os.path.join(log_dir, f"eraser_stat_dict_{test_explainer}.pt"))
                    logger.info("eraser_metrics")
                    for k in stat_dict:
                        for metric in stat_dict[k]:
                            logger.info(
                                f"{k}-{metric}: {np.mean(stat_dict[k][metric]).item()} ({np.std(stat_dict[k][metric]).item()})")
                    logger.remove(handler_id_test)
            #
            logger.remove(handler_id)

import torch
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
    ks_meta_spearman = []
    ks_meta_spearman_1 = []
    ks_meta_spearman_2 = []
    ks_meta_spearman_3 = []
    ks_meta_spearman_5 = []
    ks_meta_spearman_use_imp = []
    ks_meta_spearman_use_imp_temper = []
    ks_meta_spearman_use_init_1 = []
    ks_meta_spearman_use_init_2 = []
    ks_meta_spearman_use_init_3 = []
    ks_meta_spearman_use_init_5 = []
    kendals = []
    intersection = []
    # dropout = nn.Dropout(inplace=True)
    desc = "testing"
    do_ks_meta_eval = True
    if is_train:
        assert optimizer is not None
        optimizer.zero_grad()
        desc = 'training'
    for batch in tqdm(dataloader, desc=desc):
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
        if not is_train and do_ks_meta_eval:
            global target_model
            ks_meta_output = model.svs_compute_meta(batch, 10, "cuda", target_model).cpu()
            ks_meta_output_1 = model.svs_compute_meta(batch, 1, "cuda", target_model).cpu()
            ks_meta_output_2 = model.svs_compute_meta(batch, 2, "cuda", target_model).cpu()
            ks_meta_output_3 = model.svs_compute_meta(batch, 3, "cuda", target_model).cpu()
            ks_meta_output_5 = model.svs_compute_meta(batch, 5, "cuda", target_model).cpu()
            ks_meta_output_use_imp = model.svs_compute_meta(batch, 10, "cuda", target_model, use_imp=True).cpu()
            ks_meta_output_use_imp_temper = model.svs_compute_meta(batch, 1, "cuda", target_model, use_imp=True, inv_temper=0.1).cpu()
            ks_meta_output_use_init_1 = model.svs_compute_meta(batch, 1, "cuda", target_model, use_init=True).cpu()
            ks_meta_output_use_init_2 = model.svs_compute_meta(batch, 2, "cuda", target_model, use_init=True).cpu()
            ks_meta_output_use_init_3 = model.svs_compute_meta(batch, 3, "cuda", target_model, use_init=True).cpu()
            ks_meta_output_use_init_5 = model.svs_compute_meta(batch, 5, "cuda", target_model, use_init=True).cpu()
        else:
            ks_meta_output = None
            ks_meta_output_1 = None
            ks_meta_output_2 = None
            ks_meta_output_3 = None
            ks_meta_output_5 = None
            ks_meta_output_use_imp = None
            ks_meta_output_use_imp_temper = None
            ks_meta_output_use_init_1 = None
            ks_meta_output_use_init_2 = None
            ks_meta_output_use_init_3 = None
            ks_meta_output_use_init_5 = None

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
            if ks_meta_output is not None and _ind < len(ks_meta_output):
                if len(attn_mask[_ind]) == len(ks_meta_output[_ind]):
                   _ks_meta_output = ks_meta_output[_ind][attn_mask[_ind] > 0]
                   _ks_meta_output_1 = ks_meta_output_1[_ind][attn_mask[_ind] > 0]
                   _ks_meta_output_2 = ks_meta_output_2[_ind][attn_mask[_ind] > 0]
                   _ks_meta_output_3 = ks_meta_output_3[_ind][attn_mask[_ind] > 0]
                   _ks_meta_output_5 = ks_meta_output_5[_ind][attn_mask[_ind] > 0]
                   _ks_meta_output_use_imp = ks_meta_output_use_imp[_ind][attn_mask[_ind] > 0]
                   _ks_meta_output_use_imp_temper = ks_meta_output_use_imp_temper[_ind][attn_mask[_ind] > 0]
                   _ks_meta_output_use_init_1 = ks_meta_output_use_init_1[_ind][attn_mask[_ind] > 0]
                   _ks_meta_output_use_init_2 = ks_meta_output_use_init_2[_ind][attn_mask[_ind] > 0]
                   _ks_meta_output_use_init_3 = ks_meta_output_use_init_3[_ind][attn_mask[_ind] > 0]
                   _ks_meta_output_use_init_5 = ks_meta_output_use_init_5[_ind][attn_mask[_ind] > 0]
                else:
                   _ks_meta_output = ks_meta_output[_ind]
                   _ks_meta_output_1 = ks_meta_output_1[_ind]
                   _ks_meta_output_2 = ks_meta_output_2[_ind]
                   _ks_meta_output_3 = ks_meta_output_3[_ind]
                   _ks_meta_output_5 = ks_meta_output_5[_ind]
                   _ks_meta_output_use_imp = ks_meta_output_use_imp[_ind]
                   _ks_meta_output_use_imp_temper = ks_meta_output_use_imp_temper[_ind]
                   _ks_meta_output_use_init_1 = ks_meta_output_use_init_1[_ind]
                   _ks_meta_output_use_init_2 = ks_meta_output_use_init_2[_ind]
                   _ks_meta_output_use_init_3 = ks_meta_output_use_init_3[_ind]
                   _ks_meta_output_use_init_5 = ks_meta_output_use_init_5[_ind]
                _ks_meta_spearman, _ = spearmanr(_ks_meta_output, _ref, axis=0)
                _ks_meta_spearman_1, _ = spearmanr(_ks_meta_output_1, _ref, axis=0)
                _ks_meta_spearman_2, _ = spearmanr(_ks_meta_output_2, _ref, axis=0)
                _ks_meta_spearman_3, _ = spearmanr(_ks_meta_output_3, _ref, axis=0)
                _ks_meta_spearman_5, _ = spearmanr(_ks_meta_output_5, _ref, axis=0)
                _ks_meta_spearman_use_imp, _ = spearmanr(_ks_meta_output_use_imp, _ref, axis=0)
                _ks_meta_spearman_use_imp_temper, _ = spearmanr(_ks_meta_output_use_imp_temper, _ref, axis=0)
                _ks_meta_spearman_use_init_1, _ = spearmanr(_ks_meta_output_use_init_1, _ref, axis=0)
                _ks_meta_spearman_use_init_2, _ = spearmanr(_ks_meta_output_use_init_2, _ref, axis=0)
                _ks_meta_spearman_use_init_3, _ = spearmanr(_ks_meta_output_use_init_3, _ref, axis=0)
                _ks_meta_spearman_use_init_5, _ = spearmanr(_ks_meta_output_use_init_5, _ref, axis=0)
                ks_meta_spearman.append(_ks_meta_spearman)
                ks_meta_spearman_1.append(_ks_meta_spearman_1)
                ks_meta_spearman_2.append(_ks_meta_spearman_2)
                ks_meta_spearman_3.append(_ks_meta_spearman_3)
                ks_meta_spearman_5.append(_ks_meta_spearman_5)
                ks_meta_spearman_use_imp.append(_ks_meta_spearman_use_imp)
                ks_meta_spearman_use_imp_temper.append(_ks_meta_spearman_use_imp_temper)
                ks_meta_spearman_use_init_1.append(_ks_meta_spearman_use_init_1)
                ks_meta_spearman_use_init_2.append(_ks_meta_spearman_use_init_2)
                ks_meta_spearman_use_init_3.append(_ks_meta_spearman_use_init_3)
                ks_meta_spearman_use_init_5.append(_ks_meta_spearman_use_init_5)
                global logger
                if len(ks_meta_spearman) >= 100:
                    do_ks_meta_eval = False
                    logger.info("ks_meta_spearman: {}".format(np.mean(ks_meta_spearman)))
                    logger.info("ks_meta_spearman_1: {}".format(np.mean(ks_meta_spearman_1)))
                    logger.info("ks_meta_spearman_2: {}".format(np.mean(ks_meta_spearman_2)))
                    logger.info("ks_meta_spearman_3: {}".format(np.mean(ks_meta_spearman_3)))
                    logger.info("ks_meta_spearman_5: {}".format(np.mean(ks_meta_spearman_5)))
                    logger.info("ks_meta_spearman_use_imp: {}".format(np.mean(ks_meta_spearman_use_imp)))
                    logger.info("ks_meta_spearman_use_imp_temper_sample_1: {}".format(np.mean(ks_meta_spearman_use_imp_temper)))
                    logger.info("ks_meta_spearman_use_init_1: {}".format(np.mean(ks_meta_spearman_use_init_1)))
                    logger.info("ks_meta_spearman_use_init_2: {}".format(np.mean(ks_meta_spearman_use_init_2)))
                    logger.info("ks_meta_spearman_use_init_3: {}".format(np.mean(ks_meta_spearman_use_init_3)))
                    logger.info("ks_meta_spearman_use_init_5: {}".format(np.mean(ks_meta_spearman_use_init_5)))
            _kendal, kp_val = kendalltau(_output, _ref)
            spearman.append(_spearman)
            kendals.append(_kendal)
            intersection.append(intersect_num)
            all_outputs.append(_output)
            all_refs.append(_ref)

        count_elements += batch["attention_mask"].sum().item()
    if save and args is not None:
        torch.save([all_outputs, all_refs, all_attn, all_ins],
                   os.path.join(os.path.dirname(args.save_path),
                                os.path.basename(args.save_path).strip(".pt"),
                                "test_outputs.pkl")
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
    if "MNLI" in global_args.target_model:
        dataset_dir = "./amortized_dataset/mnli_test"
    if "yelp" in global_args.target_model:
        dataset_dir = "./amortized_dataset/yelp_test"
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
                        storage_root=global_args.storage_root
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
            log_dir = os.path.join(os.path.dirname(args.save_path), os.path.basename(args.save_path).strip(".pt"))
            handler_id = logger.add(os.path.join(log_dir, "log_{time}.txt"))
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
                                     f"test_log_no_pad_{test_explainer}.txt"))
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
                    all_outputs = np.concatenate(all_outputs)
                    all_refs = np.concatenate(all_refs)
                    logger.info(f"testing spearman: {spearmanr(all_outputs, all_refs)}")
                    logger.info(f"testing kendaltau: {kendalltau(all_outputs, all_refs)}")
                    if len(all_test_aux_output) > 0:
                        logger.info(f"testing aux acc: {np.mean(all_test_aux_output)}")

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

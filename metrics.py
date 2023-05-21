import torch
from transformers import PreTrainedTokenizer
from tqdm import tqdm
import random

def get_eraser_metrics(dataloader, target_model, amortized_model, tokenizer: PreTrainedTokenizer, label_mapping=None):
    stat_dict = {}
    target_model = target_model.cuda()
    for batch in tqdm(dataloader, desc="eraser_eval"):
        output, loss = amortized_model(batch)
        # again, assuming bsz == 1
        attn_mask = batch["attention_mask"].clone()
        #attn_mask[:, 0] = 0
        attn_mask = attn_mask.squeeze(0).cuda()
        interpret = batch["output"].squeeze(0).cuda()[attn_mask > 0]
        output = output.squeeze(0)[attn_mask > 0]
        sorted_interpret, sorted_interpret_indices = interpret.sort(descending=True)
        sorted_output, sorted_output_indices = output.sort(descending=True)
        random_order = sorted_output_indices.cpu().tolist()
        random.shuffle(random_order)
        random_order = torch.LongTensor(random_order).to(sorted_output_indices.device)
        target_model_output = target_model(
            input_ids=batch["input_ids"].cuda(),
            attention_mask=batch["attention_mask"].cuda(),
            token_type_ids=batch["token_type_ids"].cuda() if "token_type_ids" in batch else None,
            position_ids=batch["position_ids"].cuda() if "position_ids" in batch else None
        )
        target_logits = target_model_output.logits
        target_model_pred = target_logits.argmax(dim=-1).squeeze(0)
        target_logits_pred = target_logits[:, target_model_pred]
        #for K in [10, 50, 100, 200, 500]:
        for K in [0, 0.01, 0.05, 0.10, 0.20, 0.50]:
            if K not in stat_dict:
                stat_dict[K] = {}
                for model_type in ["interpret", "output", "random"]:
                    stat_dict[K][f"sufficiency_{model_type}"] = []
                    stat_dict[K][f"comprehensiveness_{model_type}"] = []
            input_ids = batch["input_ids"].clone()
            for indices, model_type in zip([sorted_interpret_indices, sorted_output_indices, random_order], ["interpret", "output", "random"]):
                _input_ids = input_ids.clone()
                # compute sufficiency
                #_input_ids[:, indices[: K]] = tokenizer.mask_token_id
                #print(indices)
                #exit()
                _input_ids[:, indices[: int(K * len(indices))]] = tokenizer.mask_token_id
                _target_model_output = target_model(
                    input_ids=_input_ids.cuda(),
                    attention_mask=batch["attention_mask"].cuda(),
                    token_type_ids=batch["token_type_ids"].cuda() if "token_type_ids" in batch else None,
                    position_ids=batch["position_ids"].cuda() if "position_ids" in batch else None
                )
                _logits = _target_model_output.logits
                _label = batch["ft_label"].cpu().item()
                _pred = _logits.argmax(dim=-1).squeeze(0)
                if label_mapping is not None:
                    _pred = label_mapping(_pred.item())
                _pred_logits = _logits[:, _pred]
                delta = target_logits_pred - _pred_logits
                #stat_dict[K][f"comprehensiveness_{model_type}"].append(delta.cpu().item())
                stat_dict[K][f"comprehensiveness_{model_type}"].append(int(_pred == _label))
                _input_ids = input_ids.clone()
                #_input_ids[:, indices[K: ]] = tokenizer.mask_token_id
                _input_ids[:, indices[int(K * len(indices)): ]] = tokenizer.mask_token_id
                _target_model_output = target_model(
                    input_ids=_input_ids.cuda(),
                    attention_mask=batch["attention_mask"].cuda(),
                    token_type_ids=batch["token_type_ids"].cuda() if "token_type_ids" in batch else None,
                    position_ids=batch["position_ids"].cuda() if "position_ids" in batch else None
                )
                _logits = _target_model_output.logits
                _pred = _logits.argmax(dim=-1).squeeze(0)
                if label_mapping is not None:
                    _pred = label_mapping(_pred.item())
                _pred_logits = _logits[:, _pred]
                delta = target_logits_pred - _pred_logits
                #stat_dict[K][f"sufficiency_{model_type}"].append(delta.cpu().item())
                stat_dict[K][f"sufficiency_{model_type}"].append(int(_pred == _label))

    return stat_dict


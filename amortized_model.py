from transformers import AutoModel
from tqdm import tqdm, trange
import math
import torch
from torch import nn
import diffsort
from samplers import ShapleySampler
from sklearn.linear_model import LinearRegression
class AmortizedModel(nn.Module):
    def __init__(self, model_name_or_path, cache_dir, args=None, target_model=None, tokenizer=None):
        super(AmortizedModel, self).__init__()
        self.args = args
        self.model = AutoModel.from_pretrained(model_name_or_path, cache_dir)
        if hasattr(self.args, "extra_feat_dim"):
            self.extra_feat_dim = self.args.extra_feat_dim
        else:
            self.extra_feat_dim = 0
        self.dim = self.model.config.hidden_size + self.extra_feat_dim
        self.output = nn.Linear(self.dim, 1)
        self.discrete = False
        self.multitask = False
        self.remove_columns = ["output", "output_rank", "ft_label", "prediction_dist", "special_tokens_mask", "id", "zero_baseline"]
        if self.args is not None and self.args.discrete:
            self.output = nn.Linear(self.dim, 2)
            self.discrete = True
            self.loss_func = nn.CrossEntropyLoss(reduction="none")
        if self.args is not None and hasattr(self.args, "neuralsort") and self.args.neuralsort:
            self.sortnn = diffsort.DiffSortNet(sorting_network_type=self.args.sort_arch, size=512, device='cuda')
            self.loss_func = torch.nn.BCELoss()
        if self.args is not None and hasattr(self.args, "multitask") and self.args.multitask:
            self.multitask = True
            # imdb is binary classification task
            # [todo]: modify 2 to be some arguments that can specify the number of classification labels
            self.ft_output = nn.Linear(self.model.config.hidden_size, 2)
            self.ft_loss_func = nn.CrossEntropyLoss()
        if self.args is not None and hasattr(self.args, "fastshap") and self.args.fastshap:
            assert self.extra_feat_dim == 0
            self.sampler = ShapleySampler(self.model.config.max_position_embeddings)
            assert target_model is not None
            self.target_model = target_model.eval()
            assert tokenizer is not None
            self.tokenizer = tokenizer
            self.target_label = 0
            self.n_sample = 16
        if self.args is not None and hasattr(self.args, "suf_reg") and self.args.suf_reg:
            assert target_model is not None
            self.target_model = target_model.eval()
            assert tokenizer is not None
            self.tokenizer = tokenizer

    def create_new_batch(self, batch, device="cuda"):
        new_batch = dict()
        for k in batch:
            if k not in self.remove_columns:
                # remove irrelevant columns for bert.forward()
                new_batch[k] = batch[k].to(device)
        batch["output"] = batch["output"].to(device)
        if "prediction_dist" in batch:
            batch["prediction_dist"] = batch["prediction_dist"].to(device)
        return batch, new_batch

    def forward(self, batch, device="cuda"):
        new_batch = dict()
        for k in batch:
            if k not in self.remove_columns:
                # remove irrelevant columns for bert.forward()
                new_batch[k] = batch[k].to(device)

        encoding = self.model(**new_batch)
        batch["output"] = batch["output"].to(device)
        if "prediction_dist" in batch:
            batch["prediction_dist"] = batch["prediction_dist"].to(device)
        hidden_states = encoding.last_hidden_state
        batch_size, seq_len, dim = hidden_states.shape
        if self.extra_feat_dim > 0:
            assert "prediction_dist" in batch
            output = self.output(
                torch.cat(
                    [hidden_states, batch["prediction_dist"].unsqueeze(1).expand(
                        batch_size, seq_len, self.extra_feat_dim)],
                    dim=-1
                )
            ).squeeze(dim=-1)
        else:
            output = self.output(hidden_states).squeeze(dim=-1)
        if self.args is not None and hasattr(self.args, "fastshap") and self.args.fastshap:
            # adapted from official fastshap repo code
            assert len(batch["input_ids"]) == 1, "batch_size for fastshap must be 1 to allow shapley masking sampling"
            attn_mask = new_batch["attention_mask"]
            sampler = ShapleySampler(attn_mask.sum().item())
            shap_mask = sampler.sample(batch_size * self.n_sample, paired_sampling=True).to(device)
            shap_mask = torch.cat([shap_mask, torch.zeros(*shap_mask.shape[:-1], attn_mask.shape[-1] - sampler.num_players).to(attn_mask.device)], dim=-1)
            # attn_mask_shap = attn_mask * shap_mask
            zero_mask = torch.zeros_like(attn_mask)
            expand_batch = dict()
            expand_output = output.expand(self.n_sample, batch_size, seq_len).reshape(self.n_sample * batch_size, seq_len)
            for k in batch:
                if k not in self.remove_columns:
                    expand_batch[k] = batch[k].to(device).expand(self.n_sample, batch_size, -1).reshape(self.n_sample * batch_size, -1)
            backup_expand_input_ids = expand_batch["input_ids"].clone()
            target_model_original_output = self.target_model(**new_batch)[0].detach()
            original_prediction = target_model_original_output.argmax(-1)
            # full_original_output = target_model_original_output[torch.arange(batch_size), original_prediction].expand(self.n_sample, batch_size).reshape(self.n_sample * batch_size)
            expand_batch['input_ids'] = backup_expand_input_ids.masked_fill(~(shap_mask.bool()), self.tokenizer.pad_token_id)
            target_model_masked_output = self.target_model(**expand_batch)[0].data
            masked_prediction = target_model_masked_output.argmax(-1)
            masked_original_output = target_model_masked_output[torch.arange(len(masked_prediction)), original_prediction]
            expand_batch['input_ids'] = backup_expand_input_ids * 0 + self.tokenizer.pad_token_id

            target_model_zero_output = self.target_model(**expand_batch)[0].data
            zero_original_output = target_model_zero_output[torch.arange(batch_size), original_prediction]
            norm_output = expand_output
            loss_fn = nn.MSELoss()
            loss = loss_fn(masked_original_output, zero_original_output + (shap_mask * norm_output).sum(dim=-1))

            return self.post_processing(output, loss, encoding, batch, device)



        # backward compatibility
        if self.args is not None and hasattr(self.args, "neuralsort") and self.args.neuralsort:
            _, perm_pred = self.sortnn(output)
            tgt = batch["output"]
            perm_gt = torch.nn.functional.one_hot(batch["output_rank"]).transpose(-2, -1).float().to(device)
            loss = self.loss_func(perm_pred, perm_gt)
            return self.post_processing(output, loss, encoding, batch, device)
        if not hasattr(self, "discrete") or not self.discrete:
            tgt = batch["output"]
            if hasattr(self.args, "normalization") and self.args.normalization:
                tgt = 100 * (tgt - tgt.mean(dim=-1, keepdim=True)) / (1e-5 + tgt.std(dim=-1, keepdim=True))
            if self.args is not None and hasattr(self.args, "suf_reg") and self.args.suf_reg:
                if "zero_baseline" not in batch:
                    new_batch['input_ids'] = new_batch["input_ids"] * 0 + self.tokenizer.pad_token_id
                    target_model_zero_output = self.target_model(**new_batch)[0].data
                else:
                    target_model_zero_output = batch["zero_baseline"].to(device)
                original_prediction = batch["prediction_dist"].argmax(dim=-1)
                zero_original_output = target_model_zero_output[torch.arange(batch_size), original_prediction]
                full_original_output = batch['prediction_dist'][torch.arange(batch_size), original_prediction]
                output = output + 1/self.model.config.max_position_embeddings * (full_original_output - zero_original_output - output.sum(dim=-1)).unsqueeze(-1)
            loss = ((new_batch["attention_mask"] * (tgt - output)) ** 2).sum() / new_batch["attention_mask"].sum()
            return self.post_processing(output, loss, encoding, batch, device)
        else:
            gt = batch["output"]
            val, ind = torch.topk(gt, math.ceil(self.args.top_class_ratio * gt.shape[-1]), dim=-1)
            tgt = torch.zeros_like(gt).scatter(-1, ind, 1)
            loss = self.loss_func(
                output.reshape(-1, output.shape[-1]),
                tgt.reshape(-1).long(),
            ).reshape(output.shape[0], output.shape[1])
            loss = (new_batch["attention_mask"] * loss).sum() / new_batch["attention_mask"].sum()
            return self.post_processing(torch.argmax(output, dim=-1), loss, encoding, batch, device)

    def post_processing(self, main_output, main_loss, encoding, batch, device):
        # special handles in case we want to do multi-task fine-tuning
        if not hasattr(self, "multitask"):
            # backward compatibility
            return main_output, main_loss

        if not self.multitask:
            return main_output, main_loss
        else:
            pooled_output = encoding.pooler_output
            labels = batch['ft_label'].to(device)
            logits = self.ft_output(pooled_output)
            ft_loss = self.ft_loss_func(logits, labels)
            return main_output, main_loss, logits, ft_loss

    def svs_compute(self, batch, new_batch, device):
        batch["output"] = batch["output"].to(device)
        batch["prediction_dist"] = batch["prediction_dist"].to(device)
        batch_size, seq_len = batch['input_ids'].shape
        num_feature = self.sampler.num_players
        baseline = new_batch['input_ids'] * (batch["special_tokens_mask"].to(device))
        mask = torch.arange(num_feature)
        input_ids = new_batch['input_ids'].clone()
        # [batch_size, seq_len]
        output = torch.zeros_like(input_ids)
        original_output = self.target_model(**new_batch)[0].detach()
        target = original_output.argmax(dim=-1)
        new_batch['input_ids'] = baseline
        target_model_original_output = self.target_model(**new_batch)[0].detach()
        initial_logits = target_model_original_output[torch.arange(batch_size), target]
        for _sample_i in trange(self.n_sample, desc="sampling permutation..", leave=False):
            permutation = torch.randperm(num_feature).tolist()
            current_input = baseline
            prev_res = initial_logits
            for _permu_j in trange(num_feature, desc='doing masking...', leave=False):
                # only update one element at one time, reuse permutation across batch
                _mask = (mask == permutation[_permu_j]).unsqueeze(0).to(device)
                current_input = current_input * (~_mask) + input_ids * (_mask)
                new_batch["input_ids"] = current_input
                # [batch_size]
                modified_logits = self.target_model(**new_batch)[0].detach()[torch.arange(batch_size), target]
                # [batch_size, seq_len]  *   ([batch_size] -> [batch_size, 1])
                output = output + (modified_logits - prev_res).reshape(batch_size, 1) * _mask.float()
                prev_res = modified_logits
        return output / self.n_sample

    def _single_run(self, batch, new_batch):
        encoding = self.model(**new_batch)
        hidden_states = encoding.last_hidden_state
        batch_size, seq_len, dim = hidden_states.shape
        if self.extra_feat_dim > 0:
            assert "prediction_dist" in batch
            output = self.output(
                torch.cat(
                    [hidden_states, batch["prediction_dist"].unsqueeze(1).expand(
                        batch_size, seq_len, self.extra_feat_dim)],
                    dim=-1
                )
            ).squeeze(dim=-1)
        else:
            output = self.output(hidden_states).squeeze(dim=-1)
        return output


    def svs_compute_meta(self, batch, n_samples, device, target_model, use_imp=False, use_init=False, inv_temper=-1):
        # doing guided importance sampling for ICLR rebuttal
        batch, new_batch = self.create_new_batch(batch, device)
        batch_size = new_batch["input_ids"].shape[0]
        assert batch_size == 1
        baseline = new_batch['input_ids'] * (batch["special_tokens_mask"].to(device))
        baseline = baseline[0][new_batch["attention_mask"][0] > 0].unsqueeze(0)
        for key in new_batch:
            if torch.is_tensor(new_batch[key]):
                for _batch_i in range(batch_size):
                    new_batch[key] = new_batch[key][_batch_i][new_batch["attention_mask"][_batch_i] > 0].unsqueeze(0)
        explainer_output = self._single_run(batch, new_batch)
        for _batch_i in range(batch_size):
            explainer_output = explainer_output[_batch_i][new_batch["attention_mask"][_batch_i] > 0].unsqueeze(0)
        batch["output"] = batch["output"].to(device)
        batch["prediction_dist"] = batch["prediction_dist"].to(device)
        #hidden_states = encoding.last_hidden_state
        # batch_size, seq_len, dim = hidden_states.shape
        batch_size, seq_len = new_batch['input_ids'].shape
        #batch_size, seq_len = batch['input_ids'].shape
        #if not hasattr(self, "sampler"):
            #self.sampler = ShapleySampler(self.model.config.max_position_embeddings)
        #num_feature = self.sampler.num_players
        num_feature = seq_len
        gumbel_dist = torch.distributions.gumbel.Gumbel(torch.Tensor([0]), torch.Tensor([1]))
        gumbel_noise = gumbel_dist.sample([n_samples, num_feature]).squeeze(-1)
        if inv_temper > 0:
            noised_output = inv_temper * explainer_output + torch.log(gumbel_noise).cuda()
        else:
            noised_output = explainer_output + torch.log(gumbel_noise).cuda()
        noised_output_ranking = torch.argsort(-1.0 * noised_output, dim=-1)
        mask = torch.arange(num_feature)
        input_ids = new_batch['input_ids'].clone()
        # [batch_size, seq_len]
        output = torch.zeros_like(input_ids).float()
        if use_init:
            output += explainer_output
        original_output = target_model(**new_batch)[0].detach()
        target = original_output.argmax(dim=-1)
        new_batch['input_ids'] = baseline
        target_model_original_output = target_model(**new_batch)[0].detach()
        initial_logits = target_model_original_output[torch.arange(batch_size), target]
        for _sample_i in trange(n_samples, desc="sampling permutation..", leave=False):
            if use_imp:
                permutation = noised_output_ranking[_sample_i].cpu().tolist()
            else:
                permutation = torch.randperm(num_feature).tolist()
            current_input = baseline
            prev_res = initial_logits
            for _permu_j in trange(num_feature, desc='doing masking...', leave=False):
                # only update one element at one time, reuse permutation across batch
                _mask = (mask == permutation[_permu_j]).unsqueeze(0).to(device)
                current_input = current_input * (~_mask) + input_ids * (_mask)
                new_batch["input_ids"] = current_input
                # [batch_size]
                modified_logits = target_model(**new_batch)[0].detach()[torch.arange(batch_size), target]
                # [batch_size, seq_len]  *   ([batch_size] -> [batch_size, 1])
                output = output + (modified_logits - prev_res).reshape(batch_size, 1) * _mask.float()
                prev_res = modified_logits
        return output / n_samples

    def kernelshap_meta(self, batch, n_samples, device, target_model=None):
        # doing guided importance sampling for ICLR rebuttal
        batch, new_batch = self.create_new_batch(batch, device)
        explainer_output = self._single_run(batch, new_batch)
        batch["output"] = batch["output"].to(device)
        batch["prediction_dist"] = batch["prediction_dist"].to(device)
        batch_size, seq_len = batch['input_ids'].shape
        if not hasattr(self, "sampler"):
            self.sampler = ShapleySampler(self.model.config.max_position_embeddings)
        num_feature = self.sampler.num_players
        baseline = new_batch['input_ids'] * (batch["special_tokens_mask"].to(device))
        mask = torch.arange(num_feature)
        input_ids = new_batch['input_ids'].clone()
        # [batch_size, seq_len]
        output = torch.zeros_like(input_ids)
        if target_model is None:
            original_output = self.target_model(**new_batch)[0].detach()
        else:
            original_output = target_model(**new_batch)[0].detach()
        target = original_output.argmax(dim=-1)
        new_batch['input_ids'] = baseline
        if target_model is None:
            target_model_original_output = self.target_model(**new_batch)[0].detach()
        else:
            target_model_original_output = target_model(**new_batch)[0].detach()
        initial_logits = target_model_original_output[torch.arange(batch_size), target]
        new_output = []


        for _batch_i in trange(batch_size, desc="processing instance..", leave=False):
            output_batch_i = explainer_output[_batch_i][new_batch["attention_mask"][_batch_i] > 0]
            regressor = LinearRegression()
            sampler = ShapleySampler(len(output_batch_i))
            seq_len_i = len(output_batch_i)
            mask_samples, weights = self.sampler.dummy_sample_with_weight(n_samples, False, output_batch_i)
            mask_samples = mask_samples.to(device)
            batch_i_masked = {}
            # [batch_size, seq_len] * [1, seq_len]
            batch_i_masked["input_ids"] = (mask_samples * (new_batch["input_ids"][_batch_i][new_batch["attention_mask"][_batch_i] > 0]).unsqueeze(0)).int()
            for key in new_batch:
                if key == "input_ids":
                    continue
                else:
                    batch_i_masked[key] = (new_batch[key][_batch_i][new_batch["attention_mask"][_batch_i] > 0]).unsqueeze(0).expand(n_samples, seq_len_i)
            if target_model is None:
                output_i = self.target_model(**batch_i_masked)[0].detach()[:, target[_batch_i]]
            else:
                output_i = target_model(**batch_i_masked)[0].detach()[:, target[_batch_i]]
            try:
                regressor.fit(mask_samples.cpu().numpy(), output_i.cpu().numpy())
                new_ks_weight = regressor.coef_
                new_output.append((new_ks_weight, batch["output"][_batch_i][new_batch['attention_mask'][_batch_i] > 0].cpu().numpy()))
            except:
                print("cannot fit, debug:")
                print(mask_samples.min(), mask_samples.max())
                print(weights.min(), weights.max())
                print(output_i.min(), output_i.max())
        return new_output

        #
        # for _sample_i in trange(self.n_sample, desc="sampling permutation..", leave=False):
        #     permutation = torch.randperm(num_feature).tolist()
        #     current_input = baseline
        #     prev_res = initial_logits
        #     for _permu_j in trange(num_feature, desc='doing masking...', leave=False):
        #         # only update one element at one time, reuse permutation across batch
        #         _mask = (mask == permutation[_permu_j]).unsqueeze(0).to(device)
        #         current_input = current_input * (~_mask) + input_ids * (_mask)
        #         # print((current_input > 0).sum())
        #         new_batch["input_ids"] = current_input
        #         # [batch_size]
        #         modified_logits = self.target_model(**new_batch)[0].detach()[torch.arange(batch_size), target]
        #         # [batch_size, seq_len]  *   ([batch_size] -> [batch_size, 1])
        #         output = output + (modified_logits - prev_res).reshape(batch_size, 1) * _mask.float()
        #         prev_res = modified_logits




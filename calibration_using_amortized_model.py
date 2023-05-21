from amortized_model import AmortizedModel
from transformers import AutoTokenizer
import torch
import os
import glob
from tqdm import tqdm
# required by the bin file loading
#from InterpCalib.NLI import dataset_utils
#from NLI import dataset_utils

# example path, change it to your own
model_fn = "/path/to/amortized_model_formal/multi_nli/lr_5e-05-epoch_30/seed_3_prop_1.0/model_svs_norm_False_discrete_False.pt"
model = torch.load(model_fn).cuda().eval()

model_cache_dir = "./models/"
model_name = "textattack/bert-base-uncased-MNLI"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
# clone InterpCalib repo and change the path to your own
# interpretations_custom is the folder containing the output files, see their README for more details
source_dirs = glob.glob("/path/to/InterpCalib/NLI/interpretations_custom/shap/mnli_mrpc_*")

for source_dir in source_dirs:
    target_dir = source_dir.replace("custom", "amortized")
    os.makedirs(target_dir, exist_ok=True)
    bin_fns = glob.glob(os.path.join(source_dir, "*.bin"))
    for bin_fn in tqdm(bin_fns):
        basename = os.path.basename(bin_fn)
        data = torch.load(bin_fn)
        with open(os.path.join(target_dir, basename), "wb") as f_out:
            premise, hypo = data['example'].premise, data['example'].hypothesis
            batch = tokenizer([premise, ], [hypo, ], truncation=True, return_tensors="pt", return_special_tokens_mask=True)
            assert (batch['input_ids'][0] == torch.LongTensor(data['example'].input_ids)).all()
            batch["output"] = torch.stack([torch.tensor([1, ] * len(batch['input_ids'][0])), ] * len(batch['input_ids']))
            batch["prediction_dist"] = torch.stack([torch.tensor([1, ] * 3), ] * len(batch['input_ids']))
            output, loss = model(batch)
            if len(output.shape) == 2:
                output = output[0]
            output = output.cpu().detach()
            assert len(output) == len(data['attribution'])
            data['attribution'] = output
            torch.save(data, os.path.join(target_dir, basename))



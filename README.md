# Amortized-Interpretability
Codebase for the ACL 2023 paper "Efficient Shapley Values Estimation by Amortization for Text Classification"

Author Team: 

1. Chenghao Yang (yangalan1996@gmail.com) (University of Chicago)

2. Fan Yin (fanyin20@cs.ucla.edu) (University of California, Los Angeles)

Supervisor Team:
1. He He (New York University)
2. Kai-Wei Chang (University of California, Los Angeles)

Industrial Support:

Xiaofei Ma, Bing Xiang (AWS AI Labs)


## Reference
If you use this code as part of any published research, please acknowledge the following paper (it encourages researchers who publish their code!):

```
@inproceedings{yang-2023-amortized,
    title = "Efficient Shapley Values Estimation by Amortization for Text Classification",
    author = "Yang, Chenghao and Yin, Fan and He, He and Chang, Kai-Wei and Ma, Xiaofei and Xiang, Bing",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics",
    year = "2023",
    publisher = "Association for Computational Linguistics",
}
```

## Project Structure
1. `thermostat`: Configuration files we use to run Ranking Stability experiments. You should update it with original [thermotet](https://github.com/DFKI-NLP/thermostat) Repo to get most up-to-date implementation. Here we just upload our updated thermostat repo. 
2. `InterpCalib`: [Calibration via Interpretation (ACL'22)](https://arxiv.org/abs/2110.07586) We update the [implementation](https://github.com/xiye17/InterpCalib) contributed by Xi Ye to 1) allow using different random seeds when computing Shapley Values; 2) updated the I/O interface to use our own outputs to do calibration. We here upload our updated InterpCalib repo.
3. Other parts are just our own codes. Chenghao will do better packaging later to improve readability. But importantly, `amortized_model.py` is the main Amortized Model code and you can run via `run.py`. 

## Dependency Installation
For the main repo:
```
git clone https://github.com/yangalan123/Amortized-Interpretability.git
cd Amortized-Interpretability
git submodule update --init --recursive
conda create -p ./env python=3.9 # Python 3.8 should also work, but we use 3.9 here for better compatibility. This is subject to future updates.
conda activate ./env # the environment position is optional, you can choose whatever places you like to save dependencies. Here I choose ./env for example.
cd thermostat # we now have the local thermostat copy, so let's build it first
pip install -e . # please ignore the requirements.txt in thermostat/, as it is older and may result in unexpected errors
cd ..
pip install -r requirements.txt # build the main repo dependencies
```

For the dependency of `thermostat` and `InterpCalib`, please check their individual README.

## Running Instructions (Ranking Stability for Shapley Values)
1. First we need to compute a bunch of Shapley Values with different random seeds. We use `thermostat` to do this. Please check `thermostat/README.md` for more details. 
We prepare a running script `thermostat/run.sh` to assist you. For example, you can run 
```
   bash run.sh task=yelp_polarity model=bert explainer=svs-3600 seed=1 batch_size=1 device=0
```

We understand it might be computationally expensive to run all the seeds. So we provide a pre-computed Shapley Values [here](https://drive.google.com/file/d/1kOIEEuEHG-zDmZ3rwYLDOV-RtRyvBxiu/view?usp=sharing).
You can download and unzip it under the `thermostat` directory. The resulted directory structure should be `thermostat/experiments/...`.
2. Then you can compute Spearman's Ranking Correlation Coefficient (SRCC) between different Shapley Values. 
Check out `internal_correlation.py` for more details. You need to update the directory of Shapley Values in the code.
This file will automatically create Table 1-2 and Figure 2 in the paper.

## Running Instructions (Amortized Model Training)
1. We mainly use the pre-computed `thermostat` Shapley Values to train our Amortized Model. You should follow the instructions in `thermostat/README.md` to compute Shapley Values first.
2. We need to create the dataset for training and evaluation purposes. Please check out `create_dataset.py` for more details. You need to update the filepath there. 
3. Then, you can run `run.py` to train the Amortized Model. You can check out `run.sh` for more details. The example running command would be:
```
  CUDA_VISIBLE_DEVICES=${device} python run.py --seed ${seed}  --lr ${lr} -e ${epoch} --train_bsz ${train_bsz} --explainer ${explainer} --topk 10 --task ${task} -tm ${target_model} --storage_root ${output_dir}
```
Note that running `run.py` will automatically compute the performance numbers in Table 3 and you can find them in the output directory. To save computational resources, this code will first check the model is already trained and saved. If not, it will train the model.

Feel free to change the setting in `config.py` for better training performance.

4. You can use `compute_amortized_model_consistency.py` to compute the consistency between Amortized Model and Shapley Values. You need to update the filepath there. This file will automatically create Table 5 in the paper.

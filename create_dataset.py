import thermostat
import random
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
# output_dir = "./amortized_dataset/imdb_test"
output_dir = "./amortized_dataset/mnli_test"
# output_dir = "./amortized_dataset/yelp_test"
model_cache_dir = "./models/"
if __name__ == '__main__':
    # data_cache_dir = "./datasets/imdb"
    # data_cache_dir = "./datasets/mnli"
    data_cache_dir = "./datasets/yelp_polarity"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_cache_dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = data_cache_dir
    # data = thermostat.load("imdb-bert-lime", cache_dir=data_cache_dir)
    # dataset = load_dataset("imdb")
    # task = "multi_nli"
    task = "yelp_polarity"
    #dataset = load_from_disk("thermostat/experiments/thermostat/datasets/imdb")
    # dataset = load_from_disk(f"thermostat/experiments/thermostat/datasets/{task}")
    dataset = load_from_disk(f"thermostat/experiments/thermostat/datasets/{task}")
    # model_name = "textattack/bert-base-uncased-imdb"
    # model_name = "textattack/bert-base-uncased-MNLI"
    model_name = "textattack/bert-base-uncased-yelp-polarity"
    #if model_name == "textattack/bert-base-uncased-MNLI":
        #label_mapping_dict = {
            #0: 2,
            #1: 0,
            #2: 1
        #}
        #label_mapping = lambda x: label_mapping_dict[x]
    #else:
        #label_mapping = lambda x: x
    label_mapping = lambda x: x
    #explainer = "svs"
    for explainer in ["svs", ]:
    # for explainer in ["svs", "lime", "lig"]:
        data = thermostat.load(f"{task}-bert-{explainer}", cache_dir=data_cache_dir)
        instance = data[0]

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
        # id_pkl_name = "dumped_split_ids_0803.pkl"
        id_pkl_name = "dumped_split_ids.pkl"
        try:
            # train_ids, valid_ids, test_ids = torch.load(os.path.join(output_dir, "dumped_split_ids_0523.pkl"))
            train_ids, valid_ids, test_ids = torch.load(os.path.join(output_dir, id_pkl_name))
            print("successfully load pre-split data ids")
        except:
            print("fail to load pre-split data ids, re-splitting...")
            all_ids = list(range(len(data)))
            assert len(all_ids) > 2000
            # assert 0.1 * len(all_ids) <= 2000
            # to make sure our amortized model can compare to traditional interpretation methods
            # random.shuffle(all_ids)
            # test_ids = random.sample(list(range(2000)), int(0.1 * len(all_ids)))
            test_ids = list(range(500)) + random.sample(list(range(500, 3000)),
                                                        max(int(0.1 * len(all_ids)) - 500, 0))
            rest_ids = list(set(all_ids) - set(test_ids))
            random.shuffle(rest_ids)
            # train_ids = all_ids[: int(0.8 * len(all_ids))]
            train_ids = rest_ids[: int(0.8 * len(all_ids))]
            valid_ids = rest_ids[len(train_ids): ]
            # test_ids = all_ids[len(train_ids) + len(valid_ids): ]
            torch.save([train_ids, valid_ids, test_ids], os.path.join(output_dir, id_pkl_name))
        # train_dataset = [dataset["test"][i]['text'] for i in train_ids]
        # valid_dataset = [dataset["test"][i]['text'] for i in valid_ids]
        # test_dataset = [dataset["test"][i]['text'] for i in test_ids]
        # test_ids = [x for x in test_ids if x < 2000]
        if task == "multi_nli":
            train_dataset = [(dataset[i]['premise'], dataset[i]['hypothesis']) for i in train_ids]
            valid_dataset = [(dataset[i]["premise"], dataset[i]['hypothesis']) for i in valid_ids]
            test_dataset = [(dataset[i]["premise"], dataset[i]['hypothesis']) for i in test_ids]
        else:
            train_dataset = [dataset[i]['text'] for i in train_ids]
            valid_dataset = [dataset[i]['text'] for i in valid_ids]
            test_dataset = [dataset[i]['text'] for i in test_ids]
        # train_dataset = tokenizer(train_dataset, return_tensors='pt', padding='max_length', truncation=True)
        train_dataset = tokenizer(train_dataset,  padding='max_length', truncation=True, return_special_tokens_mask=True)
        train_dataset["output"] = [data[i].attributions for i in train_ids]
        train_dataset["output_rank"] = [torch.argsort(torch.tensor(data[i].attributions)).tolist() for i in train_ids]
        # train_dataset["ft_label"] = [dataset['test'][i]["label"] for i in train_ids]
        train_dataset["ft_label"] = [label_mapping(dataset[i]["label"]) for i in train_ids]
        train_dataset["prediction_dist"] = [data[i].predictions for i in train_ids]
        train_dataset["id"] = train_ids

        valid_dataset = tokenizer(valid_dataset,  padding='max_length', truncation=True, return_special_tokens_mask=True)
        valid_dataset["output"] = [data[i].attributions for i in valid_ids]
        valid_dataset["output_rank"] = [torch.argsort(torch.tensor(data[i].attributions)).tolist() for i in valid_ids]
        # valid_dataset["ft_label"] = [dataset['test'][i]["label"] for i in valid_ids]
        valid_dataset["ft_label"] = [label_mapping(dataset[i]["label"]) for i in valid_ids]
        valid_dataset["prediction_dist"] = [data[i].predictions for i in valid_ids]
        valid_dataset["id"] = valid_ids

        test_dataset = tokenizer(test_dataset,  padding='max_length', truncation=True, return_special_tokens_mask=True)
        test_dataset["output"] = [data[i].attributions for i in test_ids]
        test_dataset["output_rank"] = [torch.argsort(torch.tensor(data[i].attributions)).tolist() for i in test_ids]
        # test_dataset["ft_label"] = [dataset['test'][i]["label"] for i in test_ids]
        test_dataset["ft_label"] = [label_mapping(dataset[i]["label"]) for i in test_ids]
        test_dataset["prediction_dist"] = [data[i].predictions for i in test_ids]
        test_dataset["id"] = test_ids
        for _dataset, ids, status in zip([train_dataset, valid_dataset, test_dataset], [train_ids, valid_ids, test_ids], ['train', "valid", "test"]):
            for id_i, _id in enumerate(ids):
                assert _dataset["input_ids"][id_i] == data[_id].input_ids
            print(f"{status} input ids check complete")

        torch.save([train_dataset, valid_dataset, test_dataset], os.path.join(output_dir, f"data_{explainer}.pkl"))

        # for data_i, data_entry in enumerate(dataset):
        #     all_data.append({
        #         "output": [x[1] for x in data[data_i].explanation]
        #     })


        # model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=model_cache_dir)

        # print(len(dataset["test"]))
        # print(len(data))
        # print(instance.explanation)
        # print(len(instance.explanation))
        # print(instance.attributions[-10:])
        # print(instance.attributions)
        # print(len(instance.attributions))
        # print(tokenizer(dataset['test'][0]["text"], padding="max_length", return_tensors='pt')["attention_mask"][:10].sum())
        # print(tokenizer(dataset['test'][0]["text"], padding="max_length", return_tensors='pt')["attention_mask"][-10:].sum())
        # print(tokenizer(dataset['test'][0]["text"], return_special_tokens_mask=True, padding="max_length", return_tensors='pt')["attention_mask"][:10])
        # print(tokenizer(dataset['test'][0]["text"], return_special_tokens_mask=True, padding="max_length", return_tensors='pt')["attention_mask"][-10:])
        # dataset_sample = dataset["test"][0]
        # print(dataset["test"][0])
        # # for i in range(len(data)):
        # #     len_lime_sample = len(data[i].explanation)
        # #     len_data_sample = len(tokenizer(dataset["test"][i]["text"], truncation=True)["input_ids"])
        # #     assert len_lime_sample == len_data_sample, f"len_lime: {len_lime_sample}, len_data: {len_data_sample}"
        # print(tokenizer(dataset_sample['text'])["input_ids"])
        # print(tokenizer.convert_ids_to_tokens(tokenizer(dataset_sample['text'])["input_ids"]))
        # print(len(tokenizer(dataset_sample['text'])["input_ids"]))

        # print(help(instance))

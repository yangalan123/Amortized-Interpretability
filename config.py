class Args:
    def __init__(self, seed, task_name, batch_size=16, epochs=20, explainer="svs", proportion="1.0", normalization=True,
                 discretization=False, validation_period=5, top_class_ratio=0.2, lr=2e-4, multitask=False, neuralsort=True, sort_arch="bitonic",
                 suf_reg=False, path_name_suffix="formal", storage_root="path/to/dir"):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        # self.explainer = "svs"
        # self.explainer = "lime"
        self.explainer = explainer
        self.proportion = proportion
        self.extra_feat_dim = 0
        self.discrete = discretization
        self.top_class_ratio = top_class_ratio
        self.validation_period = validation_period
        self.seed = seed
        self.normalization = normalization
        self.neuralsort = neuralsort
        self.sort_arch = sort_arch
        self.multitask = multitask
        self.suf_reg = suf_reg
        # path_name_suffix = "formal"
        self.fastshap = False
        if self.multitask:
            path_name_suffix = "multi_task"
        if self.fastshap:
            self.extra_feat_dim = 0
            path_name_suffix += "fastshap"
        path_name_suffix += f"/{task_name}"
        storage_root = storage_root
        if self.suf_reg:
            path_name_suffix += "_suf_reg"
        self.save_path = f"{storage_root}/amortized_model_{path_name_suffix}" \
                         f"{'-extradim-' + str(self.extra_feat_dim) if self.extra_feat_dim > 0 else ''}" \
                         f"/lr_{self.lr}-epoch_{self.epochs}/seed_{self.seed}_prop_{self.proportion}/model_{self.explainer}_norm_{self.normalization}_discrete_{self.discrete}.pt"
        if self.neuralsort:
            self.save_path = f"{storage_root}/amortized_model_{path_name_suffix}" \
                             f"{'-extradim-' + str(self.extra_feat_dim) if self.extra_feat_dim > 0 else ''}" \
                             f"/{self.sort_arch}_trans_sort_lr_{self.lr}-epoch_{self.epochs}/seed_{self.seed}_prop_{self.proportion}/model_{self.explainer}_norm_{self.normalization}_discrete_{self.discrete}.pt"

def GetParser(parser):
    parser.add_argument("-s", "--seed", default=1111, type=int, help="seed?")
    parser.add_argument("-e", "--epoch", default=20, type=int, help="seed?")
    parser.add_argument("--lr", default=2e-4, type=float, help="lr?")
    parser.add_argument("--train_bsz", default=1, type=int, help="batch_size for training?")
    parser.add_argument("--test_bsz", default=1, type=int, help="batch_size for testing?")
    parser.add_argument("-K", "--topk", default=50, type=int, help="top-k intersection?")
    parser.add_argument("--explainer", default="lime", type=str, help="which explainer you want to use? svs/lime/lig? if multiple, concat by one single comma")
    parser.add_argument("--task", default="imdb", type=str, help="which task you want to use? ")
    parser.add_argument("-am", "--amortized_model", default="bert-base-uncased", type=str,
                        help="use which arch for amortized model?")
    parser.add_argument("-tm", "--target_model", default="textattack/bert-base-uncased-MNLI", type=str,
    # parser.add_argument("-tm", "--target_model", default="textattack/bert-base-uncased-imdb", type=str,
                        help="use which arch for target model?")
    parser.add_argument("-norm", "--normalization", action="store_true",
                        help="use normalization?")
    parser.add_argument("-disc", "--discrete", action="store_true",
                        help="use discretization?")
    parser.add_argument("-sort", "--neuralsort", action="store_true",
                        help="use neuralsorting?")
    parser.add_argument("-mul", "--multitask", action="store_true",
                        help="use multitasking (add fine-tuning original classification tasks)?")
    parser.add_argument("--suf_reg", action="store_true",
                        help="add sufficiency regularization?")
    parser.add_argument("--storage_root", type=str, help="where to store the output?")
    return parser

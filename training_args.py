# -----------ARGS---------------------
train_path = "data/train_clean_v2.csv"
#dev_path = "data/dev_clean.csv"
submit_path="data/test_final_clean_v2.csv"
max_seq_length = 100
use_cuda=True
do_train = True
do_lower_case = True
load_model=False
train_batch_size = 16
eval_batch_size = 16
lr = 2e-5
max_lr=2e-5
mean_lr=2e-5
cls_lr=2e-5
epochs = 20
warmup_proportion = 0.1
local_rank = -1
seed = 42
gradient_accumulation_steps = 1
bert_config_json = "../weights/robert/bert_config.json"
vocab_file = "../weights/chinese_roberta_wwm_large_ext_pytorch/vocab.txt"
label_file="tags.txt"
output_dir = "outputs"
masked_lm_prob = 0.15
max_predictions_per_seq = 20

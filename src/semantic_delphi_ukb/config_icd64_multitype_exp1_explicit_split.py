import time

out_dir = "ckpt/Delphi_semantic_icd64_multitype_exp1_explicit_split"
eval_interval = 250
eval_iters = 25
log_interval = 25
seed = 42

always_save_checkpoint = False

wandb_log = False
wandb_project = "delphi"
wandb_run_name = "semantic_icd64_multitype_exp1_explicit_split_" + str(time.time())

dataset = "ukb_semantic_multitype_exp1_explicit_split"
batch_size = 128
block_size = 48
data_fraction = 1.0

n_layer = 6
n_head = 8
n_embd = 64
dropout = 0.1
weight_decay = 2e-1
vocab_size = 1270

learning_rate = 6e-4
max_iters = 100000
lr_decay_iters = 100000
min_lr = 6e-5
beta2 = 0.99

warmup_iters = 1000
ignore_tokens = [0]
t_min = 0.1
token_dropout = 0.0
no_event_token_rate = 5

token_embedding_path = "auto"
freeze_input_embeddings = False
tie_input_output_embeddings = False

static_dim = 10
static_hidden_dim = 64
static_dropout = 0.05
fusion_mode = "residual"
fuse_static_to_logits = False

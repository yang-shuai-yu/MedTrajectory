out_dir = "ckpt/MedTrajectory_exp2_tte_w01_h10"
init_from_ckpt = "ckpt/MedTrajectory_exp2_modern_baseline/ckpt.pt"

dataset = "ukb_semantic_multitype_explicit_split"
batch_size = 96
block_size = 128
data_fraction = 1.0

learning_rate = 1e-4
max_iters = 20000
lr_decay_iters = 20000
min_lr = 1e-5
warmup_iters = 500

eval_interval = 250
eval_iters = 25
log_interval = 25

tte_loss_weight = 0.1
tte_horizon_years = 10.0
diseases_yaml = "selected_diseases.yaml"
no_event_token_rate = 5

model_family = "medtrajectory_tte_w01_h10"

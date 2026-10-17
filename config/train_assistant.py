# Optimized modern assistant character-level GPT configuration
# Enhanced block_size (128) and extended training iterations (1200) for low loss and coherent answers

out_dir = 'out-assistant'
eval_interval = 100
eval_iters = 20
log_interval = 50

always_save_checkpoint = True

wandb_log = False
dataset = 'assistant_chat'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 128

# Efficient 4-layer transformer for smooth laptop CPU training
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

learning_rate = 1.2e-3
max_iters = 1200
lr_decay_iters = 1200
min_lr = 1e-4
beta2 = 0.99

device = 'cpu'
compile = False

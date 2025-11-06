import torch
import inspect
import math
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import os

# --- START DEBUG CODE ---
# print("--- CUDA DEBUG INFO ---")
# print(f"CUDA available: {torch.cuda.is_available()}")
# print(f"Device count: {torch.cuda.device_count()}")

# if torch.cuda.is_available():
#     print(f"Current device: {torch.cuda.current_device()}")
#     print(f"Device name: {torch.cuda.get_device_name(0)}")

# print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
# print("-------------------------")
# --- END DEBUG CODE ---

# ----------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projections
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # at the end of each block the residual stream is projected and added back.
        # This increases the variance of the residual stream, so to compensate,
        # scaling factor of 1/sqrt(2*number of layers) is applied to the projection
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, following OpenAI/HF naming
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))


    def forward(self, x):
        B, T, C = x.size() # batch, sequence length, Embedding dimentionality (n_embd)
        # Calculate query, key, values for all heads in batch and move head forward to be the batch
        # nh: number of heads, hs: head size, C: number of channels: nh * hs
        # eg GPT-2 (124M) n_head = 12, hs = 64, nh*hs=C=768 channels in transformer
        qkv = self.c_attn(x) # (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embd, dim =2)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        # attention (materializes the large T,T matrix for all queries and keys)

        # For optimization, we can use torch's built-in scaled_dot_product_attention function
        # Flash Attention implementation - Fusing softmax and matmul kernels for better performance and memory access efficiency
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        # att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] ==0, float('-inf')) # auto regressive mask
        # att = F.softmax(att, dim=-1)
        # y = att @ v   # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1,2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4* config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int =  12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme (Implemented both in Attention is all you need and GPT-2)
        self.transformer.wte.weight = self.lm_head.weight

        # parameter initialization to match the GPT-2 paper
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # two residual paths per layer, attn and mlp
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    # To be able to generate from the model, we have to use forward method
    def forward(self, idx, targets=None):
        # idx is of shape (B,T) Inputs to the model
        B, T = idx.shape
        # every row is upto the size of block_size-> max sequence length
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}" 
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        x = pos_emb + tok_emb
        # forward the blocks of transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layerNorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # cross_entropy doesn't like multi-dimentional inputs (B, T, vocab_size), so flatten 3D to 2D
            # 1st dim is calculated automatically == (B*T, vocab_size)
            # targets are 2D, flattening them to 1D (B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # Load parameters from hugging face GPT model and initialize GPT class here
    @classmethod
    def from_pretrained(cls, model_type):
        """ Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print('Loading weights from pretrained gpt: %s' % model_type)

        # n_layer, n_head, n_embd are determined from model_type
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),    # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),   # 350M params
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),   # 774M params
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),   # 1558M params
        } [model_type]
        config_args['vocab_size'] = 50257 # always for GPT model checkpoints
        config_args['block_size'] = 1024
        # create a minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer

        # init a huggingface transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these masks
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # openAI checkpoints use Conv1D module, but only want to use a vanilla version
        # so, we have to transpose these weights when importing
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for conv1D weights 
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all the candidate parameters (that require gradients)
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optimizer groups
        # 1. Any parameter which is 2D will be weight decayed else no
        # ie all weight tensors in matmuls + embeddings will decay, all biases and LayerNorm/BatchNorm parameters not
        decay_params = [p for pn, p in param_dict.items() if p.ndim >= 2]
        no_decay_params = [p for pn, p in param_dict.items() if p.ndim < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(f"num decay parameters tensors: {len(decay_params)}, num decay parameters: {num_decay_params}")
        print(f"num no decay parameters tensors: {len(no_decay_params)}, num no decay parameters: {num_no_decay_params}")
        # create the AdamW optimizer and use fused version if possible
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device 
        print(f"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


# ----------------------------------------------------------------------

# Using multiple devices gpus if available
from torch.distributed import init_process_group, destroy_process_group

# set up DDP (distributed data parallel)
# torchrun cmd sets the env vars RANK, LOCAL_RANK, WORLD_SIZE
# torchrun --standalone --nproc_per_node=4 train_gpt2.py
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    assert torch.cuda.is_available(), "DDP mode requires CUDA"
    init_process_group(backend='nccl') # initialize the process group for ddp
    ddp_rank = int(os.environ['RANK']) # the rank of current gpu
    local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE']) # no. of gpus across all nodes
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # only the master process will do logging, saving model checkpoints etc.
    print(f"DDP: global rank {ddp_rank} local rank {local_rank} world size {ddp_world_size}")
else:
    ddp_rank = 0
    local_rank = 0
    ddp_world_size = 1
    master_process = True
    # auto-detect the device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f'Using device: {device}')
    #device = 'cpu' # OVERRIDE

# for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

import time

# gradient accumulation steps - used to simulate larger batch sizes that don't fit
total_batch_size = 524288 # total effective batch size 2 ** 19 ~ 0.5 M GPT-2 batch size
B = 64 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total batch size is multiple of B * T"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# To train the model from scratch ----------------------------------------------------
# get the data batch, calculate loss
import tiktoken
from data_loader import DataLoaderLite
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split = 'train') # batch size 4, sequence length 32. By default you want to max out the GPU with batch size
torch.set_float32_matmul_precision('high')  # use high precision for matmuls - bFloat16


# create the model 
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
raw_model = model.module if ddp else model

#logits, loss = model(x, y) # when training the model, return loss as well and pass in targets y 

#print(loss)
#print(logits.shape) # (B, T, vocab_size)

# At initialization, we would like to intialize the weights roughly uniformly
# so that we are not favouring any particular token and we are not confidently wrong at initialization

# cross-entrpy loss is -ve log likelihood of the true token
# total no. of tokens = vocab size = 50257
# prob of each token at initialization = 1/50257
# -log(1/50257) = log(50257) = 10.82
# - at initialization, our probabilities are roughly diffused and it is a good starting point

# optimizing the model parameters
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # 375e6 warmup tokens
max_steps = 19073 # we have 10e9  tokens with 2**19 tokens/step 

def get_lr(step):
    # 1. linear warmup for first warmup steps
    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps
    # 2. if step > lr_decay_iter, return min_lr
    if step >= max_steps:
        return min_lr
    # 3. cosine decay down to 10% of original lr over remaining steps
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # cosine from 1 to 0
    return min_lr + coeff * (max_lr - min_lr) # from 1.0 to 0.1

#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, device=device)

for step in range(max_steps):
    # forward the model
    t0 = time.time()
    optimizer.zero_grad() ##### Always zero the gradients before running the backward pass
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.get_batch()
        x = x.to(device)
        y = y.to(device)
        # only possible on Ampere GPUs and later
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps # normalize the loss because it is accumulated
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        # backward pass
        loss.backward()
    if ddp:
        torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # global gradient clipping to prevent shock to the model
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step() # to update the parameters and decrease the loss
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0 # time in ms
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if step % 10 == 0 and master_process:
        print(f"step {step}, loss {loss_accum.item():.4f} | lr: {lr:.4f} | norm: {norm:.4f} | time {dt*1000:.2f}ms tokens/sec {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()

import sys
sys.exit(0)

# Sampling logic -------------------------------------------------------------------
num_return_sequences= 5
max_length = 30

# model = GPT(GPTConfig()) # initialize a new model from scratch
model = GPT.from_pretrained('gpt2')
model.eval() # Good practice to put model in eval mode when not actually training it
model.to(device)

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
x = tokens.to(device)
# x is the idx forwarded to the model to predict the 9th token for each of the 5 rows

# To generate - x is (B=5, T=8)
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take logits at last position - throw away all other logits - inefficient way of sampling
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # top-k sampling of 50 (huggingface pipeline default)
        # top_probs become (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from top-k probs
        ix = torch.multinomial(topk_probs, 1) #(B, 1)
        # gather the corresponding indices 
        xcol = torch.gather(topk_indices, -1, ix) #(B, 1)
        # append to the sequence 
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

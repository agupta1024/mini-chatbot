import torch
import torch.nn as nn
from torch.nn import functional as F

# hyper-parameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4 # self-attention can't tolerate high lr
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ---------------

torch.manual_seed(1337)

#wget https://...input.txt
with open('/Users/anubhagupta/Downloads/ml_tests/Nano GPT/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# collect all unique characters that appear in text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# tokenize the text - convert raw text of strings to some sequence of integers
# according to some vocabulary of possible elements

# eg. for char level language model, we'll simply convert individual characters into integers

stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for s,i in stoi.items()}
encode = lambda s: [stoi[c] for c in s] # encoder: takes input string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decode: takes a list of integers, output a string


# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+ block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class FeedForward(nn.Module):
    # single linear layer followed by non-linearity

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),  # 4*n_embd: based on Attention paper implementation
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),   # projection layer going back into the residual pathway
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Head(nn.Module):

    # one head of self-attention
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # tril is not a parameter of the module, so we assign it to the module using register_buffer 

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape # batch, time, channels
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] ==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # weighted aggregation of values
        v = self.value(x)  # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    # multiple heads running in parallel

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # projection layer going back into the residual pathway
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatinating over the channel dim
        out = self.dropout(self.proj(out))
        return out

class Block(nn.Module):
    # transformer block: communication followed by computation

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dim, n_head: no. of heads
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # implementing residual connection type architecture
        x = x + self.ffwd(self.ln2(x))
        return x


# Bigram transformer
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd) # positional embedding
        # self.blocks = nn.Sequential (
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd),
        # )
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm

        #self.sa_heads = MultiHeadAttention(4, n_embd//4) # 4 heads of 8-dim self-attention
        #self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) # language model head

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,n_embd)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device))  # (T,n_embd)
        x = tok_emb + pos_emb # (B,T,n_embd)
        x = self.blocks(x) # apply one head of self-attention (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets == None:
            loss = None
        else:
            #pytorch expects logits in B,C,T dimensions i.e. Chanel dim as 2nd dim
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size token bcoz of positional_embedding 
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, loss = self(idx_cond)
            # focus only on last time step. The last elements in Time dim bcoz those are the predictions of what comes next
            # in this model, we are feeding the entire sequence of elements that came before 
            # then only looked at the last piece
            logits = logits[:, -1, :] # returns (B,C)
            # apply softmax to get probabilties
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
            

model= BigramLanguageModel()
m = model.to(device)

# create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    # sample batch of data
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
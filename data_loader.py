import tiktoken
import torch
import numpy as np
import os

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank=0, num_processes=1, split='train'):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = 'edu_fineweb10B'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f'no shards found for split {split}'
        #if master_process:
        #    print(f'found {len(shards)} shards for split {split}')

        # At initialization, load tokens from disk and store in memory ------------------- tinyShakespeare
        # with open('input.txt', 'r', encoding='utf-8') as f:
        #     text = f.read()
        # enc = tiktoken.get_encoding("gpt2")
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)
        # print(f"Loaded {len(self.tokens)} tokens into memory.")
        # print(f"1 epoch = {len(self.tokens) // (B * T)} batches.")
        # -------------------------------------------------------------------------

        # state, init at shard 0
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def get_batch(self):
        # get B sequences of length T
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # wrap around if we reach the end
        if self.current_position + (B * T * self.num_processes + 1) >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y
    
if __name__ == "__main__":
    dl = DataLoaderLite(B=4, T=8)
    x, y = dl.get_batch()
    print("x:", x)
    print("y:", y)
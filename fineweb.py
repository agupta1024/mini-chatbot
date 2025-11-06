'''
FineWeb-Edu dataset
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
'''

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)  # 100M tokens per shard, total 100 shards

# create local dir if it doesn't exist
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(local_dir, exist_ok=True)

# download the dataset
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# init tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']
def tokenize(doc):
    # tokenize a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # delimiter
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token out of range. dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# tokenize all documents in parallel and write to shards
# each is shard_size tokens (except possibly the last)
nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # buffer for current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, dataset, chunksize=16):
        ntokens = tokens.shape[0]
        # is there enough space in current shard?
        # if adding this would exceed shard size, write out current shard
        if token_count + ntokens > shard_size:
            # write out current shard
            split = 'val' if shard_index == 0 else 'train'
            shard_filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            # split the doc into whatever fits; remainings go to next shard
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(shard_filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # reset buffer. populate the next shard with remaining tokens
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder
        else:
            # add tokens to current shard buffer
            all_tokens_np[token_count:token_count+ntokens] = tokens
            token_count += ntokens
            # update progress bar
            if progress_bar is not None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index:05d}")
            progress_bar.update(ntokens)

    # write out any remaining tokens as the last shard
    if token_count > 0:
        split = 'val' if shard_index == 0 else 'train'
        shard_filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(shard_filename, all_tokens_np[:token_count])
ML Libraries: 
1. pytorch

train.py: Skeleton for Bigram Language model transformer

bigram_v2.py : Implements a simple bigram language model 
  - Uses input.txt as input for tokenisation
  - Implements a multihead attention block and feed-forward layers
  - Trains the model and generates new output tokens limited by max_new_tokens

train_gpt2.py: Implements Causal self-attention block
  - Adds Multi-layer perceptron
  - Get weights from pre-trained model GPT2LMHeadModel from transformers library
  - Two variants a. train the model from scratch b. Use trained model weights

import torch
import torch.nn as nn

from torch.nn import functional as F

torch.manual_seed(1996)

batch_size = 32
block_size = 8 # Context length
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'



with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
print(chars)
vocab_size = len(chars)
print(vocab_size)

# Encoding and Decoding a string to integers and back
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}
print(stoi)
print(itos)

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[elem] for elem in l])

print("Encoding \n")
print(encode("Before we proceed any further, hear me speak."))

print("Decoding \n")
print(decode([14, 43, 44, 53, 56, 43, 1, 61, 43, 1, 54, 56, 53, 41, 43, 43, 42, 1, 39, 52, 63, 1, 44, 59, 56, 58, 46, 43, 56, 6, 1, 46, 43, 39, 56, 1, 51, 43, 1, 57, 54, 43, 39, 49, 8]))


data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))

    """
    Consider the text of context length/block_ size 8: Before w
    Encoded x: [14, 43, 44, 53, 56, 43, 1, 61]
            y: [43, 44, 53, 56, 43, 1, 61, 43]
    For a GPT Model, we want context with every other word which comes "before" it. That is,

    For 'B', e is the output --> x: [14] y:[43]
    For 'Be', f is the output --> x: [14, 43] y:[44]
    For 'Bef', o is the output and so on --> x: [14, 43, 44] y:[53]
    Hence why targets are shifted by +1

    But for Bi-Gram Model, we don't consider the context, instead we just predict the next character
    B --> e
    e --> f
    f --> o
    .
    .
    """
    x = torch.stack([data[i: i + block_size] for i in idx])
    y = torch.stack([data[i+1: i + block_size + 1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y

get_batch("train")

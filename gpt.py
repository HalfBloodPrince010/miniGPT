import torch
import torch.nn as nn

from torch.nn import functional as F

torch.manual_seed(1996)

##########################
#
# HYPERPARAMETERS
#
##########################

batch_size = 32
block_size = 8 # Context length
learning_rate = 1e-3
max_iters = 6000
eval_interval = 300
eval_iters = 200
n_embd = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'


##########################
#
# Data
#
##########################
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# Encoding and Decoding a string to integers and back
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[elem] for elem in l])

# Dataset
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape)

# Train and Validation Split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


##########################
#
# DATALOADER
#
##########################

# Get a Batch of Data
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

@torch.no_grad()
def estimate_losses():
    out = {}
    # Put Model in Eval Mode
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # Put Model back in Train Mode
    model.train()
    return out


##########################
#
# Model
#
##########################

class Head(nn.Module):
    """
    One head of Self-Attention

    Attention is a communication mechanism. Each token in the context/T (B, T, C)
    Produces a Key and a Query.
    Key is what is produce.
    Query is what it is looking for.

    For example, say the context is looking for a adjective.
    It issue the query, does anyone has the "adjective" ..?
    Other word in the context might say "I have it".
    dot product between this Query and Key of the other token will be high.

    Value is information it provides. Its like saying "if you find the information you are looking
    then this is how much Value I provide.


    We usually consider the tokens before us, hence for that we use tril
    [
       1, 0, 0, 0, 0, 0, 0, 0
       0.5, 0.5, 0, 0, 0, 0, 0, 0
       0.33, 0.33, 0.33, 0, 0, 0, 0, 0
       .
       .
    ]

    But we want to learn how much important to give on previous context in a data driven way.
    Attention acheives that.
    
    """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # (T, T)

    def forward(self, x):
        # Input: (B, time-step, channels)
        # Output: (B, time-step, head_size)
        B, T, C = x.shape
        k = self.key(x) # (B, T, H_S)
        q = self.query(x) # (B, T, H_S)
        # compute attention scores ("affinities") - What Iam looking for (query), Who has that information (query)
        # k.transpose(-2, -1) transpose only last 2 dimensions -> (H_S, T)
        # k.shape[-1]**-0.5, normalizing using dimention to maintain the variable and preventing softmax from skewing.
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, H_S) @ (B, H_S, T) -> (B, T, T)
        # Basically "future" value which are 0 in the TRIL like above, make it -inf, works well for softmax.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        v = self.value(x) #(B, T, H_S)
        out = wei @ v # (B, T, T) @ (B, T, H_S) -> (B, T, H_S)
        return out


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # (batch_size, context/block_size, Channels) --> (B, T, C)
        # idx and targets will be (B, T) integers
        B, T = idx.shape
        tok_embd = self.token_embedding(idx)
        pos_embd = self.position_embedding(torch.arange(T, device=device)) # (T,C) arange--> 0, block_size-1, each T gets a pos embedding.
        x = tok_embd + pos_embd # (B,T,C)
        x = self.sa_head(x) # (B,T,C)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C) # For cross entropy calculation
            target = targets.view(B * T)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generates the new/next character.

        idx is same as above, made of (B, T)

        As mentioned above, for BiGram, we only look at current character, whose
        target is next character.
        (B, T)
        x = [14, 43, 44, 53, 56, 43, 1, 61]
        y = [43, 44, 53, 56, 43, 1, 61, 43]

        14 --> 43
        43 --> 44
        44 --> 53

        After, getting embeddings, (B, T, C), where each character's encoding is a
        64 bit vector.

        Here, we focus only one previous time step. That is to predict 'n' timestep
        we only look at (n - 1)

        In general in transformers, we look from 1 ... (n-1)
        """

        for _ in range(max_new_tokens):
            # We need only upto block_size, else pos_embedding will fail
            idx_cond = idx[:, -block_size:]

            # Get Model Predictions
            logits, loss = self(idx_cond)

            # Focus only on the last time step
            logits = logits[:, -1, :] # (B, C)

            probs = F.softmax(logits, dim=1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)

            # Append sampled idx_next to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx

##########################
#
# TRAINING
#
##########################

model = MiniGPT()
"""
# model a is in CPU
device = torch.device('cuda:0')
b = a.to(device)
# a and b are in GPU
# a and b point to the same model
"""
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


for iter in range(max_iters):
    # Evaluate Train and Validation Losses after certain iterations
    if iter % eval_interval == 0:
        losses = estimate_losses()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    # Forward Pass, evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

##########################
#
# SAMPLE GENERATION
#
##########################

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

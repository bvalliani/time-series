from pandas import *
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ------------- HYPERPARAMTERS -------------

N_BINS = 10000
BATCH_SIZE = 256
WINDOW_SIZE = 512
EMBED_SIZE = 512
N_BLOCKS = 6
N_HEADS = 8
HEAD_SIZE = 512
LEARNING_RATE = 1e-3
EPOCHS = 1000

# ----------------- DATA -----------------
# STEP 1: Read in CSV file.
# STEP 2: Extract the column with the temperature.
# STEP 3: Quantize the data.

train_file = read_csv("./data/DailyDelhiClimateTrain.csv")
test_file = read_csv("./data/DailyDelhiClimateTest.csv")

train_data = np.array(train_file['meantemp'].tolist())
test_data = np.array(test_file['meantemp'].tolist())

# Quantize training data.
min = np.floor(np.min(train_data)) - 1e-3
max = np.ceil(np.max(train_data))
step_size = (max - min) / N_BINS
bins = np.flip(np.arange(start=max, stop=min, step=-step_size))
train_data = np.digitize(train_data, bins)

# Quantize testing data.
min = np.floor(np.min(test_data)) - 1e-3
max = np.ceil(np.max(test_data))
step_size = (max - min) / N_BINS
bins = np.flip(np.arange(start=max, stop=min, step=-step_size))
test_data = np.digitize(test_data, bins)

train_data = torch.tensor(train_data, dtype=torch.long)
test_data = torch.tensor(test_data, dtype=torch.long)

# ------------------------ MODEL ------------------------

class AttentionHead(nn.Module):

    def __init__(self, input_size, output_size, use_mask):
        super().__init__()
        self.query_matrix = nn.Linear(input_size, output_size, bias=False)
        self.key_matrix = nn.Linear(input_size, output_size, bias=False)
        self.value_matrix = nn.Linear(input_size, output_size, bias=False)

        self.use_mask = use_mask

    def forward(self, input):
        queries = self.query_matrix(input)     # (batch_size, window_size, head_size)
        keys = self.key_matrix(input)          # (batch_size, window_size, head_size)
        values = self.value_matrix(input)      # (batch_size, window_size, head_size)
        head_size = keys.shape[-1]

        attention_matrix = head_size**-0.5 * (queries @ keys.transpose(-2, -1))     # (batch_size, window_size, window_size)
        if self.use_mask:
            attention_matrix = torch.tril(attention_matrix)
            attention_matrix = attention_matrix.masked_fill(attention_matrix == 0, float('-inf'))

        attention_matrix = F.softmax(attention_matrix, dim=-1)
        output = attention_matrix @ values      # (batch_size, window_size, head_size)
        return output

class MultiHeadedAttention(nn.Module):

    def __init__(self, n_heads, input_size, output_size, use_mask):
        super().__init__()
        self.attention_heads = nn.ModuleList([AttentionHead(input_size, output_size // n_heads, use_mask) for _ in range(n_heads)])
        self.projection = nn.Linear(output_size, output_size)

    def forward(self, input):
        output = torch.cat([head(input) for head in self.attention_heads], dim=-1)
        output = self.projection(output)
        return output

class FeedForward(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(input_size, 4*input_size),
            nn.ReLU(),
            nn.Linear(4*input_size, output_size)
        )

    def forward(self, x):
        output = self.feed_forward(x)
        return output
    
class TransformerBlock(nn.Module):

    def __init__(self, n_heads, embed_size, head_size):
        super().__init__()
        self.attention = MultiHeadedAttention(n_heads, embed_size, head_size, True)
        self.layer_norm1 = nn.LayerNorm(head_size)
        self.feed_forward = FeedForward(head_size, embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)

    def forward(self, input):
        output = input + self.attention(input)
        output = output + self.layer_norm1(output)
        output = output + self.feed_forward(output)
        output = output + self.layer_norm2(output)
        return output

class Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(N_BINS, EMBED_SIZE)
        self.position_encoding = nn.Embedding(WINDOW_SIZE, EMBED_SIZE)
        self.blocks = nn.Sequential(*[TransformerBlock(N_HEADS, EMBED_SIZE, HEAD_SIZE) for _ in range(N_BLOCKS)])
        self.feed_forward = nn.Linear(EMBED_SIZE, N_BINS)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, input, labels):
        token_embedding = self.token_embedding(input)
        position_encoding = self.position_encoding(torch.arange(input.shape[-1]))
        input = token_embedding + position_encoding

        input = self.blocks(input)
        logits = self.feed_forward(input)

        batch_size, _, embed_size = logits.shape
        logits = logits[:, -1, :].reshape(batch_size, embed_size)
        labels = torch.flatten(labels)
        loss = F.cross_entropy(logits, labels)

        return logits, loss


def batch():
    indices = torch.randint(low=0, high=len(train_data) - WINDOW_SIZE, size=(BATCH_SIZE,))
    examples = torch.stack([train_data[i:i+WINDOW_SIZE] for i in indices])
    labels = torch.tensor([train_data[i+WINDOW_SIZE] for i in indices], dtype=torch.long)
    return examples, labels


# Build and train the model.
GPTmodel = Transformer()
optimizer = torch.optim.AdamW(GPTmodel.parameters(), lr=LEARNING_RATE)

for e in range(EPOCHS):
    examples, labels = batch()
    logits, loss = GPTmodel(examples, labels)
    optimizer.zero_grad(set_to_none=True)
    torch.autograd.set_detect_anomaly(True)
    loss.backward()
    optimizer.step()

    print(f"Epoch: {e}, Loss: {loss}")
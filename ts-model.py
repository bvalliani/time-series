from pandas import *
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ----------------- DATA -----------------
# STEP 1: Read in CSV file.
# STEP 2: Extract the column with the temperature.
# STEP 3: Reshape data into training examples.
# STEP 4: Scale the data.
# STEP 5: Quantize the data.

# STEP 1.
train_file = read_csv("./data/DailyDelhiClimateTrain.csv")
test_file = read_csv("./data/DailyDelhiClimateTest.csv")

# STEP 2.
train_data = train_file['meantemp'].tolist()
test_data = test_file['meantemp'].tolist()

# STEP 3.
WINDOW_SIZE = 512
train_data = np.array(train_data)
train_examples = []
for i in range(len(train_data) - WINDOW_SIZE + 1):
    train_examples.append(train_data[i:i+WINDOW_SIZE])

train_examples = np.array(train_examples)

test_data = np.array(test_data)
test_examples = []
for i in range(len(test_data) - WINDOW_SIZE + 1):
    test_examples.append(test_data[i:i+WINDOW_SIZE])

test_examples = np.array(test_examples)

# STEP 4.
train_examples /= np.sum(train_examples, axis=1, keepdims=True)
test_examples /= np.sum(test_examples, axis=1, keepdims=True)

# STEP 5.

# Train data.
train_min = np.floor(np.min(train_examples)) - 1e-6
train_max = np.ceil(np.max(train_examples))

NUM_BINS = 10000
TRAIN_STEP_SIZE = (train_max - train_min) / NUM_BINS

bins = np.flip(np.arange(start=train_max, stop=train_min, step=-TRAIN_STEP_SIZE))
quantized_train_examples = np.digitize(train_examples, bins)

# Test data.
test_min = np.floor(np.min(test_examples)) - 1e-6
test_max = np.ceil(np.max(test_examples))

TEST_STEP_SIZE = (test_max - test_min) / NUM_BINS

bins = np.flip(np.arange(start=test_max, stop=test_min, step=-TEST_STEP_SIZE))
quantized_test_examples = np.digitize(test_examples, bins)

# ----------------- MODEL -----------------

# HYPERPARAMETERS
EMBED_SIZE = 100
HEAD_SIZE = 32
NUM_HEADS = 4
NUM_BLOCKS = 10

class AttentionHead(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.query_matrix = nn.Linear(EMBED_SIZE, head_size, bias=False)
        self.key_matrix = nn.Linear(EMBED_SIZE, head_size, bias=False)
        self.value_matrix = nn.Linear(EMBED_SIZE, head_size, bias=False)

    def forward(self, x):
        # x has shape (BATCH_SIZE, WINDOW_SIZE, EMBED_SIZE).
        queries = self.query_matrix(x) # (BATCH_SIZE, WINDOW_SIZE, HEAD_SIZE)
        keys = self.key_matrix(x) # (BATCH_SIZE, WINDOW_SIZE, HEAD_SIZE)
        values = self.value_matrix(x) # (BATCH_SIZE, WINDOW_SIZE, HEAD_SIZE)

        attn_matrix = torch.tril(self.head_size**-0.5 * (queries @ keys.transpose(-2, -1))) # (BATCH_SIZE, WINDOW_SIZE, WINDOW_SIZE)
        attn_matrix = torch.Tensor.masked_fill(attn_matrix == 0, -np.inf)
        attn_matrix = F.softmax(attn_matrix, dim=-1)
        output = attn_matrix @ values # (BATCH_SIZE, WINDOW_SIZE, HEAD_SIZE)
        return output

class MultiHeadedAttention(nn.Module):
    
    def __init__(self, head_size, num_heads):
        super().__init__()
        self.attn_heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])

    def forward(self, x):
        output = torch.cat([head(x) for head in self.attn_heads], dim=-1)
        return output
    
class FeedForward(nn.Module):

    def __init__(self, embed_size):
        super().__init__()
        self.feed_forward = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        output = self.feed_forward(x)
        return output
    
class TransformerBlock(nn.Module):

    def __init__(self, embed_size, head_size, num_heads):
        super.__init__()
        self.attn = MultiHeadedAttention(head_size // num_heads, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x += self.attn(x)
        x += self.layer_norm1(x)
        x += self.feed_forward(x)
        x += self.layer_norm2(x)

        return x
    
class Transformer(nn.Module):

    def __init__(self):
        self.token_embedding = nn.Embedding(NUM_BINS, EMBED_SIZE)
        self.position_encoding = nn.Embedding(WINDOW_SIZE, EMBED_SIZE)
        self.blocks = nn.Sequential(*[TransformerBlock(EMBED_SIZE, HEAD_SIZE, NUM_HEADS) for _ in range(NUM_BLOCKS)])
        self.feed_forward = nn.Linear(EMBED_SIZE, NUM_BINS)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, y):
        token_embedding = self.token_embedding(x)
        position_encoding = self.position_encoding(x)

        x = token_embedding + position_encoding
        x = self.blocks(x)
        logits = self.feed_forward(x)

        batch_size, window_size, embed_size = logits.shape
        logits = torch.reshape(logits, (batch_size*window_size, embed_size))
        targets = torch.flatten(targets)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

GPTmodel = Transformer()
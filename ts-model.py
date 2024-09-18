from pandas import *
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ------------- HYPERPARAMTERS -------------

N_CLASSES = 10000
BATCH_SIZE = ...
WINDOW_SIZE = 512
EMBED_SIZE = 100
N_BLOCKS = 6
N_HEADS = 8
HEAD_SIZE = 512
LEARNING_RATE = 1e-3
EPOCHS = ...

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

# Training data.
train_data = np.array(train_data)
train_examples = []
for i in range(len(train_data) - WINDOW_SIZE + 1):
    train_examples.append(train_data[i:i+WINDOW_SIZE])

train_examples = np.array(train_examples)

# Testing data.
test_data = np.array(test_data)
test_examples = []
for i in range(len(test_data) - WINDOW_SIZE + 1):
    test_examples.append(test_data[i:i+WINDOW_SIZE])

test_examples = np.array(test_examples)

# STEP 4.
train_examples /= np.sum(train_examples, axis=1, keepdims=True)
test_examples /= np.sum(test_examples, axis=1, keepdims=True)

# STEP 5.

# Training data.
train_min = np.floor(np.min(train_examples)) - 1e-6
train_max = np.ceil(np.max(train_examples))
step_size = (train_max - train_min) / N_CLASSES

bins = np.flip(np.arange(start=train_max, stop=train_min, step=-step_size))
quantized_train_examples = np.digitize(train_examples, bins)

# Testing data.
test_min = np.floor(np.min(test_examples)) - 1e-6
test_max = np.ceil(np.max(test_examples))
step_size = (test_max - test_min) / N_CLASSES

bins = np.flip(np.arange(start=test_max, stop=test_min, step=-step_size))
quantized_test_examples = np.digitize(test_examples, bins)

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
            attention_matrix = torch.Tensor.masked_fill_(attention_matrix == 0, float('-inf'))

        attention_matrix = F.softmax(attention_matrix, dim=-1)
        output = attention_matrix @ values      # (batch_size, window_size, head_size)
        return output

class MultiHeadedAttention(nn.Module):

    def __init__(self, n_heads, input_size, output_size, use_mask):
        super().__init__()
        self.attention_heads = nn.ModuleList([AttentionHead(input_size, output_size // n_heads, use_mask) for _ in range(n_heads)])
        self.projection = nn.Linear(n_heads*output_size, n_heads*output_size)

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
        output += self.layer_norm1(output)
        output += self.feed_forward(output)
        output += self.layer_norm2(output)
        return output

class Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(N_CLASSES, EMBED_SIZE)
        self.position_encoding = nn.Embedding(WINDOW_SIZE, EMBED_SIZE)
        self.blocks = nn.Sequential(*[TransformerBlock(N_HEADS, EMBED_SIZE, HEAD_SIZE) for _ in range(N_BLOCKS)])
        self.feed_forward = nn.Linear(EMBED_SIZE, N_CLASSES)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, input, labels):
        token_embedding = self.token_embedding(input)
        position_encoding = self.position_encoding(input)

        input = token_embedding + position_encoding
        input = self.blocks(input)
        logits = self.feed_forward(input)

        batch_size, window_size, embed_size = logits.shape
        logits = torch.reshape(logits, (batch_size*window_size, embed_size))
        labels = torch.flatten(labels)
        loss = F.cross_entropy(logits, labels)

        return logits, loss

# Build and train the model.
GPTmodel = Transformer()
optimizer = torch.optim.AdamW(GPTmodel.parameters(), lr=LEARNING_RATE)

for _ in range(EPOCHS):
    logits, loss = GPTmodel(train_data, ...)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 8
batch_size = 32
learning_rate = 1e-2
eval_interval = 300
eval_iters= 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

with open('input.txt', 'r', encoding='utf-8') as f:
          text = f.read()
caracteres = sorted(set(text))
vocabulary_size = len(caracteres)


string_to_int = { ch:i for i,ch in enumerate(caracteres) }
int_to_string = { i:ch for i,ch in enumerate(caracteres) }
encode = lambda s: [string_to_int[c] for c in s] #codifica e decodifica strings (array de char) para ints
decode = lambda l: ''.join([int_to_string[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)

training_size = int(0.9*len(data))
training_data = data[:training_size]
validation_data = data[training_size:]

torch.manual_seed(1337)


def batch(split):
    data = training_data if split == 'train' else validation_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])     #inputs para o transformer
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #inputs para cada target
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimaste_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


class BigramLanguageModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, vocabulary_size) #mapeia tokens para vetores de embedding

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx) #calcula os logits (previsões) e a perda para a sequência de palavras idx
            logits = logits[:,-1,:]
            probability = F.softmax(logits, dim=1) #transforma em probabilidade
            idx_next = torch.multinomial(probability, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocabulary_size)
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


for iter in range(10000):
    if iter % eval_interval == 0:
        losses = estimaste_loss()

    xb, yb = batch('train')
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1,1), dtype=torch.long, device=device)

print(loss.item())

print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
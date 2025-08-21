#!/usr/bin/env python3
"""
All-in-one Tiny GPT trainer + stable chat interface.

Usage:
1. Train:
   python tiny_gpt_chat.py --data input.txt --epochs 5
2. Chat after training:
   python tiny_gpt_chat.py --ckpt ckpt.pt --chat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse

# ---------------------------
# Tokenizer
# ---------------------------
class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for ch,i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s: str):
        return torch.tensor([self.stoi[c] for c in s if c in self.stoi], dtype=torch.long)

    def decode(self, ids):
        return "".join([self.itos[int(i)] for i in ids])

# ---------------------------
# Model
# ---------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size=128, n_layer=1, n_head=1, n_embd=32, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        B,T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=50):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v,_ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

# ---------------------------
# Training helper
# ---------------------------
def get_batch(data_tensor, block_size, batch_size, device):
    if data_tensor.size(0) <= block_size:
        block_size = data_tensor.size(0)-1
    ix = torch.randint(0, data_tensor.size(0)-block_size-1, (batch_size,))
    x = torch.stack([data_tensor[i:i+block_size] for i in ix])
    y = torch.stack([data_tensor[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

def train_model(cfg):
    torch.manual_seed(cfg.seed)
    with open(cfg.data, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    data_tensor = tokenizer.encode(text)
    n = int(0.9*len(data_tensor))
    train_data = data_tensor[:n]
    val_data = data_tensor[n:]

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device.startswith('cuda')))

    for epoch in range(cfg.epochs):
        steps = max(1, len(train_data)//(cfg.batch_size*cfg.block_size))
        for step in range(steps):
            xb, yb = get_batch(train_data, cfg.block_size, cfg.batch_size, cfg.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.device.startswith('cuda'))):
                _, loss = model(xb, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % 50 == 0:
                print(f"Epoch {epoch+1}/{cfg.epochs} Step {step}/{steps} | Loss: {loss.item():.3f}")

    # save checkpoint
    torch.save({
        'model': model.state_dict(),
        'stoi': tokenizer.stoi,
        'itos': tokenizer.itos,
        'config': {
            'block_size': cfg.block_size,
            'n_layer': cfg.n_layer,
            'n_head': cfg.n_head,
            'n_embd': cfg.n_embd,
            'dropout': cfg.dropout
        }
    }, cfg.ckpt)
    print(f"Checkpoint saved to {cfg.ckpt}")

# ---------------------------
# Chat interface
# ---------------------------
def chat(cfg):
    if not os.path.exists(cfg.ckpt):
        print("Checkpoint not found. Train first!")
        return

    ck = torch.load(cfg.ckpt, map_location=cfg.device)
    stoi, itos = ck['stoi'], ck['itos']
    vocab_size = len(itos)
    model_cfg = ck['config']

    tokenizer = CharTokenizer("")
    tokenizer.stoi = stoi
    tokenizer.itos = itos
    tokenizer.vocab_size = vocab_size

    model = TinyGPT(
        vocab_size=vocab_size,
        block_size=model_cfg['block_size'],
        n_layer=model_cfg['n_layer'],
        n_head=model_cfg['n_head'],
        n_embd=model_cfg['n_embd'],
        dropout=model_cfg['dropout']
    ).to(cfg.device)
    model.load_state_dict(ck['model'])
    model.eval()

    encode = tokenizer.encode
    decode = tokenizer.decode
    block_size = model_cfg['block_size']

    print("---- Chat with GPT (type 'quit' to exit) ----")
    context = torch.randint(0, vocab_size, (1,1), device=cfg.device)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            print("Please type something.")
            continue

        tokens = [stoi[c] for c in user_input if c in stoi]
        if not tokens:
            print("No valid characters in input.")
            continue

        tokens = torch.tensor([tokens], dtype=torch.long, device=cfg.device)
        context = torch.cat([context, tokens], dim=1)
        context = context[:, -block_size:]

        out = model.generate(context, max_new_tokens=100, temperature=1.0, top_k=50)
        reply = decode(out[0].tolist())
        reply_text = reply[len(user_input):].strip()
        print("AI:", reply_text)

        # FIXED: unsqueeze instead of wrapping in [ ... ]
        reply_tokens = encode(reply_text).unsqueeze(0).to(cfg.device)
        context = torch.cat([context, reply_tokens], dim=1)
        context = context[:, -block_size:]

# ---------------------------
# Config / CLI
# ---------------------------
class Config:
    def __init__(self):
        self.data = ""
        self.block_size = 128
        self.batch_size = 4
        self.n_layer = 1
        self.n_head = 1
        self.n_embd = 32
        self.dropout = 0.2
        self.lr = 3e-4
        self.epochs = 5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ckpt = "ckpt.pt"
        self.seed = 1337
        self.chat = False

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, help="Training text file")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--ckpt", type=str, default="ckpt.pt")
    p.add_argument("--chat", action="store_true")
    args = p.parse_args()
    cfg = Config()
    cfg.data = args.data
    cfg.epochs = args.epochs
    cfg.ckpt = args.ckpt
    cfg.chat = args.chat
    return cfg

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    cfg = parse_args()
    if cfg.chat:
        chat(cfg)
    else:
        if not cfg.data:
            print("Provide --data for training.")
        else:
            train_model(cfg)

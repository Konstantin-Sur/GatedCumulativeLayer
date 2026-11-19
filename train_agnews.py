"""Train + compare GatedCumulativeLayer vs nn.MultiheadAttention on AG News.

Tokenization is the simplest possible: lowercase, split on non-alphanumeric.
Vocab is built from the (subsampled) training set. Sequence length is short
(64 tokens) because AG News titles/descriptions are short.

The model is identical for both variants except for the attention sublayer:

    emb -> positional -> N x [block(attention-replacement)] -> mean-pool -> linear

Runs are short (configurable; default ~2 min each). Prints a final summary
table so the comparison is one glance.
"""

import csv
import math
import os
import re
import sys
import time
import random
from collections import Counter

import torch
import torch.nn as nn

sys.path.insert(0, "/home/z/my-project/scripts")
from gated_cumulative import (
    GatedCumulativeLayer,
    TransformerBlock,
    MultiheadAttentionAdapter,
)


DATA_DIR = "/home/z/my-project/data"
OUT_DIR = "/home/z/my-project/download"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")

# Hyperparameters -- tuned for a "few-minute" run on CPU.
SEED = 0
N_TRAIN = 12000       # 12000 / 120000 of AG News train
N_TEST = 2000        # 2000 / 7600 of AG News test
MAX_LEN = 64
MIN_FREQ = 2
VOCAB_CAP = 20000

D_MODEL = 128
N_LAYERS = 2
FFN_MULT = 4
DROPOUT = 0.1
NHEAD = 4             # for the MHA baseline

BATCH_SIZE = 64
EPOCHS = 2
LR = 3e-4
WEIGHT_DECAY = 1e-4

random.seed(SEED)
torch.manual_seed(SEED)


# -------------------- data --------------------

_TOK_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str):
    return _TOK_RE.findall(text.lower())


def read_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if len(row) < 3:
                continue
            label = int(row[0]) - 1  # 1..4 -> 0..3
            text = row[1] + " " + row[2]
            rows.append((text, label))
    return rows


def build_vocab(texts, min_freq=MIN_FREQ, cap=VOCAB_CAP):
    c = Counter()
    for t in texts:
        c.update(tokenize(t))
    items = [(w, n) for w, n in c.items() if n >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    items = items[:cap]
    # 0: pad, 1: unk, 2: bos, 3: eos
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    for w, _ in items:
        vocab[w] = len(vocab)
    return vocab


def encode(text, vocab, max_len=MAX_LEN):
    ids = [2] + [vocab.get(w, 1) for w in tokenize(text)] + [3]
    if len(ids) > max_len:
        ids = ids[:max_len]
    return ids


def collate(batch, vocab):
    # batch: list of (text, label)
    encs = [encode(t, vocab) for t, _ in batch]
    labels = torch.tensor([l for _, l in batch], dtype=torch.long)
    maxlen = max(len(e) for e in encs)
    ids = torch.full((len(batch), maxlen), vocab["<pad>"], dtype=torch.long)
    mask = torch.zeros((len(batch), maxlen), dtype=torch.bool)
    for i, e in enumerate(encs):
        ids[i, : len(e)] = torch.tensor(e, dtype=torch.long)
        mask[i, : len(e)] = True
    return ids, mask, labels


# -------------------- model --------------------

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_classes, attn_factory,
                 ffn_mult=FFN_MULT, dropout=DROPOUT, max_len=MAX_LEN):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len + 1, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, attn_factory(), ffn_mult, dropout)
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, ids, mask):
        B, T = ids.shape
        positions = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        x = self.tok(ids) + self.pos(positions)
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # mean-pool over non-pad tokens
        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


# -------------------- training --------------------

def evaluate(model, loader, vocab, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            ids, mask, labels = collate(batch, vocab)
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            logits = model(ids, mask)
            loss = nn.functional.cross_entropy(logits, labels)
            loss_sum += loss.item() * labels.numel()
            pred = logits.argmax(-1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return correct / total, loss_sum / max(total, 1)


def train_one(model_name, attn_factory, train_rows, test_rows, vocab,
              device, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, log_every=50):
    print(f"\n=== Training {model_name} ===")
    model = TextClassifier(
        vocab_size=len(vocab),
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_classes=4,
        attn_factory=attn_factory,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_attn_params = sum(
        p.numel() for blk in model.blocks for p in blk.attn.parameters()
    )
    print(f"  total params: {n_params:,}  attn-sublayer params: {n_attn_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    def batch_iter(rows, bs, shuffle=True):
        idx = list(range(len(rows)))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), bs):
            yield [rows[j] for j in idx[i:i + bs]]

    t_start = time.perf_counter()
    history = []
    step = 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        running_n = 0
        for batch in batch_iter(train_rows, batch_size):
            ids, mask, labels = collate(batch, vocab)
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            logits = model(ids, mask)
            loss = nn.functional.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * labels.numel()
            running_n += labels.numel()
            step += 1
            if step % log_every == 0:
                tr_loss = running / running_n
                tr_acc = (logits.argmax(-1) == labels).float().mean().item()
                elapsed = time.perf_counter() - t_start
                print(f"  epoch {epoch+1} step {step:4d} "
                      f"tr_loss {tr_loss:.3f} tr_acc {tr_acc:.3f} "
                      f"({elapsed:.0f}s)")
                history.append((epoch, step, tr_loss, tr_acc, elapsed))
                running = 0.0
                running_n = 0
        # eval after each epoch
        te_acc, te_loss = evaluate(model, _iter_batches(test_rows, 128), vocab, device)
        print(f"  >> epoch {epoch+1} done: test_acc {te_acc:.4f} test_loss {te_loss:.3f}")
    total_time = time.perf_counter() - t_start
    final_acc, final_loss = evaluate(model, _iter_batches(test_rows, 128), vocab, device)
    print(f"  total train time: {total_time:.1f}s   final test_acc: {final_acc:.4f}")
    return {
        "name": model_name,
        "test_acc": final_acc,
        "test_loss": final_loss,
        "train_time_s": total_time,
        "n_params": n_params,
        "n_attn_params": n_attn_params,
        "history": history,
    }


def _iter_batches(rows, bs):
    for i in range(0, len(rows), bs):
        yield rows[i:i + bs]


def main():
    print("loading AG News...")
    train_all = read_csv(os.path.join(DATA_DIR, "ag_news_train.csv"))
    test_all = read_csv(os.path.join(DATA_DIR, "ag_news_test.csv"))
    print(f"  train rows: {len(train_all)}, test rows: {len(test_all)}")

    random.shuffle(train_all)
    random.shuffle(test_all)
    train_rows = train_all[:N_TRAIN]
    test_rows = test_all[:N_TEST]
    print(f"  using train={len(train_rows)}, test={len(test_rows)}")

    print("building vocab...")
    vocab = build_vocab([t for t, _ in train_rows])
    print(f"  vocab size: {len(vocab)}")

    # Token length stats (informational)
    lens = [len(tokenize(t)) for t, _ in train_rows[:2000]]
    print(f"  token length: mean {sum(lens)/len(lens):.1f}, "
          f"p95 {sorted(lens)[int(0.95*len(lens))]}, max {max(lens)}")

    results = []
    # ---- Variant 1: standard MultiheadAttention ----
    results.append(train_one(
        "MHA (nn.MultiheadAttention)",
        lambda: MultiheadAttentionAdapter(d_model=D_MODEL, nhead=NHEAD, dropout=0.0),
        train_rows, test_rows, vocab, DEVICE,
    ))

    # ---- Variant 2: our GatedCumulativeLayer ----
    results.append(train_one(
        "GatedCumulative (ours)",
        lambda: GatedCumulativeLayer(d_model=D_MODEL),
        train_rows, test_rows, vocab, DEVICE,
    ))

    # ---- Summary ----
    print("\n" + "=" * 78)
    print("FINAL COMPARISON")
    print("=" * 78)
    print(f"{'model':32s} {'test_acc':>9s} {'test_loss':>10s} "
          f"{'time_s':>8s} {'params':>10s}")
    for r in results:
        print(f"{r['name']:32s} {r['test_acc']:9.4f} {r['test_loss']:10.3f} "
              f"{r['train_time_s']:8.1f} {r['n_params']:10,}")

    # Save history + summary to download/
    summary_path = os.path.join(OUT_DIR, "ag_news_results.txt")
    with open(summary_path, "w") as f:
        f.write("AG News comparison: GatedCumulativeLayer vs nn.MultiheadAttention\n")
        f.write(f"train_rows={N_TRAIN} test_rows={N_TEST} "
                f"d_model={D_MODEL} n_layers={N_LAYERS} "
                f"max_len={MAX_LEN} epochs={EPOCHS} batch={BATCH_SIZE} lr={LR}\n\n")
        for r in results:
            f.write(f"{r['name']}\n")
            f.write(f"  test_acc       = {r['test_acc']:.4f}\n")
            f.write(f"  test_loss      = {r['test_loss']:.3f}\n")
            f.write(f"  train_time (s) = {r['train_time_s']:.1f}\n")
            f.write(f"  total params   = {r['n_params']:,}\n")
            f.write(f"  attn params    = {r['n_attn_params']:,}\n\n")
            f.write("  step history: epoch step tr_loss tr_acc elapsed_s\n")
            for h in r["history"]:
                f.write(f"    {h[0]+1} {h[1]} {h[2]:.4f} {h[3]:.4f} {h[4]:.1f}\n")
            f.write("\n")
    print(f"\nresults saved to: {summary_path}")


if __name__ == "__main__":
    main()

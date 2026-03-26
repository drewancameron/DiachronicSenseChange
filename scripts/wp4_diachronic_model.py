#!/usr/bin/env python3
"""
WP4: Diachronic sense change model with FiLM conditioning.

Architecture:
  aristoBERTo → target token embedding (768-dim)
  → FiLM modulation by period (learned scale + shift per period)
  → FiLM modulation by register (learned scale + shift per register)
  → sense classification head (per lemma)

The key insight: the same Greek context produces a *different*
effective embedding depending on when it was written and what
register it's in. The model learns:
  - how period transforms sense (diachronic change)
  - how register transforms sense (stylistic variation)
  - the base sense from Greek context alone

This directly tests H2 and H3 from the project plan:
  H2: contextual models will recover structured relationships
      among senses, registers, and historical layers
  H3: register must be modelled explicitly to avoid confusing
      semantic persistence with archaizing usage
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

DATA_DIR = Path(__file__).parent.parent / "models" / "data"
MODEL_DIR = Path(__file__).parent.parent / "models" / "checkpoints"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.

    Given a conditioning variable (period or register), learns
    scale (gamma) and shift (beta) parameters that transform
    the embedding: output = gamma * input + beta

    Each period/register gets its own learned transformation,
    so the model can represent how meaning shifts across time
    or register.
    """
    def __init__(self, n_conditions: int, hidden_size: int):
        super().__init__()
        # Learnable scale and shift per condition
        self.gamma = nn.Embedding(n_conditions, hidden_size)
        self.beta = nn.Embedding(n_conditions, hidden_size)
        # Initialize gamma near 1, beta near 0 (identity transform)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x, condition_ids):
        """
        x: (batch, hidden_size) — the embedding to modulate
        condition_ids: (batch,) — integer IDs for the condition
        """
        gamma = self.gamma(condition_ids)  # (batch, hidden)
        beta = self.beta(condition_ids)    # (batch, hidden)
        return gamma * x + beta


class DiachronicSenseModel(nn.Module):
    """
    Diachronic sense classification with FiLM-conditioned embeddings.

    The model extracts a target token embedding from aristoBERTo,
    then applies FiLM modulation conditioned on:
      1. Period (homeric, archaic, classical, hellenistic, imperial, koine)
      2. Register (unmarked, poetic_elevated, archaizing, etc.)

    The modulated embedding is then classified per-lemma.

    This means the model can answer questions like:
    "Given this Greek context, if it's from the Classical period
     in unmarked register, what sense is this?"
    vs.
    "Same Greek context, but from the Imperial period in
     archaizing register — now what sense?"
    """
    def __init__(self, encoder, hidden_size, sense_maps,
                 n_periods, n_registers, freeze_layers=8):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size

        # Freeze lower layers
        for i, layer in enumerate(encoder.encoder.layer):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False
        for p in encoder.embeddings.parameters():
            p.requires_grad = False

        # Projection from BERT hidden to a smaller working space
        self.project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        work_dim = hidden_size // 2  # 384

        # FiLM conditioning layers
        self.period_film = FiLMLayer(n_periods, work_dim)
        self.register_film = FiLMLayer(n_registers, work_dim)

        # Layer norm after conditioning (stabilises training)
        self.post_film_norm = nn.LayerNorm(work_dim)

        # Per-lemma sense classification heads
        self.sense_heads = nn.ModuleDict()
        for lemma, smap in sense_maps.items():
            key = lemma.encode("utf-8").hex()
            self.sense_heads[key] = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(work_dim, work_dim // 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(work_dim // 2, len(smap)),
            )

        self.lemma_key_map = {l: l.encode("utf-8").hex() for l in sense_maps}

    def forward(self, input_ids, attention_mask, target_pos,
                period_ids, register_ids):
        # Get contextual embeddings from BERT
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        # Extract target token embedding
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        target_pos = target_pos.clamp(0, hidden.size(1) - 1)
        target_emb = hidden[batch_idx, target_pos]  # (batch, 768)

        # Project to working dimension
        projected = self.project(target_emb)  # (batch, 384)

        # Apply FiLM conditioning
        # Period modulation: "how does the period transform this embedding?"
        modulated = self.period_film(projected, period_ids)
        # Register modulation: "how does the register further transform it?"
        modulated = self.register_film(modulated, register_ids)
        # Normalise
        modulated = self.post_film_norm(modulated)

        return {
            "base_embedding": projected,       # before conditioning
            "conditioned_embedding": modulated, # after conditioning
            "target_embedding": target_emb,    # raw BERT output
        }

    def predict_sense(self, conditioned_emb, lemma):
        key = self.lemma_key_map.get(lemma)
        if key and key in self.sense_heads:
            return self.sense_heads[key](conditioned_emb)
        return None


class DiachronicDataset(Dataset):
    def __init__(self, examples, tokenizer, period_map, register_map,
                 max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.period_map = period_map
        self.register_map = register_map
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex["greek_context"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Find target position
        tokens = self.tokenizer.tokenize(ex["greek_context"])
        surface_tokens = self.tokenizer.tokenize(ex["surface_form"])
        target_pos = 1
        if surface_tokens:
            for i, t in enumerate(tokens):
                if t == surface_tokens[0]:
                    target_pos = i + 1
                    break
        target_pos = min(target_pos, self.max_length - 1)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "target_pos": target_pos,
            "sense_label": ex["sense_id"],
            "lemma": ex["lemma"],
            "period_id": self.period_map.get(ex["period"], 0),
            "register_id": self.register_map.get(ex["register"], 0),
        }


def train(epochs=20, batch_size=16, lr=2e-5, freeze_layers=8):
    print(f"Device: {DEVICE}")

    # Load data
    with open(DATA_DIR / "train.json") as f:
        train_data = json.load(f)
    with open(DATA_DIR / "val.json") as f:
        val_data = json.load(f)
    with open(DATA_DIR / "sense_labels.json") as f:
        sense_maps = json.load(f)

    # Build label maps
    all_periods = sorted(set(ex["period"] for ex in train_data + val_data))
    all_registers = sorted(set(ex["register"] for ex in train_data + val_data))
    period_map = {p: i for i, p in enumerate(all_periods)}
    register_map = {r: i for i, r in enumerate(all_registers)}

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Periods: {all_periods}")
    print(f"Registers: {all_registers}")
    print(f"Senses: {sum(len(v) for v in sense_maps.values())} across {len(sense_maps)} lemmata")

    # Load encoder
    print("\nLoading aristoBERTo...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("Jacobo/aristoBERTo")
    encoder = AutoModel.from_pretrained("Jacobo/aristoBERTo")

    model = DiachronicSenseModel(
        encoder=encoder,
        hidden_size=encoder.config.hidden_size,
        sense_maps=sense_maps,
        n_periods=len(all_periods),
        n_registers=len(all_registers),
        freeze_layers=freeze_layers,
    )
    model = model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total_params:,} total "
          f"({trainable*100//total_params}%)")

    # Datasets
    train_ds = DiachronicDataset(train_data, tokenizer, period_map, register_map)
    val_ds = DiachronicDataset(val_data, tokenizer, period_map, register_map)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        n_total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            target_pos = batch["target_pos"].to(DEVICE)
            period_ids = batch["period_id"].to(DEVICE)
            register_ids = batch["register_id"].to(DEVICE)

            outputs = model(input_ids, attention_mask, target_pos,
                           period_ids, register_ids)

            loss = torch.tensor(0.0, device=DEVICE)
            for i in range(len(batch["lemma"])):
                logits = model.predict_sense(
                    outputs["conditioned_embedding"][i:i+1],
                    batch["lemma"][i],
                )
                if logits is not None:
                    target = torch.tensor([batch["sense_label"][i]],
                                         device=DEVICE)
                    loss = loss + criterion(logits, target)
                    correct += (logits.argmax(-1) == target).sum().item()
                    n_total += 1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        train_acc = correct / n_total if n_total > 0 else 0

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                target_pos = batch["target_pos"].to(DEVICE)
                period_ids = batch["period_id"].to(DEVICE)
                register_ids = batch["register_id"].to(DEVICE)

                outputs = model(input_ids, attention_mask, target_pos,
                               period_ids, register_ids)

                for i in range(len(batch["lemma"])):
                    logits = model.predict_sense(
                        outputs["conditioned_embedding"][i:i+1],
                        batch["lemma"][i],
                    )
                    if logits is not None:
                        target = batch["sense_label"][i]
                        pred = logits.argmax(-1).item()
                        val_correct += (pred == target)
                        val_total += 1

        val_acc = val_correct / val_total if val_total > 0 else 0

        print(f"Epoch {epoch+1:>2d}/{epochs}: "
              f"loss={total_loss/len(train_loader):.4f} "
              f"train={train_acc:.3f} val={val_acc:.3f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "sense_maps": sense_maps,
                "period_map": period_map,
                "register_map": register_map,
                "epoch": epoch,
                "val_acc": val_acc,
                "model_type": "diachronic_film",
            }, MODEL_DIR / "diachronic_best.pt")
            print(f"  → Saved (val={val_acc:.3f})")

    print(f"\nBest val accuracy: {best_val_acc:.3f}")

    # Analyse FiLM parameters
    print(f"\n{'='*60}")
    print("FiLM PARAMETER ANALYSIS")
    print(f"{'='*60}")

    model.eval()
    model = model.cpu()

    # Period FiLM: how does each period transform the embedding?
    print("\n── Period Modulation (gamma scale factors) ──")
    period_gammas = model.period_film.gamma.weight.detach().numpy()
    period_betas = model.period_film.beta.weight.detach().numpy()
    inv_period = {v: k for k, v in period_map.items()}

    for pid in range(len(all_periods)):
        gamma = period_gammas[pid]
        beta = period_betas[pid]
        # How much does this period deviate from identity?
        gamma_dev = np.abs(gamma - 1.0).mean()
        beta_mag = np.abs(beta).mean()
        print(f"  {inv_period[pid]:>12s}: gamma_dev={gamma_dev:.4f} beta_mag={beta_mag:.4f}")

    # Which periods are most similar/different?
    print("\n── Period Similarity (cosine of gamma vectors) ──")
    from numpy.linalg import norm
    for i in range(len(all_periods)):
        for j in range(i+1, len(all_periods)):
            cos = np.dot(period_gammas[i], period_gammas[j]) / (
                norm(period_gammas[i]) * norm(period_gammas[j])
            )
            print(f"  {inv_period[i]:>12s} ↔ {inv_period[j]:<12s}: {cos:.4f}")

    # Register FiLM
    print("\n── Register Modulation ──")
    reg_gammas = model.register_film.gamma.weight.detach().numpy()
    reg_betas = model.register_film.beta.weight.detach().numpy()
    inv_reg = {v: k for k, v in register_map.items()}

    for rid in range(len(all_registers)):
        gamma = reg_gammas[rid]
        beta = reg_betas[rid]
        gamma_dev = np.abs(gamma - 1.0).mean()
        beta_mag = np.abs(beta).mean()
        print(f"  {inv_reg[rid]:>16s}: gamma_dev={gamma_dev:.4f} beta_mag={beta_mag:.4f}")

    return best_val_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--freeze-layers", type=int, default=8)
    args = parser.parse_args()
    train(**vars(args))

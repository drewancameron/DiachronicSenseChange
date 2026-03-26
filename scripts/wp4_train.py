#!/usr/bin/env python3
"""
WP4 Tasks 4.2-4.4: Fine-tune aristoBERTo for sense classification.

Architecture:
  aristoBERTo (frozen lower layers) → target token embedding →
  classification head per lemma → sense prediction

Also predicts register and period as auxiliary tasks (Task 4.4).

Designed for M3 MacBook Pro with MPS acceleration.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

DATA_DIR = Path(__file__).parent.parent / "models" / "data"
MODEL_DIR = Path(__file__).parent.parent / "models" / "checkpoints"

# Detect device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class SenseDataset(Dataset):
    """Dataset for sense classification from Greek context."""

    def __init__(self, examples: list, tokenizer, max_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # Tokenize Greek context
        encoding = self.tokenizer(
            ex["greek_context"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Find which token position corresponds to the target word
        # Use a simple heuristic: find the subword token(s) that match
        # the surface form, then use the first one
        tokens = self.tokenizer.tokenize(ex["greek_context"])
        surface_tokens = self.tokenizer.tokenize(ex["surface_form"])

        target_pos = 1  # default to first real token (after [CLS])
        if surface_tokens:
            first_sub = surface_tokens[0]
            for i, t in enumerate(tokens):
                if t == first_sub:
                    target_pos = i + 1  # +1 for [CLS]
                    break

        target_pos = min(target_pos, self.max_length - 1)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "target_pos": target_pos,
            "sense_label": ex["sense_id"],
            "lemma": ex["lemma"],
            "period": ex["period"],
            "register": ex["register"],
        }


class SenseClassifier(nn.Module):
    """
    Multi-task sense classifier.

    Uses the target token's contextual embedding from aristoBERTo
    to predict:
      1. Word sense (per-lemma classification head)
      2. Register (shared head)
      3. Period (shared head)
    """

    def __init__(self, encoder, hidden_size: int, sense_maps: dict,
                 register_labels: list, period_labels: list,
                 freeze_layers: int = 8):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size

        # Freeze lower transformer layers
        for i, layer in enumerate(encoder.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        # Also freeze embeddings
        for param in encoder.embeddings.parameters():
            param.requires_grad = False

        # Per-lemma sense heads
        self.sense_heads = nn.ModuleDict()
        for lemma, smap in sense_maps.items():
            n_senses = len(smap)
            # Use lemma transliteration as key (ModuleDict needs string keys)
            key = lemma.encode("utf-8").hex()
            self.sense_heads[key] = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size // 2, n_senses),
            )

        # Shared register head
        self.register_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size, len(register_labels)),
        )

        # Shared period head
        self.period_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size, len(period_labels)),
        )

        self.lemma_key_map = {lemma: lemma.encode("utf-8").hex()
                              for lemma in sense_maps}

    def forward(self, input_ids, attention_mask, target_pos, lemma=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, seq, hidden)

        # Extract target token embedding
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        target_pos = target_pos.clamp(0, hidden.size(1) - 1)
        target_emb = hidden[batch_idx, target_pos]  # (batch, hidden)

        # Register and period predictions (always)
        register_logits = self.register_head(target_emb)
        period_logits = self.period_head(target_emb)

        return {
            "target_embedding": target_emb,
            "register_logits": register_logits,
            "period_logits": period_logits,
        }

    def predict_sense(self, target_emb, lemma):
        key = self.lemma_key_map.get(lemma)
        if key and key in self.sense_heads:
            return self.sense_heads[key](target_emb)
        return None


def train(epochs: int = 15, batch_size: int = 16, lr: float = 2e-5,
          freeze_layers: int = 8):
    """Train the sense classifier."""
    print(f"Device: {DEVICE}")

    # Load data
    with open(DATA_DIR / "train.json") as f:
        train_data = json.load(f)
    with open(DATA_DIR / "val.json") as f:
        val_data = json.load(f)
    with open(DATA_DIR / "sense_labels.json") as f:
        sense_maps = json.load(f)

    # Build label maps
    all_registers = sorted(set(ex["register"] for ex in train_data + val_data))
    all_periods = sorted(set(ex["period"] for ex in train_data + val_data))
    register_map = {r: i for i, r in enumerate(all_registers)}
    period_map = {p: i for i, p in enumerate(all_periods)}

    # Add numeric labels
    for ex in train_data + val_data:
        ex["register_id"] = register_map.get(ex["register"], 0)
        ex["period_id"] = period_map.get(ex["period"], 0)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Senses: {sum(len(v) for v in sense_maps.values())} across {len(sense_maps)} lemmata")
    print(f"Registers: {len(all_registers)}: {all_registers}")
    print(f"Periods: {len(all_periods)}: {all_periods}")

    # Load model
    print("\nLoading aristoBERTo...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("Jacobo/aristoBERTo")
    encoder = AutoModel.from_pretrained("Jacobo/aristoBERTo")

    model = SenseClassifier(
        encoder=encoder,
        hidden_size=encoder.config.hidden_size,
        sense_maps=sense_maps,
        register_labels=all_registers,
        period_labels=all_periods,
        freeze_layers=freeze_layers,
    )
    model = model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total "
          f"({trainable*100//total}%)")

    # Datasets
    train_ds = SenseDataset(train_data, tokenizer)
    val_ds = SenseDataset(val_data, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    sense_criterion = nn.CrossEntropyLoss()
    register_criterion = nn.CrossEntropyLoss()
    period_criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            target_pos = batch["target_pos"].to(DEVICE)

            outputs = model(input_ids, attention_mask, target_pos)

            loss = torch.tensor(0.0, device=DEVICE)

            # Sense loss — per lemma
            for i in range(len(batch["lemma"])):
                lemma = batch["lemma"][i]
                sense_logits = model.predict_sense(
                    outputs["target_embedding"][i:i+1], lemma
                )
                if sense_logits is not None:
                    target = torch.tensor([batch["sense_label"][i]],
                                         device=DEVICE)
                    loss = loss + sense_criterion(sense_logits, target)
                    pred = sense_logits.argmax(dim=-1)
                    correct += (pred == target).sum().item()
                    total += 1

            # Register loss (weight: 0.3)
            reg_targets = torch.tensor(
                [register_map.get(r, 0) for r in batch["register"]],
                device=DEVICE,
            )
            loss = loss + 0.3 * register_criterion(
                outputs["register_logits"], reg_targets
            )

            # Period loss (weight: 0.3)
            per_targets = torch.tensor(
                [period_map.get(p, 0) for p in batch["period"]],
                device=DEVICE,
            )
            loss = loss + 0.3 * period_criterion(
                outputs["period_logits"], per_targets
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        train_acc = correct / total if total > 0 else 0

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                target_pos = batch["target_pos"].to(DEVICE)

                outputs = model(input_ids, attention_mask, target_pos)

                for i in range(len(batch["lemma"])):
                    lemma = batch["lemma"][i]
                    sense_logits = model.predict_sense(
                        outputs["target_embedding"][i:i+1], lemma
                    )
                    if sense_logits is not None:
                        target = batch["sense_label"][i]
                        pred = sense_logits.argmax(dim=-1).item()
                        val_correct += (pred == target)
                        val_total += 1

        val_acc = val_correct / val_total if val_total > 0 else 0

        print(f"Epoch {epoch+1:>2d}/{epochs}: "
              f"loss={total_loss/len(train_loader):.4f} "
              f"train_acc={train_acc:.3f} "
              f"val_acc={val_acc:.3f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "sense_maps": sense_maps,
                "register_map": register_map,
                "period_map": period_map,
                "epoch": epoch,
                "val_acc": val_acc,
            }, MODEL_DIR / "best_model.pt")
            print(f"  → Saved best model (val_acc={val_acc:.3f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    return best_val_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--freeze-layers", type=int, default=8)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, freeze_layers=args.freeze_layers)

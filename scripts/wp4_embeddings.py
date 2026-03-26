#!/usr/bin/env python3
"""
WP4 Task 4.5: Embedding-space probing and interpretability.

Extracts target-token embeddings from the fine-tuned model for all
labelled occurrences, then analyses whether the embedding space
encodes interpretable structure:

1. Sense clustering — do same-sense occurrences cluster together?
2. Period separation — do different periods occupy different regions?
3. Register separation — does register map onto embedding geometry?
4. Diachronic trajectories — does sense prevalence shift visibly?
5. Cross-lemma structure — do semantically related senses across
   different lemmata cluster (e.g., kosmos 'world' near physis 'nature')?

Following Grindrod & Grindrod's approach: examine the geometry of the
learned embedding space for human-interpretable semantic structure.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

DATA_DIR = Path(__file__).parent.parent / "models" / "data"
MODEL_DIR = Path(__file__).parent.parent / "models" / "checkpoints"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "embeddings"

TRANSLIT = {
    "κόσμος": "kosmos", "λόγος": "logos", "ψυχή": "psyche",
    "ἀρετή": "arete", "δίκη": "dike", "τέχνη": "techne",
    "νόμος": "nomos", "φύσις": "physis", "δαίμων": "daimon",
    "σῶμα": "soma", "θεός": "theos", "χάρις": "charis",
}


def extract_embeddings():
    """Extract target-token embeddings from fine-tuned model."""
    from wp4_train import SenseClassifier

    # Load checkpoint
    checkpoint = torch.load(MODEL_DIR / "best_model.pt", map_location="cpu",
                            weights_only=False)
    sense_maps = checkpoint["sense_maps"]
    register_map = checkpoint["register_map"]
    period_map = checkpoint["period_map"]

    # Load encoder
    tokenizer = AutoTokenizer.from_pretrained("Jacobo/aristoBERTo")
    encoder = AutoModel.from_pretrained("Jacobo/aristoBERTo")

    model = SenseClassifier(
        encoder=encoder,
        hidden_size=encoder.config.hidden_size,
        sense_maps=sense_maps,
        register_labels=list(register_map.keys()),
        period_labels=list(period_map.keys()),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Load all labelled data
    all_data = []
    for split in ["train", "val", "test"]:
        with open(DATA_DIR / f"{split}.json") as f:
            all_data.extend(json.load(f))
    print(f"Extracting embeddings for {len(all_data)} examples...")

    embeddings = []
    metadata = []

    with torch.no_grad():
        for i, ex in enumerate(all_data):
            encoding = tokenizer(
                ex["greek_context"],
                max_length=128, truncation=True, padding="max_length",
                return_tensors="pt",
            )

            # Find target position
            tokens = tokenizer.tokenize(ex["greek_context"])
            surface_tokens = tokenizer.tokenize(ex["surface_form"])
            target_pos = 1
            if surface_tokens:
                for j, t in enumerate(tokens):
                    if t == surface_tokens[0]:
                        target_pos = j + 1
                        break
            target_pos = min(target_pos, 127)

            outputs = model.encoder(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            )
            emb = outputs.last_hidden_state[0, target_pos].numpy()
            embeddings.append(emb)

            metadata.append({
                "lemma": ex["lemma"],
                "sense": ex["sense_label"],
                "period": ex["period"],
                "author": ex["author"],
                "register": ex["register"],
                "surface_form": ex["surface_form"],
            })

            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(all_data)}...", flush=True)

    embeddings = np.array(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, ensure_ascii=False)

    return embeddings, metadata


def analyse_embeddings(embeddings, metadata):
    """Analyse embedding space structure."""
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    n = len(metadata)
    lemmata = [m["lemma"] for m in metadata]
    senses = [m["sense"] for m in metadata]
    periods = [m["period"] for m in metadata]
    registers = [m["register"] for m in metadata]

    print(f"\n{'='*60}")
    print("EMBEDDING SPACE ANALYSIS")
    print(f"{'='*60}")

    # 1. Sense clustering quality
    print("\n── 1. Sense Clustering ──")
    for lemma in sorted(set(lemmata)):
        idx = [i for i, l in enumerate(lemmata) if l == lemma]
        if len(idx) < 10:
            continue
        lemma_emb = embeddings[idx]
        lemma_senses = [senses[i] for i in idx]
        unique_senses = sorted(set(lemma_senses))
        if len(unique_senses) < 2:
            continue
        sense_ids = [unique_senses.index(s) for s in lemma_senses]
        try:
            score = silhouette_score(lemma_emb, sense_ids)
            print(f"  {TRANSLIT.get(lemma, lemma):>10s}: silhouette={score:.3f} "
                  f"({len(unique_senses)} senses, {len(idx)} examples)")
        except Exception:
            pass

    # 2. Period separation
    print("\n── 2. Period Separation ──")
    period_labels = sorted(set(periods))
    if len(period_labels) >= 2:
        period_ids = [period_labels.index(p) for p in periods]
        score = silhouette_score(embeddings, period_ids)
        print(f"  Overall silhouette by period: {score:.3f}")

        # Per-lemma period separation
        for lemma in sorted(set(lemmata)):
            idx = [i for i, l in enumerate(lemmata) if l == lemma]
            if len(idx) < 20:
                continue
            lemma_emb = embeddings[idx]
            lemma_periods = [periods[i] for i in idx]
            unique_periods = sorted(set(lemma_periods))
            if len(unique_periods) < 2:
                continue
            per_ids = [unique_periods.index(p) for p in lemma_periods]
            try:
                score = silhouette_score(lemma_emb, per_ids)
                print(f"  {TRANSLIT.get(lemma, lemma):>10s}: period_silhouette={score:.3f}")
            except Exception:
                pass

    # 3. Register separation
    print("\n── 3. Register Separation ──")
    reg_labels = sorted(set(registers))
    if len(reg_labels) >= 2:
        reg_ids = [reg_labels.index(r) for r in registers]
        score = silhouette_score(embeddings, reg_ids)
        print(f"  Overall silhouette by register: {score:.3f}")

    # 4. Nearest neighbours — Grindrod-style
    print("\n── 4. Nearest Neighbour Analysis ──")
    # For each lemma, find examples of different senses and check if
    # nearest neighbours share the same sense
    nn_accuracy = {}
    for lemma in sorted(set(lemmata)):
        idx = [i for i, l in enumerate(lemmata) if l == lemma]
        if len(idx) < 20:
            continue
        lemma_emb = embeddings[idx]
        lemma_senses = [senses[i] for i in idx]

        # For each point, find k nearest neighbours
        k = 5
        correct = 0
        total = 0
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(lemma_emb)
        for i in range(len(idx)):
            # Get k nearest (excluding self)
            sim_scores = sims[i].copy()
            sim_scores[i] = -1  # exclude self
            nn_idx = np.argsort(sim_scores)[-k:]
            nn_senses = [lemma_senses[j] for j in nn_idx]
            correct += sum(1 for s in nn_senses if s == lemma_senses[i])
            total += k

        nn_acc = correct / total if total > 0 else 0
        nn_accuracy[lemma] = nn_acc
        print(f"  {TRANSLIT.get(lemma, lemma):>10s}: kNN sense accuracy={nn_acc:.3f} (k={k})")

    # 5. Dimensionality reduction for visualisation
    print("\n── 5. Generating Visualisations ──")

    # PCA
    pca = PCA(n_components=50)
    emb_pca50 = pca.fit_transform(embeddings)
    print(f"  PCA: {pca.explained_variance_ratio_[:5].sum():.1%} variance in first 5 components")

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(emb_pca50)

    # Save for plotting
    np.save(OUTPUT_DIR / "tsne_2d.npy", emb_2d)

    # Generate summary report
    report = {
        "n_examples": n,
        "n_lemmata": len(set(lemmata)),
        "n_senses": len(set(senses)),
        "nn_accuracy": {TRANSLIT.get(k, k): v for k, v in nn_accuracy.items()},
        "pca_variance_5d": float(pca.explained_variance_ratio_[:5].sum()),
    }
    with open(OUTPUT_DIR / "analysis_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 6. Diachronic trajectory analysis
    print("\n── 6. Diachronic Sense Trajectories ──")
    period_order = ["homeric", "archaic", "classical", "hellenistic", "imperial", "koine"]

    for lemma in sorted(set(lemmata)):
        idx = [i for i, l in enumerate(lemmata) if l == lemma]
        if len(idx) < 50:
            continue

        lemma_meta = [metadata[i] for i in idx]
        lemma_emb = embeddings[idx]

        # Compute centroid per sense per period
        sense_period = defaultdict(lambda: defaultdict(list))
        for j, m in enumerate(lemma_meta):
            sense_period[m["sense"]][m["period"]].append(lemma_emb[j])

        print(f"\n  {TRANSLIT.get(lemma, lemma)}:")
        for sense in sorted(sense_period.keys()):
            periods_present = [p for p in period_order if p in sense_period[sense]]
            if len(periods_present) >= 2:
                counts = [len(sense_period[sense].get(p, [])) for p in period_order]
                timeline = " → ".join(
                    f"{p[:4]}:{len(sense_period[sense].get(p, []))}"
                    for p in period_order
                    if p in sense_period[sense]
                )
                print(f"    {sense[:40]:40s} {timeline}")

    print(f"\nAnalysis saved to {OUTPUT_DIR}")


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    # Check if embeddings already extracted
    if (OUTPUT_DIR / "embeddings.npy").exists():
        print("Loading cached embeddings...")
        embeddings = np.load(OUTPUT_DIR / "embeddings.npy")
        with open(OUTPUT_DIR / "metadata.json") as f:
            metadata = json.load(f)
    else:
        embeddings, metadata = extract_embeddings()

    # Install sklearn if needed
    try:
        import sklearn
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn"],
                      capture_output=True)

    analyse_embeddings(embeddings, metadata)


if __name__ == "__main__":
    main()

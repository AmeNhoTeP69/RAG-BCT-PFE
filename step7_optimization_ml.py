"""
step7_optimization_ml.py
────────────────────────────────────────────────────────────────────────────────
STEP 7 — ML Optimization: Regulation Domain Classifier
         with SMOTE Oversampling + AdamW Optimizer

Architecture:
    Document Chunks (TF-IDF features)
           │
           ├──► SMOTE Oversampling  (handle class imbalance)
           │          │
           │          ▼
           └──► PyTorch MLP Classifier
                      │  Trained with AdamW optimizer (weight decay = L2 reg)
                      ▼
               Document Category Labels
               (e.g., "Changes", "LCB-FT", "Reporting", ...)
                      │
                      ▼
               Saved to classifier_meta.json + regulation_classifier.pt
               These labels enrich chunk metadata for Topic-Boosted RAG.

Run after step6_topic_modeling.py:
    python step7_optimization_ml.py

Key Concepts Demonstrated:
  ■ Feature Engineering  — TF-IDF sparse features from regulatory text
  ■ Data Augmentation    — SMOTE oversampling for imbalanced classes
  ■ Modern Optimizer     — AdamW (decoupled weight decay from gradient updates)
  ■ Supervised Learning  — Multi-class classification with CrossEntropyLoss
  ■ Model Persistence    — PyTorch state_dict saved for inference in RAG engine
"""

import json
import logging
import warnings
import numpy as np
from pathlib import Path

import config

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Keyword-based labeling rules ──────────────────────────────────────────────
# Since BCT documents are unlabeled, we use keyword heuristics to generate
# training labels. This is standard practice for weakly-supervised classification.
CATEGORY_KEYWORDS = {
    "Réglementation des Changes": [
        "devises", "change", "transfert", "étranger", "convertibilité",
        "résidents", "non-résidents", "opérations de change", "rapatriement",
    ],
    "Blanchiment d'Argent / LCB-FT": [
        "blanchiment", "financement du terrorisme", "lcb-ft", "gafi",
        "suspicious", "vigilance", "déclaration de soupçon", "kyc",
        "connaissance du client",
    ],
    "Reporting et Supervision Prudentielle": [
        "reporting", "déclaration", "statistiques", "états financiers",
        "ratio", "capitalisation", "fonds propres", "solvabilité",
        "bilan", "surveillance prudentielle",
    ],
    "Opérations Bancaires et Crédit": [
        "crédit", "prêt", "emprunt", "intérêt", "taux", "financement",
        "ouverture de compte", "dépôt", "retrait", "virement",
    ],
    "Gestion des Risques": [
        "risque", "exposition", "contrepartie", "liquidité", "stress test",
        "concentration", "marché", "opérationnel", "provision",
    ],
    "Marchés Financiers et Valeurs": [
        "valeurs mobilières", "bourse", "titre", "action", "obligation",
        "fonds commun", "sicav", "marché financier", "placement",
    ],
    "Microfinance et Inclusion Financière": [
        "microfinance", "microcrédit", "imf", "inclusion financière",
        "population", "associations", "tpe", "petite entreprise",
    ],
    "Gouvernance et Conformité": [
        "gouvernance", "conformité", "conseil d'administration", "comité",
        "auditeur", "commissaire aux comptes", "éthique", "déontologie",
        "politique", "procédures",
    ],
}

CATEGORIES = list(CATEGORY_KEYWORDS.keys())
N_CLASSES = len(CATEGORIES)


def label_text(text: str) -> int:
    """
    Assign a regulatory category to a text using keyword matching.
    Returns the index of the best-matching category.
    Falls back to 'Gouvernance et Conformité' (last class) if no match.
    """
    text_lower = text.lower()
    scores = np.zeros(N_CLASSES)
    for i, (category, keywords) in enumerate(CATEGORY_KEYWORDS.items()):
        for kw in keywords:
            scores[i] += text_lower.count(kw)
    if scores.max() == 0:
        return N_CLASSES - 1  # Default to last class
    return int(np.argmax(scores))


def load_chunks() -> list:
    log.info(f"Loading chunks from {config.CHUNKS_PATH}...")
    with open(config.CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    log.info(f"Loaded {len(chunks)} chunks.")
    return chunks


def build_features_and_labels(chunks: list):
    """
    Build TF-IDF feature matrix and label vector from document chunks.
    Returns (X, y, vectorizer).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [c.get("text", "") for c in chunks]
    labels = [label_text(t) for t in texts]

    log.info("Building TF-IDF feature matrix (max 2000 features)...")
    vectorizer = TfidfVectorizer(
        max_features=2000,
        min_df=2,
        sublinear_tf=True,
        analyzer="word",
        ngram_range=(1, 2),  # Unigrams + bigrams for better regulatory term capture
    )
    X = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    y = np.array(labels, dtype=np.int64)

    # Log class distribution
    unique, counts = np.unique(y, return_counts=True)
    log.info("Class distribution BEFORE SMOTE:")
    for cls_idx, cnt in zip(unique, counts):
        log.info(f"  [{cls_idx:02d}] {CATEGORIES[cls_idx][:35]:<35} : {cnt} samples")

    return X, y, vectorizer


def apply_smote(X: np.ndarray, y: np.ndarray):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.
    SMOTE creates synthetic samples for minority classes by interpolating
    between existing samples in feature space.
    """
    from imblearn.over_sampling import SMOTE

    log.info("Applying SMOTE oversampling to balance class distribution...")
    # k_neighbors must be <= minority class size - 1
    min_class_count = np.bincount(y).min()
    k_neighbors = min(5, max(1, min_class_count - 1))
    log.info(f"  Using k_neighbors={k_neighbors} (auto-adjusted for small classes)")

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X, y)

    unique, counts = np.unique(y_res, return_counts=True)
    log.info(f"Class distribution AFTER SMOTE: {dict(zip(unique.tolist(), counts.tolist()))}")
    log.info(f"Dataset grew: {len(y)} → {len(y_res)} samples")
    return X_res, y_res


# ── PyTorch MLP Classifier ────────────────────────────────────────────────────
def build_pytorch_model(input_dim: int, hidden_dim: int, n_classes: int):
    """Build a simple 2-layer MLP for regulation classification."""
    import torch
    import torch.nn as nn

    class RegulationMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, n_classes),
            )

        def forward(self, x):
            return self.network(x)

    return RegulationMLP()


def train_classifier(X: np.ndarray, y: np.ndarray):
    """
    Train the MLP with AdamW optimizer and CrossEntropyLoss.
    AdamW = Adam with decoupled weight decay (better generalization than L2 in Adam).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on device: {device}")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    # Model
    input_dim = X.shape[1]
    model = build_pytorch_model(
        input_dim=input_dim,
        hidden_dim=config.CLASSIFIER_HIDDEN_DIM,
        n_classes=N_CLASSES,
    ).to(device)

    # ── AdamW Optimizer (key contribution) ───────────────────────────────────
    # AdamW decouples weight decay from the gradient update,
    # unlike standard Adam which conflates L2 regularization with weight decay.
    # This leads to better generalization on NLP tasks.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.CLASSIFIER_LR,
        weight_decay=config.CLASSIFIER_WEIGHT_DECAY,  # L2 regularization strength
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Cosine Annealing LR scheduler (modern complement to AdamW)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.CLASSIFIER_EPOCHS, eta_min=1e-6
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Training Loop ─────────────────────────────────────────────────────────
    log.info(f"Training MLP for {config.CLASSIFIER_EPOCHS} epochs (AdamW, lr={config.CLASSIFIER_LR})...")
    best_val_acc = 0.0
    best_state = None
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, config.CLASSIFIER_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            preds = val_logits.argmax(dim=1)
            val_acc = (preds == y_val_t).float().mean().item()

        avg_loss = epoch_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            log.info(
                f"  Epoch {epoch:03d}/{config.CLASSIFIER_EPOCHS} "
                f"| Loss: {avg_loss:.4f} "
                f"| Val Acc: {val_acc:.4f} "
                f"| LR: {lr_now:.2e}"
            )

    log.info(f"Training complete. Best Validation Accuracy: {best_val_acc:.4f}")

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    return model, input_dim, history, best_val_acc


def save_model(model, input_dim: int, best_val_acc: float):
    import torch

    log.info(f"Saving classifier model to {config.CLASSIFIER_MODEL_PATH}...")
    torch.save(model.state_dict(), str(config.CLASSIFIER_MODEL_PATH))

    meta = {
        "input_dim": input_dim,
        "hidden_dim": config.CLASSIFIER_HIDDEN_DIM,
        "n_classes": N_CLASSES,
        "categories": CATEGORIES,
        "best_val_accuracy": round(best_val_acc, 4),
        "optimizer": "AdamW",
        "lr": config.CLASSIFIER_LR,
        "weight_decay": config.CLASSIFIER_WEIGHT_DECAY,
        "smote_applied": True,
    }
    with open(config.CLASSIFIER_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log.info(f"Classifier meta saved to {config.CLASSIFIER_META_PATH}")


def annotate_chunks_with_categories(chunks: list, model, vectorizer, input_dim: int):
    """
    Run inference: add predicted 'category' to each chunk's metadata.
    Saves an updated chunks file that the RAG engine can use.
    """
    import torch

    log.info("Annotating all chunks with predicted regulation categories...")
    texts = [c.get("text", "") for c in chunks]
    X = vectorizer.transform(texts).toarray().astype(np.float32)
    X_t = torch.tensor(X, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        preds = logits.argmax(dim=1).numpy()

    for i, chunk in enumerate(chunks):
        chunk.setdefault("metadata", {})
        chunk["metadata"]["regulation_category"] = CATEGORIES[preds[i]]
        chunk["metadata"]["category_id"] = int(preds[i])

    # Save annotated chunks
    annotated_path = config.CHUNKS_PATH.parent / "chunks_annotated.json"
    with open(annotated_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    log.info(f"Annotated chunks saved to {annotated_path}")
    return chunks


def print_summary(history: dict, best_val_acc: float):
    print("\n" + "=" * 70)
    print("  REGULATION CLASSIFIER — TRAINING SUMMARY")
    print("=" * 70)
    print(f"  Optimizer  : AdamW (lr={config.CLASSIFIER_LR}, wd={config.CLASSIFIER_WEIGHT_DECAY})")
    print(f"  Scheduler  : CosineAnnealingLR")
    print(f"  Epochs     : {config.CLASSIFIER_EPOCHS}")
    print(f"  SMOTE      : Applied (balanced minority classes)")
    print(f"  Classes    : {N_CLASSES}")
    print(f"  Best Val Acc: {best_val_acc:.2%}")
    print()
    print("  CATEGORY LABELS:")
    for i, cat in enumerate(CATEGORIES):
        print(f"    [{i}] {cat}")
    print()
    print(f"  Model saved:    {config.CLASSIFIER_MODEL_PATH}")
    print(f"  Metadata saved: {config.CLASSIFIER_META_PATH}")
    print("\n✓ Step 7 complete! SMOTE + AdamW classifier trained and saved.")


def run():
    config.ensure_dirs()

    # Check required inputs
    if not config.CHUNKS_PATH.exists():
        log.error(f"Chunks file not found at {config.CHUNKS_PATH}. Run step2_chunk.py first.")
        return

    # 1. Load data
    chunks = load_chunks()

    # 2. Feature Engineering (TF-IDF) + Weak Labeling
    X, y, vectorizer = build_features_and_labels(chunks)

    # 3. SMOTE Oversampling
    X_resampled, y_resampled = apply_smote(X, y)

    # 4. Train classifier with AdamW
    model, input_dim, history, best_val_acc = train_classifier(X_resampled, y_resampled)

    # 5. Save model and metadata
    save_model(model, input_dim, best_val_acc)

    # 6. Annotate chunks with predicted categories
    annotate_chunks_with_categories(chunks, model, vectorizer, input_dim)

    # 7. Print summary
    print_summary(history, best_val_acc)


if __name__ == "__main__":
    run()

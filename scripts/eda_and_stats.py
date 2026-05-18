"""
eda_and_stats.py
────────────────
Exploratory Data Analysis for BCT Documents.
Generates charts and metrics for the PFE report.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
INPUT_FILE = Path("data/extracted/documents.json")
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

def load_data():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Run step1_extract.py first.")
        return None
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def generate_eda():
    df = load_data()
    if df is None:
        return

    print(f"Analyzing {len(df)} documents...")

    # 1. Language Distribution (Academic Requirement: Bilingual Ar/Fr)
    if 'language' in df.columns:
        plt.figure()
        lang_counts = df['language'].value_counts()
        plt.pie(lang_counts, labels=lang_counts.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
        plt.title("Répartition des langues (Arabe vs Français)")
        plt.savefig(OUTPUT_DIR / "language_distribution.png")
        print("Saved language_distribution.png")

    # 2. Document Types (Circulaire vs Note)
    plt.figure()
    sns.countplot(data=df, x='type', palette="viridis")
    plt.title("Répartition par Type de Document")
    plt.ylabel("Nombre de documents")
    plt.savefig(OUTPUT_DIR / "document_types.png")
    print("Saved document_types.png")

    # 3. Evolution over time (Bar chart by Year)
    if 'year' in df.columns:
        plt.figure()
        # Clean years (remove None)
        years = df['year'].dropna().astype(int).sort_values()
        sns.countplot(x=years, palette="magma")
        plt.title("Évolution du nombre de publications par année")
        plt.xticks(rotation=45)
        plt.savefig(OUTPUT_DIR / "yearly_evolution.png")
        print("Saved yearly_evolution.png")

    # 4. Text Length Analysis (Feature Analysis)
    plt.figure()
    sns.boxplot(data=df, x='type', y='char_count', palette="Set2")
    plt.title("Nombre de caractères par type de document")
    plt.yscale("log")
    plt.savefig(OUTPUT_DIR / "content_length_analysis.png")
    print("Saved content_length_analysis.png")

    # 5. Extraction Method (Academic Requirement: OCR)
    if 'extraction_method' in df.columns:
        plt.figure()
        sns.countplot(data=df, x='extraction_method', palette="coolwarm")
        plt.title("Méthodes d'extraction utilisées (Fitz vs OCR)")
        plt.savefig(OUTPUT_DIR / "extraction_method.png")
        print("Saved extraction_method.png")

    # Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Total Documents : {len(df)}")
    if 'year' in df.columns:
        valid_years = pd.to_numeric(df['year'], errors='coerce').dropna()
        if not valid_years.empty:
            print(f"Year Range      : {int(valid_years.min())} - {int(valid_years.max())}")
    if 'language' in df.columns:
        print("\nLanguage Counts:")
        print(df['language'].value_counts())
    
    print("\nDone! All plots saved in the 'plots/' directory.")

if __name__ == "__main__":
    generate_eda()

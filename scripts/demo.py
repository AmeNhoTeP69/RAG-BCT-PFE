"""
demo.py
───────
BCT RAG & Graph RAG Interactive Demo

Allows users to switch between standard RAG and Graph RAG modes and ask questions 
about the BCT circulars and notes.
"""

import sys
import logging
import argparse
from termcolor import colored

import config
from rag_engine import RAGEngine
from graph_rag_engine import GraphRAGEngine

# Define styles
def print_header(text):
    print(colored(f"\n{'='*60}\n{text.center(60)}\n{'='*60}", "cyan", attrs=["bold"]))

def print_result(answer, sources, related=None):
    print(colored("\n[ RÉPONSE ]", "green", attrs=["bold"]))
    print(answer)
    print(colored("\n[ SOURCES ]", "yellow", attrs=["bold"]))
    for source in sources:
        print(f" - {source}")
    if related:
        print(colored("\n[ RELATIONS GRAPHE ]", "magenta", attrs=["bold"]))
        print(", ".join(related[:10]))

def main():
    parser = argparse.ArgumentParser(description="BCT RAG Demo")
    parser.add_argument("--mode", choices=["rag", "graph"], default="rag", help="Mode: rag or graph")
    args = parser.parse_args()

    print_header("BCT RAG & GRAPH RAG DEMO")
    print(f"Mode actuel: {args.mode.upper()}")
    print(f"Provider LLM: {config.LLM_PROVIDER.upper()}")
    
    # Check if necessary data exists
    if not config.FAISS_INDEX_PATH.exists():
        print(colored(f"\nERREUR: Index FAISS non trouvé à {config.FAISS_INDEX_PATH}", "red"))
        print("Veuillez d'abord exécuter les étapes 3 et 4 :")
        print("  python step3_embed.py")
        print("  python step4_index.py")
        sys.exit(1)

    if args.mode == "graph" and not config.GRAPH_PATH.exists():
        print(colored(f"\nAVERTISSEMENT: Graphe non trouvé à {config.GRAPH_PATH}", "yellow"))
        print("Certaines fonctionnalités du Graph RAG seront limitées.")
        print("Exécutez : python step5_build_graph.py")

    # Load Engine
    print("Chargement des ressources... (cela peut prendre quelques secondes)")
    if args.mode == "rag":
        engine = RAGEngine()
    else:
        engine = GraphRAGEngine()

    print(colored("\nPrêt pour vos questions ! (tapez 'quit' ou 'exit' pour arrêter)", "green"))
    
    while True:
        try:
            query = input(colored("\nVotre question > ", "blue", attrs=["bold"]))
            if query.lower() in ["quit", "exit", "q"]:
                break
            if not query.strip():
                continue

            print(colored("Réflexion en cours...", "grey"))
            result = engine.query(query)
            
            print_result(
                result.get("answer", "Erreur: Pas de réponse."),
                result.get("sources", []),
                result.get("related_nodes") if args.mode == "graph" else None
            )

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(colored(f"\nErreur: {str(e)}", "red"))

    print_header("MERCI D'AVOIR UTILISÉ BCT RAG")

if __name__ == "__main__":
    main()

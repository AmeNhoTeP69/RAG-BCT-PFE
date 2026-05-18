"""
run_project.py
──────────────
The Master Entry Point for the BCT RAG & Graph RAG Project.
Use this script to run the pipeline, generate EDA, or start the web UI.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner(text):
    print("\n" + "="*60)
    print(text.center(60))
    print("="*60 + "\n")

def run_script(script_name, args=None):
    """Run a python script and wait for it to finish."""
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    
    print(f"--- Executing: {script_name} ---")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"\n[ERROR] Failed to run {script_name}")
        return False

def full_pipeline():
    print_banner("RUNNING FULL DATA PIPELINE")
    steps = [
        "step1_extract.py",
        "step2_chunk.py",
        "step3_embed.py",
        "step4_index.py",
        "step5_build_graph.py"
    ]
    for step in steps:
        if not run_script(step):
            print("\nPipeline interrupted due to error.")
            return
    print("\n✓ Full Pipeline Completed Successfully!")

def analysis_lab():
    print_banner("GENERATING ANALYSIS & EDA FOR SUPERVISOR")
    run_script("eda_and_stats.py")
    run_script("nlp_research_lab.py")
    print("\n✓ Analysis Lab Complete. Check the 'plots/' folder and 'academic_summary.md'.")

def start_web_ui():
    print_banner("STARTING WEB INTERFACE (FASTAPI)")
    print("Server starting at: http://localhost:8000")
    print("Keep this terminal open to keep the server running.")
    print("Press Ctrl+C to stop.\n")
    try:
        run_script("api.py")
    except KeyboardInterrupt:
        print("\nServer stopped.")

def start_cli_demo():
    print_banner("STARTING INTERACTIVE CLI DEMO")
    mode = input("Choose mode (1: Standard RAG, 2: Graph RAG) [1]: ")
    flag = "--mode graph" if mode == "2" else "--mode rag"
    run_script("demo.py", args=flag.split())

def main_menu():
    while True:
        print_banner("BCT PROJECT MASTER CONTROL")
        print("1. [Pipeline]   Run Full Data Processing (Extract -> Graph)")
        print("2. [Analysis]   Generate EDA & NLP Research (For Supervisor)")
        print("3. [Interface]  Launch Web UI (Browser)")
        print("4. [Chat]       Interactive CLI Demo (Terminal)")
        print("Q. Quit")
        
        choice = input("\nSelect an option: ").strip().lower()
        
        if choice == '1':
            full_pipeline()
        elif choice == '2':
            analysis_lab()
        elif choice == '3':
            start_web_ui()
        elif choice == '4':
            start_cli_demo()
        elif choice in ['q', 'quit', 'exit']:
            print("\nGoodbye!")
            break
        else:
            print("\n[!] Invalid selection. Please try again.")
        
        input("\nPress Enter to return to menu...")

if __name__ == "__main__":
    # Check if we are in a venv
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("\n[WARNING] You are not running in a Virtual Environment (venv).")
        print("It is highly recommended to activate your venv before running the project.\n")
    
    main_menu()

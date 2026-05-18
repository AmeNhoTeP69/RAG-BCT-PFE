import requests
import json
import time
import sys

# Force UTF-8 for printing if possible
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_URL = "http://localhost:8000/api/query"

TEST_CASES = [
    {
        "name": "Fact Retrieval (Simple)",
        "query": "Qu'est-ce qu'une circulaire de la BCT ?",
    },
    {
        "name": "Structural/Relationship",
        "query": "Quels sont les textes liés à la circulaire 2017-08 ?",
    },
    {
        "name": "Topic/Categorization",
        "query": "Quelles sont les obligations en matière de lutte contre le blanchiment ?",
    },
    {
        "name": "Arabic Support",
        "query": "ماهي شروط فتح مكتب صرف في تونس؟",
    },
    {
        "name": "Out of Scope (Safety)",
        "query": "Comment préparer un tajine tunisien ?",
    }
]

def run_tests():
    results = []
    print("="*50)
    print("BCT RAG SYSTEM TEST SUITE")
    print("="*50)

    for test in TEST_CASES:
        print(f"\n[TEST] {test['name']}")
        
        test_results = {"test": test['name'], "query": test['query']}
        
        for mode in ["rag", "graph"]:
            print(f"  Mode: {mode.upper()}...", end="", flush=True)
            try:
                resp = requests.post(BASE_URL, json={"query": test['query'], "mode": mode}, timeout=90)
                if resp.status_code == 200:
                    data = resp.json()
                    test_results[f"{mode}_answer"] = data.get("answer", "")
                    test_results[f"{mode}_sources"] = data.get("sources", [])
                    print(" OK")
                else:
                    print(f" ERROR ({resp.status_code})")
                    test_results[f"{mode}_error"] = resp.text
            except Exception as e:
                print(f" FAILED")
                test_results[f"{mode}_error"] = str(e)
        
        results.append(test_results)

    with open("c:/Users/sakhi/Desktop/projects/PFE/Scraping BCT circumaires+notes/report_notes/test_results_raw.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n[SUCCESS] Results saved to report_notes/test_results_raw.json")

if __name__ == "__main__":
    run_tests()

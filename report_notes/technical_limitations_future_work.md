# Technical Limitations & Future Improvements
**Project:** Hybrid Graph RAG for BCT Regulatory Analysis
**Date:** May 2026

## 1. Engineering Limitations (The "Honest" View)

### A. Graph Extraction Sensitivity
The current system utilizes a rule-based engine (Regex + SpaCy) to identify relationships between documents. While highly effective for standard citations (e.g., "Circulaire n°2017-08"), it may miss "implied" or "narrative" relationships that do not follow a fixed pattern.
*   **Impact:** Some edges in the Knowledge Graph might be missing if the source text uses non-standard phrasing.

### B. Inference Latency vs. Model Size
To achieve high-quality reasoning, we use the **Llama 3.3 70B** model. While significantly smarter than smaller models, it introduces a dependency on external APIs (Groq) and can hit rate limits if queried too rapidly.
*   **Trade-off:** We prioritize **Accuracy** (crucial for legal/banking) over **Latency** (speed).

### C. Multilingual Semantic Nuance
The system uses a multilingual transformer model. While it handles Arabic and French excellently, it may occasionally struggle with highly specialized "Tunisian Legal Dialect" terms that are not common in general Arabic datasets.

---

## 2. Hardening & Mitigation Strategies

| Limitation | Mitigation Implemented |
| :--- | :--- |
| **Hallucination** | Implemented a "Strict Grounding" system prompt + metadata verification. |
| **Rate Limiting** | Added an exponential backoff retry mechanism (Error 429 handling). |
| **Naming Mismatch** | Developed a Fuzzy Regex matching logic for Graph Node retrieval. |

---

## 3. Future Roadmap (Post-PFE Improvements)

### A. Domain-Specific Fine-Tuning (LoRA)
Fine-tune a smaller model (like Llama 3 8B) specifically on the BCT's historical archives to improve the understanding of specialized banking terminology without needing 70B parameters.

### B. Interactive Graph Visualization
Integrate a React-based interactive graph (using D3.js or React-Force-Graph) to allow the user to manually explore the connections between circulars visually.

### C. Automated OCR Pipeline 2.0
Implement a layout-aware OCR (like LayoutLM) to better handle complex tables and signatures in scanned BCT PDFs, which are often the most difficult data to parse.

---
> [!TIP]
> **Defense Strategy:** Use these points to answer questions about "Why did you use this model?" or "How would you make this commercial?".

# Réponses aux Notes de l'Encadrante (Mise à Jour Finale)

Ce document récapitule comment chaque point technique et remarque spécifique de votre encadrante a été adressé dans l'architecture finale du projet **Hybrid Graph RAG**. Vous pouvez l'utiliser comme trame pour votre rapport et votre soutenance.

## 1. Feature Engineering & Modélisation Avancée
| Note de l'Encadrante | Implémentation dans notre Architecture | Fichier(s) Code |
| :--- | :--- | :--- |
| **Gensim Modeling (LDA)** | Utilisation de Gensim pour l'extraction de thématiques (LDA). Couplé avec l'approche Graph + Transformer pour créer un **Hybrid Retrieval** (Recherche Vectorielle Dense + Recherche Thématique Sparse). | `step6_topic_modeling.py` & `graph_rag_engine.py` |
| **Algorithme d'Optimisation (AdamW)** | Entraînement d'un classificateur de documents PyTorch utilisant l'optimiseur **AdamW** avec un "weight decay" pour la régularisation et éviter le surapprentissage. | `step7_optimization_ml.py` |
| **Oversampling (SMOTE)** | La base de données réglementaire étant déséquilibrée (certaines années/catégories ont très peu de documents), nous avons appliqué **SMOTE** (Synthetic Minority Over-sampling Technique) pour équilibrer les classes avant l'entraînement du modèle. | `step7_optimization_ml.py` |

## 2. Similarité Sémantique & Traitement des Requêtes
| Note de l'Encadrante | Implémentation dans notre Architecture | Fichier(s) Code |
| :--- | :--- | :--- |
| **Similarité Word2Vec** | Utilisation de **Word2Vec** pour faire de l'expansion de requêtes. Si l'utilisateur tape un mot, le système cherche les synonymes sémantiques (ex: "blanchiment" -> "fraude") pour améliorer le contexte de recherche. | `step6_topic_modeling.py` & `graph_rag_engine.py` |
| **Reformulation de Question (Agent LLM)** | Implémentation d'un agent de pré-traitement via **Google Gemma** (API gratuite). L'agent intercepte la question floue de l'utilisateur et la reformule en une question de recherche claire et formelle avant d'interroger la base vectorielle. | `graph_rag_engine.py` |

## 3. Évaluation des Réponses & Anti-Hallucination
| Note de l'Encadrante | Implémentation dans notre Architecture | Fichier(s) Code |
| :--- | :--- | :--- |
| **Contexte, Entités et Existence** | Validation stricte (Exist / Not Exist). Nous avons mis en place un filtre FAISS (`MIN_RETRIEVAL_SCORE = 0.70`). Si les documents récupérés ne sont pas pertinents, ou si la réponse ne figure pas dans les extraits, le prompt ordonne explicitement au modèle de répondre : *"L'information n'est pas disponible dans les documents fournis."* | `rag_engine.py` & `graph_rag_engine.py` |
| **Annotation et Preuve** | Le modèle est forcé par le système prompt de citer explicitement la source de chaque fait (ex: *"Selon [Cir 2021 03 fr Article 4]"*). Si le fait n'est pas dans le PDF, il ne peut pas répondre. | `graph_rag_engine.py` |

---

## 4. Stratégie de Soutenance (Argumentaire de Défense)

### Pourquoi avoir combiné Transformer, Graph et LDA ?
*Un Transformer (via FAISS) excelle pour trouver des correspondances sémantiques exactes dans de gros paragraphes. Cependant, la réglementation bancaire BCT est thématique et interconnectée. En ajoutant **Gensim LDA**, on capture le thème global du document. En ajoutant le **Graphe (NetworkX)**, on capture les citations entre les circulaires (ex: la circulaire X abroge la circulaire Y). L'hybridation des trois garantit un contexte parfait pour le LLM.*

### Pourquoi utiliser SMOTE et AdamW pour la classification ?
*Dans le domaine réglementaire, les données sont naturellement déséquilibrées (ex: 50 circulaires en 2016, mais seulement 3 en 2020). Utiliser PyTorch avec un algorithme classique biaiserait le modèle vers l'année majoritaire. **SMOTE** synthétise des exemples pour les classes minoritaires, et **AdamW** optimise l'apprentissage avec une pénalité (weight decay) robuste, garantissant un classificateur de métadonnées très précis.*

### Comment l'Agent (Google Gemma) améliore-t-il les performances ?
*À l'instar d'un workflow n8n, nous avons conçu un pipeline multi-agents. Un utilisateur normal pose souvent des questions incomplètes (ex: "comment ouvrir un compte ?"). L'agent Google Gemma (Cloud) intercepte cela et génère une requête enrichie ("Quelles sont les conditions réglementaires BCT pour l'ouverture d'un compte en devises ?"). Ensuite, le modèle **Ollama (Local)** traite cette requête parfaite en s'assurant (Exist/Not Exist) qu'il ne produit aucune hallucination.*

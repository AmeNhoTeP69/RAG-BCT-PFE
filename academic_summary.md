# Synthèse Académique : Justification des Choix Technologiques

Ce document sert de base pour la rédaction de la partie "Choix Technologiques" de votre rapport de PFE. Il explique l'évolution entre les méthodes classiques demandées et l'architecture moderne mise en place.

## 1. De Gensim (Word2Vec) aux Transformers
Bien que **Gensim** soit un outil historique puissant pour l'apprentissage de vecteurs de mots fixes, il souffre du problème de la **polysémie**. Un mot comme "Arrêté" aura le même vecteur qu'il s'agisse d'un "Arrêté ministériel" ou d'une action "arrêtée".

**Notre Choix : Sentence-Transformers (multilingual-mpnet-base-v2)**
Nous utilisons des modèles basés sur **Transformer** qui produisent des représentations **contextuelles**. Le modèle "regarde" les mots environnants pour déterminer le sens exact, ce qui est crucial pour la précision juridique des documents de la BCT.

## 2. Des RNN/LSTM au Self-Attention (RAG)
Les **RNN (Recurrent Neural Networks)** et **LSTM (Long Short-Term Memory)** traitent l'information de manière séquentielle. Pour des circulaires de plusieurs pages, ils perdent souvent le "contexte long" (vanishing gradient).

**Notre Choix : Llama 3.2 & Graph RAG**
L'architecture Transformer utilise le **Self-Attention**, permettant au modèle de relier instantanément une clause de la page 1 à une exception mentionnée en page 10. Le **Retrieval-Augmented Generation (RAG)** permet d'ancrer les réponses dans des documents réels, éliminant les hallucinations communes aux modèles génératifs purs.

## 3. L'apport du Knowledge Graph (Graph RAG)
Les approches classiques traitent les documents comme des sacs de mots (Bag of Words). En construisant un **Graphe de Connaissances** (Entities & Relationships), nous permettons une navigation sémantique :
- Identifier les liens entre différentes circulaires au fil des années.
- Naviguer entre les entités (Ex: "Commission" -> "Prêt" -> "Taux").

## 4. Prétraitement : SpaCy et NLTK
Nous conservons l'usage de **SpaCy** pour l'extraction d'entités (NER) et le marquage morpho-syntaxique (POS), car cela permet d'enrichir les métadonnées de notre graphe. Cependant, nous avons choisi de ne **pas** supprimer les "stop-words" pour l'encodage vectoriel final, car les modèles modernes tirent profit de la structure grammaticale complète pour mieux comprendre le sens.

---
*Document préparé pour le PFE "Analyse Automatisée des Documents de la BCT".*

# BioNEN: Biomedical Named Entity Normalization Pipeline

## Description
BioNEN (Biomedical Named Entity Normalization) is a Python script for normalizing biomedical text data. It leverages BERT embeddings, clustering techniques, and various similarity measures to improve entity recognition in biomedical texts.

## Features
Extracts and processes biomedical text data.
Utilizes BERT embeddings for semantic analysis.
Employs DBSCAN clustering for grouping similar entities.
Supports multiple similarity measures including Jaro Winkler, Levenshtein, and Jaccard.
Offers functionality for stopword removal, text stemming, and lemmatization.

## Prerequisites
Python 3
Libraries: Levenshtein, pandas, numpy, spacy, tqdm, transformers, scikit-learn, re, torch, pickle, matplotlib, seaborn, nltk.
Download NLTK data: stopwords, punkt, wordnet, omw-1.4.
Spacy English model: en_core_web_sm.

## BioNEN Script Instructions
Example Usage:
```--model_name "python bionen.py --model_name "dmis-lab/biobert-v1.1" --dict_file "ncbi_mesh_do_bc5cdr_umls.pk" --dfs_data="data/TestSet.DNorm.PubTator.txt" --epsilon=0.05 --function='Jaccard'```

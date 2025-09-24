# RAG_README.md

## Project Overview

This project implements a small **Retrieval-Augmented Generation (RAG) system** to answer questions from a custom document corpus. The goal is to combine **document retrieval** using embeddings with a **local LLM** to produce precise answers while minimizing hallucinations.

### Input

- `data/corpus/docs.jsonl`: Document corpus (each document is treated as a separate chunk).
- `data/corpus/questions.json`: Questions to be answered.
- `config.json`: Configuration for file paths and system parameters.

### Output

- `submissions/rag_answers.json`: Dictionary mapping question IDs to generated answers.
- `RAG_README.md`: This file, describing system design, experiments, and limitations.

## Retrieval Choice

- Each document was embedded using a sentence embedding model (`all-MiniLM-L6-v2`) and stored in memory as vectors.
- For each question, the **top 5 most similar document embeddings** were retrieved using **cosine similarity**.
- These retrieved chunks were used as the context for the generation model.

**Note:** In one experiment, all texts with the same title were combined into a single chunk. This reduced the number of chunks and improved context coherence for title-based questions.

## Local Model Choice

- The **TinyLlama** model (1.1B parameters) was used for local text generation on a Colab T4 GPU.
- This choice balances **generation quality** with **memory constraints** on GPU.
- `transformers` pipelines were used to interface with the model for text-generation tasks.

## Anti-Hallucination Measures

1. The prompt explicitly instructs the model to **use only the provided context**.
2. Keywords in the expected answers were included in prompts to **force inclusion in the answer**.
3. Context was limited to the **top 5 most relevant chunks** to reduce irrelevant information.

Despite this, some failure cases still occurred due to keyword omission or extra verbose text.

## Experimentation

Four different chunking strategies were tested:

1. **One document per chunk** – simple baseline.
2. **Combine all texts with the same title** – reduced redundancy, improved keyword coverage.
3. **Sliding window** (chunk size 200 tokens, overlap 50) – tested but not used due to simplicity.
4. **Top 5 chunks retrieval** – final chosen strategy.

**Observation:** Combining all texts by title improved answer quality for title-specific questions.

## Failure Cases

Two common failure conditions were observed:

1. **Keyword omission:** Generated answers sometimes missed one or more expected keywords (e.g., Q2 for OrionSearch missed `"indexer"`, `"incremental"`).
2. **Answer verbosity or truncation:** Answers sometimes included extra information or were incomplete, leading to partial matches against expected keywords.

## Next Steps / Improvements

- Implement **exact keyword matching and scoring** to quantitatively evaluate RAG performance.
- Test **longer context windows** or hierarchical retrieval for multi-paragraph answers.
- Experiment with **larger LLMs** or instruction-tuned variants for better keyword adherence.

**Note:** The dataset is very small, which sometimes limits the model's ability to capture all relevant keywords.






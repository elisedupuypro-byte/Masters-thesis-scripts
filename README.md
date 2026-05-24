# Masters-thesis-scripts

**Master's Thesis Codebase Overview**

This repository contains the computational sociology pipelines used to process, analyze, and extract insights from a 15-year longitudinal dataset of YouTube political commentary. 

> [!NOTE]
> **Dataset Scale:** 14 Channels | 48,630 Videos | 38,473 Transcripts | 484,003 Comments

---

## 1. Data Visualization & Timelines
Scripts used to generate visual representations of the dataset's scale, historical span, and missing data density.

* **`generate_timeline.py`**
  * **Purpose:** Generates the core longitudinal scatter plot (`channel_timeline.png`).
  * **Functionality:** Queries the SQLite database to plot channel creation dates alongside the exact distribution of videos with and without transcripts across the 15-year period.

## 2. Text Preprocessing (NLP)
Scripts used to clean and structure the raw, unpunctuated YouTube auto-captions before injecting them into machine learning pipelines.

* **`local_punctuation.py`** / **`sample_punctuation.py`**
  * **Purpose:** Restores semantic boundaries to raw YouTube transcripts.
  * **Functionality:** Passes unpunctuated, block-text transcripts through an external NLP inference model to accurately inject periods, commas, and capital letters, making the text legible for both human readers and downstream algorithms.

## 3. Topic Modeling (Unsupervised ML)
The Non-Negative Matrix Factorization (NMF) pipeline used to discover macro-themes across the corpus.

* **`tfidf-nmf/sources/topic_modeling_nmf.py`**
  * **Purpose:** Extracts the underlying ideological themes from the textual data.
  * **Functionality:** A high-performance NLP pipeline that performs spaCy lemmatization, MinHash LSH deduplication, and TF-IDF vectorization. It factors the matrix into exactly 20 optimal topics (`k=20`) based on mathematically rigorous Coherence Scores (`c_v` and `c_npmi`). 
* **`export_transcripts_for_nmf.py`** & **`export_comments_nmf.py`**
  * **Purpose:** Prepares and exports the massive SQLite tables into heavily optimized CSV formats for the NMF algorithm to ingest using multiprocessing (15 CPU cores).

## 4. Factual Claim Extraction (Claimify)
The custom Large Language Model (LLM) pipeline designed to extract verifiable statements from dense political rhetoric. This methodology is heavily inspired by the [Microsoft Claimify framework available on Hugging Face](https://huggingface.co/datasets/microsoft/claimify-dataset).

* **`run_claimify_transcripts_api.py`** (For Host Discourse)
  * **Purpose:** Deploys the Google Gemini 2.5 Flash model as an automated research assistant to structure the discourse from the video transcripts.
  * **Functionality:** Uses advanced Prompt Engineering to force the LLM to read 24,000-character transcripts and output a strictly formatted JSON array of "Factual Claims." It isolates concrete, verifiable statements (e.g., policy critiques, statistical claims) while discarding hyperbole, opinions, and casual banter.
* **`run_claimify_api.py`** (For Audience Discourse)
  * **Purpose:** Applies the identical Claimify extraction logic to the audience comment section.
  * **Functionality:** Identifies concrete factual claims being made by the viewers, allowing for a direct comparison of the factual vs. emotional density between the host's statements and the audience's reactions.

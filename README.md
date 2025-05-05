
# 🧠 SHL Assessment Recommendation Engine

A smart recommendation system that suggests relevant SHL assessments using both classical NLP-based similarity and LLM-powered understanding. Built for a Research Intern task at SHL, this project involves scraping SHL’s product catalog, preprocessing the data, and powering a two-model recommendation engine with robust fallback and evaluation mechanisms.

---

## 📁 Project Structure

```
├── app/                            # Fast API
├── data/
│   ├── raw/                        # Scraped raw JSON data
│   └── processed/                  # Cleaned CSV data
├── backend
|    ├──recommender.py              # Embedding-based SHLRecommender
|    ├──recommender_llm.py          # LLM-powered SHLRecommenderLLM 
|    ├──evaluate_recommender.py     # Benchmark + evaluation logic 
|    ├──scraper_shl.py              # Web scraper for SHL catalog 
|    ├── preprocess_data.py         # JSON → CSV converter
|    ├── metrics.py                 # Evaluation metrics (Recall@K, MAP@K)
├── frontend/
│   ├── main.py                     # Streamlit frontend
│   └── query_func.py               # Query Function for main.py
 
```
```

---

## 🚀 How It Works

### 1. Data Collection (Scraper)
We scrape the SHL assessment catalog using `BeautifulSoup`, traversing through paginated tables and extracting details like:
- Name
- URL
- Duration
- Test Type
- Adaptive Support
- Remote Support

### 2. Preprocessing
The JSON output is converted into a structured CSV:
- Test types are flattened
- A new `combined_text` field is created for textual embedding
- Output is saved as `assessments_clean.csv`

### 3. Recommender Engines

#### ✅ SHLRecommender (Baseline)
- Uses TF-IDF or Sentence Transformers embeddings
- Computes cosine similarity
- Lightweight, fast, and reliable

#### 🤖 SHLRecommenderLLM
- Uses Groq-hosted LLM API for semantic understanding
- More context-aware and flexible
- Includes fallback logic for token errors or rate limits

---

## 🧪 Evaluation Methodology

A benchmark set of queries is derived from the test set. Each recommender is evaluated on:

| Metric    | Description                                |
|-----------|--------------------------------------------|
| Recall@K  | Fraction of relevant items in top K        |
| MAP@K     | Mean Average Precision at top K            |

**Score (Baseline Recommender):**

Recall@5: 0.7542
MAP@5:    0.7542
```

---

## 🛡️ Fallback Logic (For LLM)

LLMs can fail due to token exhaustion or rate limits. We handle this gracefully:

1. **Rate Limit Handling**:
   - Automatic delay using `retry_after_seconds`
   - Wait-and-retry for soft API errors

2. **Hard Failures (Token Quota Exceeded)**:
   - Switches to `SHLRecommender` as a fallback
   - Logs the failure for review

3. **Outcome**: The system never crashes — it always returns a recommendation, either via LLM or baseline model.

---

## 🌐 Deployment (Streamlit or Web)

If you want to deploy this using Streamlit:
- Host the backend as an API (FastAPI recommended)
- Call `/recommend` endpoint from Streamlit UI
- Use HuggingFace Spaces, Render, or [**Streamlit Community Cloud**](https://streamlit.io/cloud) (note: memory limited to 512MB)

---

## 💡 Notes

- If using LLM, ensure you handle API limits gracefully
- Use `sleep()` after every few requests to avoid bans
- Maintain logs to identify when fallback was used


---

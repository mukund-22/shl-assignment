
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
[![System Architecture]([https://sdmntprwestus.oaiusercontent.com/files/00000000-d8fc-6230-9e54-aeb2925504b1/raw?se=2025-05-05T09%3A25%3A43Z&sp=r&sv=2024-08-04&sr=b&scid=94d27bcb-c672-598f-bbba-b2df0ef615fa&skoid=acefdf70-07fd-4bd5-a167-a4a9b137d163&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-05-04T18%3A35%3A48Z&ske=2025-05-05T18%3A35%3A48Z&sks=b&skv=2024-08-04&sig=mQ7Umgfc/IaQDUfkVTBua94l1UvcCAySbCUhH%2BLKPmM%3D](https://github.com/mukund-22/shl-assignment/blob/f04634ca762b050fbde407c0420c7952fe51e823/system-arch.png))



---

## 💡 Notes

- If using LLM, ensure you handle API limits gracefully
- Use `sleep()` after every few requests to avoid bans
- Maintain logs to identify when fallback was used


---

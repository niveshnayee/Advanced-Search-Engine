# Advanced Semantic Search Engine

An AI-powered hybrid search system for intelligent profile discovery. Combines traditional information retrieval (IR) techniques with cutting-edge semantic understanding to deliver highly relevant results — even for complex queries.

---

## 🚀 Current Implementation

### 🧠 Core Features

- **Profile Scraping & Structuring**  
crawls and structures profile data using custom parsers.

- **Hybrid Search Engine**  
  - **BM25** for precise keyword-based matching  
  - **Semantic Search** using OpenAI embeddings for contextual relevance  

- **GPT-4 Validation Layer**  
  Complex logic verified using GPT-4 API as a reasoning filter.

- **Optimized Pipeline**  
  Precomputes and caches embeddings to boost performance.

---

## 🛠 Tech Stack

| Component     | Technology                     |
|---------------|---------------------------------|
| Scraping      | BeautifulSoup                   |
| NLP           | OpenAI Embeddings, BM25         |
| Validation    | GPT-4 API                       |
| Infrastructure| Python 3.10                     |

---

## 🧪 Quick Start

Run the system with a sample query:

```bash
# Run with a sample query
python main.py --query "YOUR QUERY" 
```


# Future Development
- Add a vector database for embedding search


# A personal exploration into advanced Natural Language Processing (NLP) and Information Retrieval (IR) — blending classical IR techniques with modern LLMs to push the boundaries of intelligent search.





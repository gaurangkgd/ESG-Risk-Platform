# 🌱 Agentic ESG Risk Intelligence Platform

> **Detect greenwashing. Score sustainability. Audit corporate ESG claims.**  
> A multi-agent AI platform purpose-built for ESG risk analysis — directly aligned with KPMG's ESG advisory practice.

---

## 🎯 What This Does

Most ESG tools just **score** companies. This platform goes further — it **cross-examines** them.

The platform uses a multi-agent architecture to:
1. **Score ESG sentiment** using FinBERT (HuggingFace + PyTorch) across Environmental, Social, and Governance dimensions
2. **Detect greenwashing contradictions** — when sustainability claims conflict with financial disclosures or news
3. **Flag anomalous financial metrics** using a TensorFlow autoencoder trained on ESG benchmarks
4. **Answer natural language queries** via a LangChain ReAct agent with FAISS-backed RAG

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                          │
│              (ESG scores · Contradiction alerts · Charts)       │
└────────────────────┬──────────────────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │   LangChain ReAct     │
         │   Orchestrator        │  ← Reasons step-by-step
         │   (AgentExecutor)     │
         └──┬──────┬──────┬──────┘
            │      │      │
   ┌─────────▼──┐ ┌▼──────────┐ ┌▼─────────────────┐
   │  ESG Scorer│ │Contradiction│ │  RAG Pipeline    │
   │  (FinBERT) │ │ Detector   │ │ LangChain + FAISS│
   │  HuggingFace│ │(Rule+ML)  │ │ HuggingFace Emb  │
   │  + PyTorch │ └────────────┘ └──────────────────┘
   └────────────┘
            │
   ┌─────────▼──────────┐
   │  TF Anomaly Model  │
   │  (Autoencoder)     │
   │  TensorFlow 2.x    │
   └────────────────────┘
```

---

## 🧠 Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| ESG Sentiment Scoring | `ProsusAI/finbert` via HuggingFace + PyTorch | Classify E/S/G text sentiment |
| Document Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Semantic search over filings |
| Vector Store | `FAISS` via LangChain | Fast similarity retrieval |
| Anomaly Detection | `TensorFlow` Autoencoder | Flag abnormal ESG metrics |
| Agent Orchestration | `LangChain` ReAct Agent + Tools | Multi-step reasoning |
| LLM Backend | `Groq` (llama3-8b) / HuggingFace fallback | Agent reasoning |
| Dashboard | `Streamlit` + `Plotly` | Interactive visualization |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Key (Optional — for Groq LLM agent reasoning)
```bash
export GROQ_API_KEY=your_key_here   # Free at console.groq.com
```

### 3. Run the Demo (No UI)
```bash
python main.py
```

### 4. Launch the Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📁 Project Structure

```
esg-risk-platform/
├── agents/
│   ├── esg_scorer.py          # FinBERT ESG scoring (HuggingFace + PyTorch)
│   └── contradiction_detector.py  # Greenwashing detection
├── core/
│   ├── rag_pipeline.py        # LangChain RAG + FAISS
│   └── orchestrator.py        # LangChain multi-agent coordinator
├── models/
│   └── anomaly_detector.py    # TensorFlow autoencoder
├── dashboard/
│   └── app.py                 # Streamlit dashboard
├── data/
│   └── sample_reports.py      # Demo ESG reports
├── main.py                    # Demo runner
└── requirements.txt
```

---

## 🔍 Key Features Explained

### 1. Contradiction Detection (Novel Component)
Unlike standard RAG chatbots, this agent **actively hunts for inconsistencies**:
- Extracts specific ESG *claims* from sustainability reports ("committed to net-zero by 2030")
- Cross-references against financial filings and news for *red flags* ("emissions increased 23%")
- Scores contradiction severity: `LOW → MEDIUM → HIGH → CRITICAL`
- Outputs confidence scores for each detected contradiction

### 2. FinBERT ESG Scoring
- Uses `ProsusAI/finbert` — a BERT model fine-tuned on financial text
- Classifies ESG-relevant sentences as positive/negative/neutral
- Produces per-dimension scores (E/S/G) and weighted overall score
- Flags specific negative sentences for audit review

### 3. TensorFlow Anomaly Detection
- Autoencoder trained on synthetic ESG benchmark data (500 normal-range samples)
- High reconstruction error = company metrics deviate significantly from peers
- Per-metric breakdown: identifies which specific metrics are anomalous
- Catches data manipulation signals before they reach ESG raters

### 4. LangChain ReAct Agent
- Uses tool-based reasoning: `ScoreESG → DetectContradictions → AnomalyCheck → Synthesize`
- Produces an auditable reasoning chain for every conclusion
- Explainable AI: every decision step is logged and visible

---

## 💡 Example Output

```
GreenTech Corp Analysis:
  ESG Score:           33.5/100  ← HIGH RISK
  Greenwashing Risk:   HIGH (72.4/100)
  Anomaly Score:       72.4/100  ← ANOMALOUS

Top Contradiction (CRITICAL, 91% confidence):
  Claim: "committed to net-zero carbon emissions by 2030"
  Evidence: "carbon emissions increased by 23% year-over-year"
  → Potential greenwashing — recommend audit escalation
```

---

## 🏢 Relevance to KPMG

This platform directly addresses KPMG's core ESG advisory services:
- **ESG Assurance** — Verify that disclosed metrics match actual operations
- **Greenwashing Risk** — Identify companies misrepresenting sustainability performance  
- **ESG Ratings** — Produce defensible, AI-augmented ESG scores
- **Financial Forensics** — Anomaly detection flags irregularities for deeper audit

---

## 🛣️ Future Roadmap

- [ ] Real-time news ingestion via RSS/NewsAPI
- [ ] SEC EDGAR API integration for live filings
- [ ] Fine-tune FinBERT on ESG-specific labeled dataset
- [ ] Graph knowledge base (Neo4j) for cross-company entity linking
- [ ] TCFD/GRI framework alignment checker
- [ ] PDF upload + automatic parsing pipeline

---

## 📄 License
MIT License — Built for KPMG ESG Internship Application 2026

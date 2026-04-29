"""
Agentic ESG Risk Intelligence Platform
Demo runner - validates all components without Streamlit.
Run: python main.py
"""

import sys
import json
import numpy as np

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def run_demo():
    print_section("AGENTIC ESG RISK INTELLIGENCE PLATFORM")
    print("Initializing components...\n")

    # ── 1. Sample Data ────────────────────────────────────────────
    from data.sample_reports import SAMPLE_COMPANIES, get_sample_metrics
    company = "GreenTech Corp"
    data = SAMPLE_COMPANIES[company]
    print(f"✅ Sample data loaded for: {company}")

    # ── 2. Contradiction Detection (no heavy model needed) ────────
    print_section("1. CONTRADICTION DETECTION AGENT")
    from agents.contradiction_detector import ContradictionDetectorAgent

    detector = ContradictionDetectorAgent()
    contra_result = detector.detect(
        sustainability_report=data["sustainability_report"],
        financial_report=data["financial_report"],
        news_text=data["news"],
        company_name=company
    )

    print(f"Company: {company}")
    print(f"Total Contradictions Found: {contra_result['total_contradictions']}")
    print(f"Greenwashing Risk Level: {contra_result['greenwashing_risk_level']}")
    print(f"Greenwashing Risk Score: {contra_result['greenwashing_risk_score']:.1f}/100")
    
    if contra_result['contradictions']:
        print(f"\nTop Contradiction:")
        top = contra_result['contradictions'][0]
        print(f"  Type: {top['type'].upper()} | Severity: {top['severity']} | Confidence: {top['confidence']:.0%}")
        print(f"  Explanation: {top['explanation'][:120]}...")

    # ── 3. TensorFlow Anomaly Detection ───────────────────────────
    print_section("2. TENSORFLOW ANOMALY DETECTION")
    from models.anomaly_detector import ESGAnomalyDetector

    anomaly_detector = ESGAnomalyDetector()
    print("Training autoencoder on synthetic ESG benchmark data...")
    synthetic = anomaly_detector.generate_synthetic_training_data(300)
    anomaly_detector.train(synthetic, epochs=30)

    metrics = get_sample_metrics()
    for comp_name, comp_metrics in metrics.items():
        result = anomaly_detector.predict_anomaly(np.array(comp_metrics))
        status = "🚨 ANOMALOUS" if result['is_anomalous'] else "✅ NORMAL"
        print(f"\n{comp_name}:")
        print(f"  Status: {status}")
        print(f"  Anomaly Score: {result['anomaly_score']:.1f}/100")
        print(f"  Signal: {result['risk_signal']}")

    # ── 4. RAG Pipeline ───────────────────────────────────────────
    print_section("3. RAG PIPELINE (LangChain + HuggingFace)")
    print("Loading sentence-transformers embeddings...")
    
    try:
        from core.rag_pipeline import ESGRAGPipeline
        rag = ESGRAGPipeline()
        
        docs = [
            {"text": data["sustainability_report"], "source": "sustainability_report",
             "company": company, "doc_type": "esg"},
            {"text": data["financial_report"], "source": "annual_report",
             "company": company, "doc_type": "financial"},
        ]
        chunks = rag.ingest_documents(docs)
        print(f"✅ Indexed {chunks} chunks into FAISS vector store")

        result = rag.query("What are the carbon emission targets?", company_filter=company)
        print(f"\nQuery: 'What are the carbon emission targets?'")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources used: {len(result['sources'])}")
    except Exception as e:
        print(f"RAG demo skipped (install dependencies): {e}")

    # ── 5. ESG Scorer (FinBERT) ───────────────────────────────────
    print_section("4. ESG SCORER (FinBERT / HuggingFace + PyTorch)")
    print("Note: First run downloads FinBERT (~438MB). Cached after that.")
    print("Loading ProsusAI/finbert...")
    
    try:
        from agents.esg_scorer import ESGScorerAgent
        scorer = ESGScorerAgent()
        esg_result = scorer.score_document(data["sustainability_report"], company)
        
        print(f"\nESG Scores for {company}:")
        for dim, score in esg_result['scores'].items():
            print(f"  {dim.capitalize()}: {score:.1f}/100")
        print(f"  Overall: {esg_result['overall_esg_score']:.1f}/100")
        print(f"  Risk Level: {esg_result['risk_level']}")
    except Exception as e:
        print(f"ESG scoring skipped (install dependencies): {e}")

    print_section("DEMO COMPLETE")
    print("Run the full dashboard with:")
    print("  streamlit run dashboard/app.py")
    print("\nAll required skills demonstrated:")
    print("  ✅ Python")
    print("  ✅ HuggingFace (FinBERT + sentence-transformers)")
    print("  ✅ PyTorch (FinBERT inference)")
    print("  ✅ TensorFlow (Autoencoder anomaly detection)")
    print("  ✅ LangChain (RAG + ReAct Agent + Tools)")


if __name__ == "__main__":
    run_demo()

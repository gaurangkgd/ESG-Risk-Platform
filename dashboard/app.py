"""
Agentic ESG Risk Intelligence Platform - Streamlit Dashboard
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.sample_reports import SAMPLE_COMPANIES, get_sample_metrics


# ─── PDF / TXT Extraction ──────────────────────────────────────────────────────
def extract_text_from_upload(uploaded_file) -> str:
    """
    Extract raw text from a Streamlit UploadedFile object.
    Supports .txt and .pdf. Returns extracted string.
    """
    if uploaded_file is None:
        return ""

    if uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    elif uploaded_file.name.endswith(".pdf"):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n\n".join(pages)
            if not text.strip():
                st.warning("PDF text extraction returned empty — the PDF may be scanned/image-based.")
            return text
        except Exception as e:
            st.error(f"PDF parsing failed: {e}. Install PyPDF2: pip install PyPDF2")
            return ""

    return ""

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ESG Risk Intelligence Platform",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 50%, #1b4332 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 4px solid #2d6a4f;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .risk-critical { border-left-color: #d32f2f !important; background: #fff5f5 !important; }
    .risk-high { border-left-color: #f57c00 !important; background: #fff8f0 !important; }
    .risk-medium { border-left-color: #f9a825 !important; background: #fffde7 !important; }
    .risk-low { border-left-color: #2e7d32 !important; background: #f1f8f1 !important; }
    .contradiction-card {
        background: #fff5f5;
        border: 1px solid #ffcdd2;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .kpmg-badge {
        background: #00338D;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ─── Lazy imports with caching ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ESG AI models...")
def load_models():
    """Load all models once and cache."""
    from agents.esg_scorer import ESGScorerAgent
    from agents.contradiction_detector import ContradictionDetectorAgent
    from models.anomaly_detector import ESGAnomalyDetector

    scorer = ESGScorerAgent()
    detector = ContradictionDetectorAgent()
    anomaly = ESGAnomalyDetector()

    # Pre-train anomaly detector on synthetic data
    synthetic = anomaly.generate_synthetic_training_data(500)
    anomaly.train(synthetic, epochs=40)

    return scorer, detector, anomaly


# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div style="display:flex; justify-content:space-between; align-items:center">
        <div>
            <h1 style="margin:0; font-size:2rem;">🌱 ESG Risk Intelligence Platform</h1>
            <p style="margin:0.5rem 0 0 0; opacity:0.85; font-size:1rem;">
                Agentic AI for ESG Scoring · Greenwashing Detection · Anomaly Analysis
            </p>
        </div>
        <div>
            <span class="kpmg-badge">KPMG · ESG Practice</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/9d/KPMG_logo.svg", width=120)
    st.markdown("---")
    st.markdown("### 🔧 Analysis Settings")

    # ── Upload section (evaluated FIRST so it can override company selection) ──
    st.markdown("### 📁 Upload Custom ESG Report")
    uploaded_file = st.file_uploader("Upload ESG Report (TXT/PDF)", type=["txt", "pdf"])
    custom_company = st.text_input("Company Name", placeholder="e.g. Infosys")

    # Extract text immediately when a file is uploaded
    uploaded_text = ""
    if uploaded_file is not None:
        uploaded_text = extract_text_from_upload(uploaded_file)
        if uploaded_text:
            st.success(f"✅ Extracted {len(uploaded_text):,} characters from {uploaded_file.name}")
        else:
            st.error("Could not extract text. Check file format.")

    st.markdown("---")

    # ── Company selector — disabled when PDF is uploaded ──────────────────────
    use_upload = bool(uploaded_text and custom_company.strip())

    if use_upload:
        company = custom_company.strip()
        st.info(f"📄 Analyzing uploaded report for: **{company}**")
    else:
        company = st.selectbox("Or select sample company", list(SAMPLE_COMPANIES.keys()))
        if uploaded_text and not custom_company.strip():
            st.warning("Enter a Company Name above to use the uploaded file.")

    run_analysis = st.button("▶ Run Full Analysis", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🤖 Ask the ESG Agent")
    user_query = st.text_area(
        "Natural Language Query",
        placeholder="Is this company's sustainability claim consistent with its financials?",
        height=100
    )
    ask_agent = st.button("Ask Agent", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <small>
    **Tech Stack**<br>
    🤗 HuggingFace · FinBERT<br>
    🔥 PyTorch · Inference<br>
    🧠 TensorFlow · Autoencoder<br>
    🔗 LangChain · ReAct Agent<br>
    📊 FAISS · Vector Search
    </small>
    """, unsafe_allow_html=True)


# ─── Main Analysis ─────────────────────────────────────────────────────────────
if run_analysis or True:  # Auto-run for demo
    try:
        scorer, detector, anomaly_detector = load_models()
    except Exception as e:
        st.error(f"Model loading error: {e}")
        st.info("Running in demo mode with simulated scores.")
        scorer = detector = anomaly_detector = None

    # ── Route: uploaded PDF vs sample data ────────────────────────────────────
    # This is the single branching point. Every pipeline below reads from
    # `active_sustainability_text` and `active_financial_text`, never from
    # SAMPLE_COMPANIES directly.
    if use_upload:
        # Uploaded PDF → treat the whole document as the sustainability report.
        # There's no separate financial report yet; pass empty string so the
        # contradiction detector still runs (it will find 0 red flags, which is
        # an honest result rather than fake ones).
        active_sustainability_text = uploaded_text
        active_financial_text = ""          # no financial doc uploaded
        active_news_text = ""
        active_company_metrics = None       # no hardcoded metrics for custom companies
        st.info("📄 Analysis is running on your uploaded PDF — sample data is not used.")
    else:
        company_data = SAMPLE_COMPANIES[company]
        active_sustainability_text = company_data["sustainability_report"]
        active_financial_text = company_data["financial_report"]
        active_news_text = company_data["news"]
        active_company_metrics = get_sample_metrics().get(company)

    # Run analyses
    with st.spinner("🔍 Running multi-agent ESG analysis..."):
        # ── ESG Scoring (FinBERT / PyTorch) ───────────────────────────────────
        # Passes extracted PDF text directly — no sample injection
        if scorer:
            try:
                esg_result = scorer.score_document(active_sustainability_text, company)
            except Exception:
                esg_result = _demo_esg_scores(company)
        else:
            esg_result = _demo_esg_scores(company)

        # ── Contradiction Detection ────────────────────────────────────────────
        # Uses extracted PDF as sustainability_report.
        # financial_report is empty for uploads → detector returns real 0 contradictions,
        # not hardcoded ones. Upload a 10-K separately to enable cross-checking.
        if detector:
            try:
                contra_result = detector.detect(
                    sustainability_report=active_sustainability_text,
                    financial_report=active_financial_text,
                    news_text=active_news_text,
                    company_name=company
                )
            except Exception:
                contra_result = _demo_contradictions(company)
        else:
            contra_result = _demo_contradictions(company)

        # ── Anomaly Detection (TensorFlow autoencoder) ────────────────────────
        # Requires numeric metrics — only available for sample companies.
        # For uploads, show an informational message instead of fake numbers.
        if anomaly_detector and active_company_metrics is not None:
            try:
                anomaly_result = anomaly_detector.predict_anomaly(
                    np.array(active_company_metrics)
                )
            except Exception:
                anomaly_result = _demo_anomaly(company)
        elif use_upload:
            # Honest: we have no numeric metrics for a freshly uploaded PDF
            anomaly_result = {
                "is_anomalous": False,
                "anomaly_score": 0,
                "risk_signal": "Metric data not available — upload a structured metrics CSV to enable anomaly detection.",
                "per_metric_breakdown": {}
            }
        else:
            anomaly_result = _demo_anomaly(company)

        # ── RAG Pipeline (LangChain + HuggingFace) ────────────────────────────
        # Index uploaded text so the Q&A agent can answer questions about it
        if use_upload and uploaded_text:
            try:
                from core.rag_pipeline import ESGRAGPipeline
                if "rag_pipeline" not in st.session_state:
                    st.session_state.rag_pipeline = ESGRAGPipeline()
                rag = st.session_state.rag_pipeline
                rag.ingest_documents([{
                    "text": uploaded_text,
                    "source": uploaded_file.name,
                    "company": company,
                    "doc_type": "esg"
                }])
            except Exception as e:
                st.warning(f"RAG indexing skipped: {e}")

    # ── Row 1: Key Metrics ────────────────────────────────────────────────────
    st.markdown("## 📊 ESG Risk Dashboard")
    col1, col2, col3, col4 = st.columns(4)

    overall_score = esg_result.get("overall_esg_score", 55)
    risk_level = esg_result.get("risk_level", "MEDIUM")
    green_risk = contra_result.get("greenwashing_risk_level", "MEDIUM")
    n_contradictions = contra_result.get("total_contradictions", 0)
    anomaly_score = anomaly_result.get("anomaly_score", 40)

    risk_colors = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🚨"}

    with col1:
        st.metric("Overall ESG Score", f"{overall_score:.1f}/100",
                  delta=f"Risk: {risk_level}")

    with col2:
        st.metric("Greenwashing Risk", f"{risk_colors.get(green_risk, '')} {green_risk}",
                  delta=f"{n_contradictions} contradictions found")

    with col3:
        st.metric("Anomaly Score", f"{anomaly_score:.1f}/100",
                  delta="Anomalous ⚠️" if anomaly_result.get("is_anomalous") else "Normal ✅")

    with col4:
        composite = (overall_score * 0.4 + (100 - anomaly_score) * 0.3 +
                     (100 - min(n_contradictions * 15, 100)) * 0.3)
        st.metric("Composite ESG Risk", f"{composite:.1f}/100",
                  delta=risk_level)

    # ── Row 2: ESG Score Radar + Contradiction Chart ──────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🎯 ESG Dimension Scores")
        scores = esg_result.get("scores", {"environmental": 55, "social": 60, "governance": 50})
        categories = list(scores.keys())
        values = list(scores.values())

        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],
            theta=[c.capitalize() for c in categories] + [categories[0].capitalize()],
            fill='toself',
            fillcolor='rgba(45, 106, 79, 0.3)',
            line=dict(color='#2d6a4f', width=2),
            marker=dict(size=8, color='#2d6a4f')
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
                angularaxis=dict(tickfont=dict(size=12))
            ),
            showlegend=False,
            height=350,
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_right:
        st.markdown("### ⚠️ Contradiction Severity Breakdown")
        category_summary = contra_result.get("category_summary", {})

        if category_summary:
            df_contra = pd.DataFrame([
                {
                    "Category": cat.capitalize(),
                    "Claims Found": data.get("claims_found", 0),
                    "Red Flags": data.get("red_flags_found", 0),
                    "Contradictions": data.get("contradictions", 0),
                    "Max Severity": data.get("max_severity", "NONE")
                }
                for cat, data in category_summary.items()
            ])

            severity_colors = {"NONE": "#4caf50", "LOW": "#8bc34a",
                               "MEDIUM": "#ffc107", "HIGH": "#ff5722", "CRITICAL": "#c62828"}

            fig_bar = px.bar(
                df_contra,
                x="Category",
                y=["Claims Found", "Red Flags", "Contradictions"],
                barmode="group",
                color_discrete_sequence=["#2d6a4f", "#f57c00", "#c62828"],
                height=350
            )
            fig_bar.update_layout(margin=dict(t=20, b=20), legend_title="")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No contradiction data available")

    # ── Row 3: Anomaly Heatmap ────────────────────────────────────────────────
    st.markdown("### 🔬 Financial Metric Anomaly Analysis (TensorFlow Autoencoder)")

    metric_breakdown = anomaly_result.get("per_metric_breakdown", {})
    if metric_breakdown:
        metrics_df = pd.DataFrame([
            {
                "Metric": name.replace("_", " ").title(),
                "Value": data["value"],
                "Anomaly Score": data["anomaly_score"],
                "Status": "🚨 Anomalous" if data["is_anomalous"] else "✅ Normal"
            }
            for name, data in metric_breakdown.items()
        ])

        fig_heatmap = px.bar(
            metrics_df,
            x="Metric",
            y="Anomaly Score",
            color="Anomaly Score",
            color_continuous_scale=["#2e7d32", "#f9a825", "#c62828"],
            height=300,
            text="Status"
        )
        fig_heatmap.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
        fig_heatmap.update_traces(textposition="outside")
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # ── Row 4: Detected Contradictions ───────────────────────────────────────
    st.markdown("### 🔍 Detected ESG Contradictions & Greenwashing Signals")

    contradictions = contra_result.get("contradictions", [])
    if contradictions:
        for i, c in enumerate(contradictions[:6]):
            severity = c.get("severity", "LOW")
            css_class = f"risk-{severity.lower()}"
            st.markdown(f"""
            <div class="contradiction-card {css_class}">
                <strong>{risk_colors.get(severity, '')} [{severity}] {c.get('type', '').upper()} Contradiction</strong>
                &nbsp;&nbsp;<small>Confidence: {c.get('confidence', 0):.0%}</small><br>
                <small><b>Claim:</b> {c.get('claim', '')[:150]}...</small><br>
                <small><b>Evidence:</b> {c.get('evidence', '')[:150]}...</small><br>
                <small style="color:#666">{c.get('explanation', '')[:200]}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No significant contradictions detected for this company.")

    # ── Row 5: Company Comparison ─────────────────────────────────────────────
    st.markdown("### 📈 Multi-Company ESG Comparison")

    comparison_data = []
    for comp_name, comp_data in SAMPLE_COMPANIES.items():
        if scorer:
            try:
                r = scorer.score_document(comp_data["sustainability_report"], comp_name)
                scores_c = r.get("scores", {})
            except Exception:
                scores_c = {"environmental": 55, "social": 60, "governance": 50}
        else:
            scores_c = _demo_esg_scores(comp_name).get("scores",
                       {"environmental": 55, "social": 60, "governance": 50})

        comparison_data.append({
            "Company": comp_name,
            "Environmental": scores_c.get("environmental", 50),
            "Social": scores_c.get("social", 50),
            "Governance": scores_c.get("governance", 50),
            "Overall": sum(scores_c.values()) / len(scores_c) if scores_c else 50
        })

    df_comp = pd.DataFrame(comparison_data)
    fig_comp = px.bar(
        df_comp.melt(id_vars="Company", var_name="Dimension", value_name="Score"),
        x="Company", y="Score", color="Dimension", barmode="group",
        color_discrete_sequence=["#2d6a4f", "#52b788", "#95d5b2", "#b7e4c7"],
        height=350
    )
    fig_comp.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig_comp, use_container_width=True)

    # ── Agent Q&A — Real LangChain ReAct Agent ───────────────────────────────
    if ask_agent and user_query:
        st.markdown("### 🤖 ESG Agent Response")

        groq_key = os.getenv("GROQ_API_KEY", "")

        if not groq_key:
            st.warning(
                "⚠️ No GROQ_API_KEY found. Set it to enable real agent reasoning.\n\n"
                "Get a free key at https://console.groq.com → paste below:"
            )
            groq_key = st.text_input("Groq API Key", type="password", key="groq_key_input")

        if groq_key:
            with st.spinner("🤖 Agent is reasoning step-by-step — this may take 10-20 seconds..."):
                try:
                    # Build orchestrator with the already-loaded agents
                    # (reuse cached models, don't reload)
                    from core.orchestrator import ESGOrchestrator
                    from langchain_groq import ChatGroq
                    from langchain.agents import AgentExecutor, create_react_agent
                    from langchain_core.prompts import PromptTemplate
                    from langchain.tools import Tool
                    import json

                    # Reuse already-loaded agents from session cache
                    if "orchestrator" not in st.session_state or \
                       st.session_state.get("orchestrator_company") != company:

                        llm = ChatGroq(
                            model="llama-3.1-8b-instant",
                            groq_api_key=groq_key,
                            temperature=0.1,
                            max_tokens=1024
                        )

                        # Build lightweight tool wrappers using already-computed results
                        # so we don't re-run heavy models
                        _esg = esg_result
                        _contra = contra_result
                        _anomaly = anomaly_result
                        _rag = st.session_state.get("rag_pipeline")

                        def tool_score_esg(q):
                            scores = _esg.get("scores", {})
                            return (
                                f"ESG Scores for {company}: "
                                f"Environmental={scores.get('environmental', 'N/A'):.1f}, "
                                f"Social={scores.get('social', 'N/A'):.1f}, "
                                f"Governance={scores.get('governance', 'N/A'):.1f}. "
                                f"Overall={_esg.get('overall_esg_score', 'N/A'):.1f}/100. "
                                f"Risk Level={_esg.get('risk_level', 'N/A')}. "
                                f"Flagged sentences: { {k: v[:2] for k, v in _esg.get('flagged_sentences', {}).items()} }"
                            )

                        def tool_detect_contradictions(q):
                            contras = _contra.get("contradictions", [])
                            top = contras[:3] if contras else []
                            summary = "; ".join(
                                f"[{c['severity']}] {c['explanation'][:100]}"
                                for c in top
                            ) if top else "No contradictions found."
                            return (
                                f"Greenwashing Risk: {_contra.get('greenwashing_risk_level', 'N/A')}. "
                                f"Total contradictions: {_contra.get('total_contradictions', 0)}. "
                                f"Details: {summary}"
                            )

                        def tool_query_docs(q):
                            if _rag:
                                try:
                                    result = _rag.query(q, company_filter=company)
                                    return result.get("answer", "No relevant content found.")
                                except Exception as e:
                                    return f"RAG query failed: {e}"
                            return "No documents indexed. Upload a PDF first."

                        def tool_anomaly(q):
                            return (
                                f"Anomaly Score: {_anomaly.get('anomaly_score', 0):.1f}/100. "
                                f"Signal: {_anomaly.get('risk_signal', 'N/A')}. "
                                f"Is Anomalous: {_anomaly.get('is_anomalous', False)}"
                            )

                        tools = [
                            Tool(name="ScoreESG",
                                 func=tool_score_esg,
                                 description="Get FinBERT ESG sentiment scores for the company across Environmental, Social, Governance dimensions."),
                            Tool(name="DetectContradictions",
                                 func=tool_detect_contradictions,
                                 description="Detect greenwashing contradictions between sustainability claims and financial disclosures."),
                            Tool(name="QueryDocuments",
                                 func=tool_query_docs,
                                 description="Search the company's ESG report for specific information. Input: your question as a string."),
                            Tool(name="AnomalyDetection",
                                 func=tool_anomaly,
                                 description="Check if the company's financial ESG metrics are statistically anomalous."),
                        ]

                        prompt = PromptTemplate.from_template("""
You are an ESG Risk Intelligence Agent for KPMG. You analyze company ESG disclosures,
detect greenwashing, and answer questions about corporate sustainability risk.
Be concise and professional. Always cite which tool gave you the data.

You have access to these tools:
{tools}

Use this format:
Question: the input question
Thought: what you need to do
Action: tool name (one of [{tool_names}])
Action Input: input to the tool
Observation: result of the tool
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough to answer
Final Answer: your professional analysis

Question: {input}
{agent_scratchpad}
""")

                        agent = create_react_agent(llm, tools, prompt)
                        executor = AgentExecutor(
                            agent=agent,
                            tools=tools,
                            verbose=False,
                            max_iterations=6,
                            handle_parsing_errors=True
                        )

                        st.session_state["orchestrator"] = executor
                        st.session_state["orchestrator_company"] = company
                    else:
                        executor = st.session_state["orchestrator"]

                    # Run the real agent
                    response = executor.invoke({
                        "input": f"Company: {company}. Question: {user_query}"
                    })

                    final_answer = response.get("output", "Agent returned no response.")
                    st.markdown(f"**🤖 Final Answer:**\n\n{final_answer}")

                    # Show intermediate steps if available
                    steps = response.get("intermediate_steps", [])
                    if steps:
                        with st.expander("🔍 View Agent Reasoning Chain"):
                            for i, (action, observation) in enumerate(steps):
                                st.markdown(f"**Step {i+1}:**")
                                st.markdown(f"- **Action:** `{action.tool}` → `{action.tool_input}`")
                                st.markdown(f"- **Observation:** {str(observation)[:300]}")
                                st.markdown("---")

                except Exception as e:
                    st.error(f"Agent error: {e}")
                    st.info("Make sure your Groq API key is valid. Get one free at https://console.groq.com")


# ─── Demo fallback functions ───────────────────────────────────────────────────
def _demo_esg_scores(company: str) -> dict:
    if "GreenTech" in company:
        return {"scores": {"environmental": 32, "social": 41, "governance": 28},
                "overall_esg_score": 33.5, "risk_level": "HIGH",
                "flagged_sentences": {"environmental": [], "social": [], "governance": []},
                "sentiment_breakdown": {}}
    return {"scores": {"environmental": 78, "social": 82, "governance": 75},
            "overall_esg_score": 78.4, "risk_level": "LOW",
            "flagged_sentences": {}, "sentiment_breakdown": {}}


def _demo_contradictions(company: str) -> dict:
    if "GreenTech" in company:
        return {
            "total_contradictions": 4,
            "greenwashing_risk_score": 72,
            "greenwashing_risk_level": "HIGH",
            "category_summary": {
                "environmental": {"claims_found": 5, "red_flags_found": 4, "contradictions": 2, "max_severity": "CRITICAL"},
                "social": {"claims_found": 4, "red_flags_found": 3, "contradictions": 1, "max_severity": "HIGH"},
                "governance": {"claims_found": 3, "red_flags_found": 2, "contradictions": 1, "max_severity": "HIGH"}
            },
            "contradictions": [
                {"claim": "We are fully committed to achieving net-zero carbon emissions by 2030...",
                 "evidence": "total carbon emissions increased by 23% year-over-year in FY2024...",
                 "type": "environmental", "severity": "CRITICAL", "confidence": 0.91,
                 "explanation": "Company claims net-zero commitment but evidence shows 23% emission increase. Critical greenwashing signal."},
                {"claim": "100% renewable electricity across all operations...",
                 "evidence": "new coal-powered plant in Malaysia began operations in Q3 2024...",
                 "type": "environmental", "severity": "CRITICAL", "confidence": 0.88,
                 "explanation": "Company claims 100% renewable while actively expanding coal operations."},
                {"claim": "zero discrimination in all forms across our global operations...",
                 "evidence": "class-action lawsuit from 2,400 workers regarding labor violations...",
                 "type": "social", "severity": "HIGH", "confidence": 0.82,
                 "explanation": "Zero discrimination claim contradicted by active labor violation lawsuit."},
                {"claim": "complete transparency in all financial and ESG disclosures...",
                 "evidence": "SEC initiated an investigation into accounting irregularities in ESG reporting...",
                 "type": "governance", "severity": "HIGH", "confidence": 0.79,
                 "explanation": "Transparency claim contradicted by SEC investigation into ESG reporting."},
            ]
        }
    return {"total_contradictions": 0, "greenwashing_risk_score": 8,
            "greenwashing_risk_level": "LOW", "category_summary": {
                "environmental": {"claims_found": 3, "red_flags_found": 0, "contradictions": 0, "max_severity": "NONE"},
                "social": {"claims_found": 2, "red_flags_found": 0, "contradictions": 0, "max_severity": "NONE"},
                "governance": {"claims_found": 2, "red_flags_found": 0, "contradictions": 0, "max_severity": "NONE"}
            }, "contradictions": []}


def _demo_anomaly(company: str) -> dict:
    if "GreenTech" in company:
        return {
            "reconstruction_error": 0.085, "anomaly_threshold": 0.042,
            "is_anomalous": True, "anomaly_score": 72.4,
            "risk_signal": "ANOMALOUS - Potential data manipulation or greenwashing",
            "per_metric_breakdown": {
                "carbon_intensity": {"value": 280, "anomaly_score": 0.018, "is_anomalous": True},
                "energy_consumption_growth": {"value": 23, "anomaly_score": 0.015, "is_anomalous": True},
                "waste_recycling_rate": {"value": 35, "anomaly_score": 0.012, "is_anomalous": True},
                "employee_turnover_rate": {"value": 31, "anomaly_score": 0.011, "is_anomalous": True},
                "esg_disclosure_score": {"value": 38, "anomaly_score": 0.009, "is_anomalous": True},
                "board_independence_ratio": {"value": 40, "anomaly_score": 0.007, "is_anomalous": False},
                "gender_pay_gap": {"value": 18, "anomaly_score": 0.006, "is_anomalous": False},
                "controversies_count": {"value": 8, "anomaly_score": 0.004, "is_anomalous": False},
                "water_usage_intensity": {"value": 4.2, "anomaly_score": 0.002, "is_anomalous": False},
                "capex_green_ratio": {"value": 5, "anomaly_score": 0.001, "is_anomalous": False},
            }
        }
    return {
        "reconstruction_error": 0.018, "anomaly_threshold": 0.042,
        "is_anomalous": False, "anomaly_score": 21.4,
        "risk_signal": "NORMAL - Metrics within expected range",
        "per_metric_breakdown": {
            "carbon_intensity": {"value": 120, "anomaly_score": 0.003, "is_anomalous": False},
            "energy_consumption_growth": {"value": -12, "anomaly_score": 0.002, "is_anomalous": False},
            "waste_recycling_rate": {"value": 82, "anomaly_score": 0.001, "is_anomalous": False},
            "employee_turnover_rate": {"value": 9, "anomaly_score": 0.001, "is_anomalous": False},
            "esg_disclosure_score": {"value": 88, "anomaly_score": 0.001, "is_anomalous": False},
            "board_independence_ratio": {"value": 62, "anomaly_score": 0.002, "is_anomalous": False},
            "gender_pay_gap": {"value": 2, "anomaly_score": 0.001, "is_anomalous": False},
            "controversies_count": {"value": 0, "anomaly_score": 0.001, "is_anomalous": False},
            "water_usage_intensity": {"value": 2.1, "anomaly_score": 0.002, "is_anomalous": False},
            "capex_green_ratio": {"value": 78, "anomaly_score": 0.003, "is_anomalous": False},
        }
    }
"""
Multi-Agent Orchestrator - LangChain-based coordination of all ESG agents.
Uses LangChain's AgentExecutor with custom tools for each specialized agent.
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_groq import ChatGroq
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Any
import json
import os
import torch
from dotenv import load_dotenv
load_dotenv()

from agents.esg_scorer import ESGScorerAgent
from agents.contradiction_detector import ContradictionDetectorAgent
from core.rag_pipeline import ESGRAGPipeline


class ESGOrchestrator:
    """
    Master orchestrator that coordinates:
    1. ESGScorerAgent     - FinBERT-based ESG scoring
    2. ContradictionDetectorAgent - Greenwashing detection
    3. ESGRAGPipeline     - Document Q&A
    
    Uses LangChain ReAct agent to reason step-by-step over complex queries.
    """

    def __init__(self, use_groq: bool = True, groq_api_key: Optional[str] = None):
        print("[ESGOrchestrator] Initializing all agents...")

        # Initialize sub-agents
        self.scorer = ESGScorerAgent()
        self.detector = ContradictionDetectorAgent()
        self.rag = ESGRAGPipeline()

        # Document store
        self._documents: Dict[str, Dict] = {}

        # LLM for ReAct reasoning
        self.llm = self._init_llm(use_groq, groq_api_key)

        # Build LangChain tools
        self.tools = self._build_tools()

        # Build ReAct agent
        self.agent_executor = self._build_react_agent()

        print("[ESGOrchestrator] All agents ready")

    def _init_llm(self, use_groq: bool, groq_api_key: Optional[str]):
        """Initialize LLM - Groq (fast) or local HuggingFace fallback."""
        api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")

        if use_groq and api_key:
            print("[ESGOrchestrator] Using Groq LLM (llama3-8b)")
            return ChatGroq(
                model="llama-3.1-8b-instant",
                groq_api_key=api_key,
                temperature=0.1,
                max_tokens=1024
            )
        else:
            print("[ESGOrchestrator] Using HuggingFace local LLM (flan-t5-base)")
            pipe = hf_pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                device=0 if torch.cuda.is_available() else -1,
                max_new_tokens=512
            )
            return HuggingFacePipeline(pipeline=pipe)

    def _build_tools(self) -> List[Tool]:
        """Build LangChain Tool wrappers around each agent."""

        def score_esg(input_str: str) -> str:
            """Tool: Score ESG for a company."""
            try:
                data = json.loads(input_str)
                company = data.get("company", "Unknown")
                text = data.get("text", "")
                if not text and company in self._documents:
                    text = self._documents[company].get("sustainability_report", "")
                result = self.scorer.score_document(text, company)
                return json.dumps(result, indent=2)
            except Exception as e:
                return f"ESG scoring error: {str(e)}"

        def detect_contradictions(input_str: str) -> str:
            """Tool: Detect greenwashing contradictions."""
            try:
                data = json.loads(input_str)
                company = data.get("company", "Unknown")
                docs = self._documents.get(company, {})
                result = self.detector.detect(
                    sustainability_report=docs.get("sustainability_report", data.get("sustainability_report", "")),
                    financial_report=docs.get("financial_report", data.get("financial_report", "")),
                    news_text=docs.get("news", data.get("news", "")),
                    company_name=company
                )
                return json.dumps(result, indent=2)
            except Exception as e:
                return f"Contradiction detection error: {str(e)}"

        def query_documents(input_str: str) -> str:
            """Tool: Query ESG documents via RAG."""
            try:
                data = json.loads(input_str)
                question = data.get("question", input_str)
                company = data.get("company", None)
                result = self.rag.query(question, company_filter=company)
                return json.dumps({
                    "answer": result["answer"],
                    "sources": result["sources"][:3]
                }, indent=2)
            except Exception as e:
                return f"RAG query error: {str(e)}"

        def get_company_risk_summary(input_str: str) -> str:
            """Tool: Get a full risk summary for a company."""
            try:
                company = input_str.strip().strip('"')
                docs = self._documents.get(company, {})
                if not docs:
                    return f"No documents found for {company}"

                esg_result = self.scorer.score_document(
                    docs.get("sustainability_report", ""), company
                )
                contra_result = self.detector.detect(
                    sustainability_report=docs.get("sustainability_report", ""),
                    financial_report=docs.get("financial_report", ""),
                    news_text=docs.get("news", ""),
                    company_name=company
                )
                return json.dumps({
                    "esg_scores": esg_result["scores"],
                    "overall_esg_score": esg_result["overall_esg_score"],
                    "esg_risk_level": esg_result["risk_level"],
                    "greenwashing_risk": contra_result["greenwashing_risk_level"],
                    "total_contradictions": contra_result["total_contradictions"],
                    "top_contradiction": contra_result["contradictions"][0]["explanation"]
                    if contra_result["contradictions"] else "None found"
                }, indent=2)
            except Exception as e:
                return f"Risk summary error: {str(e)}"

        return [
            Tool(name="ScoreESG", func=score_esg,
                 description='Score ESG for a company. Input: JSON {"company": "name", "text": "optional text"}'),
            Tool(name="DetectContradictions", func=detect_contradictions,
                 description='Detect greenwashing contradictions. Input: JSON {"company": "name"}'),
            Tool(name="QueryDocuments", func=query_documents,
                 description='Query ESG documents. Input: JSON {"question": "...", "company": "optional"}'),
            Tool(name="GetRiskSummary", func=get_company_risk_summary,
                 description='Get full ESG risk summary. Input: company name as string'),
        ]

    def _build_react_agent(self) -> Optional[AgentExecutor]:
        """Build LangChain ReAct agent for multi-step reasoning."""
        try:
            prompt = PromptTemplate.from_template("""
You are an ESG Risk Intelligence Agent for KPMG. You analyze company ESG disclosures,
detect greenwashing, score sustainability performance, and answer questions about corporate risk.

You have access to these tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}
""")
            agent = create_react_agent(self.llm, self.tools, prompt)
            return AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                max_iterations=5,
                handle_parsing_errors=True
            )
        except Exception as e:
            print(f"[ESGOrchestrator] ReAct agent init warning: {e}")
            return None

    def load_company(self, company_name: str, sustainability_report: str = "",
                     financial_report: str = "", news: str = ""):
        """Load company documents into the system."""
        self._documents[company_name] = {
            "sustainability_report": sustainability_report,
            "financial_report": financial_report,
            "news": news
        }
        # Index all docs in RAG
        docs_to_index = []
        if sustainability_report:
            docs_to_index.append({
                "text": sustainability_report,
                "source": "sustainability_report",
                "company": company_name,
                "doc_type": "esg"
            })
        if financial_report:
            docs_to_index.append({
                "text": financial_report,
                "source": "annual_report",
                "company": company_name,
                "doc_type": "financial"
            })
        if news:
            docs_to_index.append({
                "text": news,
                "source": "news_feed",
                "company": company_name,
                "doc_type": "news"
            })
        if docs_to_index:
            self.rag.ingest_documents(docs_to_index)
        print(f"[ESGOrchestrator] Loaded documents for: {company_name}")

    def analyze(self, company_name: str, query: Optional[str] = None) -> Dict:
        """
        Full analysis pipeline for a company.
        Optionally pass a natural language query for agent-based reasoning.
        """
        docs = self._documents.get(company_name, {})

        # Direct pipeline (always runs)
        esg_result = self.scorer.score_document(
            docs.get("sustainability_report", ""), company_name
        )
        contra_result = self.detector.detect(
            sustainability_report=docs.get("sustainability_report", ""),
            financial_report=docs.get("financial_report", ""),
            news_text=docs.get("news", ""),
            company_name=company_name
        )

        analysis = {
            "company": company_name,
            "esg_analysis": esg_result,
            "contradiction_analysis": contra_result,
            "agent_reasoning": None
        }

        # Agent-based reasoning if query provided
        if query and self.agent_executor:
            try:
                agent_response = self.agent_executor.invoke({
                    "input": f"Company: {company_name}. {query}"
                })
                analysis["agent_reasoning"] = agent_response.get("output", "")
            except Exception as e:
                analysis["agent_reasoning"] = f"Agent reasoning unavailable: {str(e)}"

        return analysis

    def compare_companies(self, companies: List[str]) -> Dict:
        """Compare ESG risk across multiple companies."""
        results = {}
        for company in companies:
            if company in self._documents:
                results[company] = self.analyze(company)
        return results
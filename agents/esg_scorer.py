"""
ESG Scorer Agent - Uses FinBERT (HuggingFace + PyTorch) to score ESG sentiment
across Environmental, Social, and Governance dimensions.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple
import re


class ESGScorerAgent:
    """
    Agent 1: ESG Scoring
    Uses FinBERT (ProsusAI/finbert) from HuggingFace to analyze financial text
    and score it across ESG dimensions using PyTorch inference.
    """

    ESG_KEYWORDS = {
        "environmental": [
            "carbon", "emission", "climate", "renewable", "energy", "waste",
            "pollution", "biodiversity", "water", "deforestation", "net zero",
            "greenhouse", "sustainability", "fossil", "solar", "wind"
        ],
        "social": [
            "employee", "diversity", "inclusion", "labor", "human rights",
            "community", "health", "safety", "gender", "wage", "supply chain",
            "workers", "discrimination", "welfare", "training"
        ],
        "governance": [
            "board", "audit", "transparency", "compliance", "shareholder",
            "executive", "compensation", "risk management", "corruption",
            "bribery", "whistleblower", "ethics", "accountability", "disclosure"
        ]
    }

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        print(f"[ESGScorerAgent] Loading FinBERT from HuggingFace: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.labels = ["positive", "negative", "neutral"]
        print(f"[ESGScorerAgent] Model loaded on {self.device}")

    def _classify_sentiment(self, text: str) -> Dict[str, float]:
        """Run FinBERT inference on a text chunk."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        return {label: float(prob) for label, prob in zip(self.labels, probs)}

    def _extract_esg_sentences(self, text: str, category: str) -> List[str]:
        """Extract sentences relevant to a specific ESG category."""
        keywords = self.ESG_KEYWORDS[category]
        sentences = re.split(r'(?<=[.!?])\s+', text)
        relevant = []
        for sentence in sentences:
            if any(kw.lower() in sentence.lower() for kw in keywords):
                relevant.append(sentence.strip())
        return relevant[:10]  # Top 10 relevant sentences

    def score_document(self, text: str, company_name: str = "Unknown") -> Dict:
        """
        Score a document across all 3 ESG dimensions.
        Returns scores, sentiment breakdown, and flagged sentences.
        """
        results = {
            "company": company_name,
            "scores": {},
            "sentiment_breakdown": {},
            "flagged_sentences": {},
            "overall_esg_score": 0.0,
            "risk_level": "LOW"
        }

        dimension_scores = []

        for category in ["environmental", "social", "governance"]:
            sentences = self._extract_esg_sentences(text, category)

            if not sentences:
                results["scores"][category] = 50.0  # Neutral if no mentions
                results["sentiment_breakdown"][category] = {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
                results["flagged_sentences"][category] = []
                dimension_scores.append(50.0)
                continue

            sentiments = [self._classify_sentiment(s) for s in sentences]

            avg_sentiment = {
                label: float(np.mean([s[label] for s in sentiments]))
                for label in self.labels
            }

            # Score: higher positive = higher ESG score, higher negative = lower
            score = (avg_sentiment["positive"] * 100) - (avg_sentiment["negative"] * 50)
            score = max(0, min(100, score + 50))  # Normalize to 0-100

            # Flag negative sentences
            flagged = [
                sentences[i] for i, s in enumerate(sentiments)
                if s["negative"] > 0.5
            ]

            results["scores"][category] = round(score, 2)
            results["sentiment_breakdown"][category] = avg_sentiment
            results["flagged_sentences"][category] = flagged
            dimension_scores.append(score)

        # Overall ESG score (weighted: E=30%, S=30%, G=40%)
        weights = [0.30, 0.30, 0.40]
        overall = sum(s * w for s, w in zip(dimension_scores, weights))
        results["overall_esg_score"] = round(overall, 2)

        # Risk classification
        if overall >= 70:
            results["risk_level"] = "LOW"
        elif overall >= 45:
            results["risk_level"] = "MEDIUM"
        else:
            results["risk_level"] = "HIGH"

        return results

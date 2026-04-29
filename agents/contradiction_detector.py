"""
Contradiction Detector Agent - Detects when company ESG claims contradict
their actual financial disclosures, news, or operational data.
This is the novel component that makes the platform stand out.
"""

import re
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class Contradiction:
    claim: str
    evidence: str
    contradiction_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float
    explanation: str


class ContradictionDetectorAgent:
    """
    Agent 2: Contradiction Detection
    Identifies greenwashing and ESG claim inconsistencies by:
    1. Extracting ESG claims from sustainability reports
    2. Cross-referencing with financial disclosures and news
    3. Flagging mismatches with confidence scores
    """

    # Patterns that indicate strong ESG claims
    CLAIM_PATTERNS = {
        "environmental": [
            r"(committed?|pledge[sd]?|target[s]?|aim[s]?)\s+to\s+(reduce|cut|eliminate|achieve|reach)\s+[\w\s]+?(emission|carbon|net.?zero|renewable)",
            r"(carbon.?neutral|net.?zero|climate.?positive|zero.?emission)",
            r"(100%|fully?|completely?)\s+(renewable|sustainable|recycled|clean)",
            r"(reduc\w+)\s+(carbon|emission|footprint)\s+(by\s+\d+%)",
        ],
        "social": [
            r"(zero.?tolerance|committed?|dedicated)\s+to\s+(diversity|inclusion|equality|human.?rights)",
            r"(fair|living)\s+wage[s]?\s+(for\s+all|across|throughout)",
            r"(no|zero)\s+(discrimination|harassment|child.?labor)",
            r"employee\s+(wellbeing|health|safety)\s+(is\s+)?our\s+(priority|commitment)",
        ],
        "governance": [
            r"(full|complete|total)\s+(transparency|disclosure|accountability)",
            r"(zero.?tolerance|strict)\s+(anti.?corruption|bribery|fraud)",
            r"(independent|diverse)\s+board",
            r"(highest|best)\s+(ethical|governance)\s+standard",
        ]
    }

    # Red flag patterns that suggest contradictions
    RED_FLAG_PATTERNS = {
        "environmental": [
            r"(increased?|grew|rising)\s+(emission|carbon|pollution|waste)",
            r"(expanding?|new|additional)\s+(coal|oil|gas|fossil.?fuel)\s+(plant|project|investment|operation)",
            r"(fined?|penaliz\w+|violat\w+)\s+.*?(environmental|pollution|emission)",
            r"(failed?|miss\w+|below)\s+.*?(emission|sustainability|climate)\s+target",
        ],
        "social": [
            r"(lawsuit|litigation|legal.?action)\s+.*?(discrimination|harassment|labor|worker)",
            r"(layoff|retrench|redundanc\w+)\s+.*?(\d+[,\d]*\s+employee|worker|staff)",
            r"(violation|breach)\s+.*?(labor|worker|human.?right)",
            r"(gender.?pay.?gap|wage.?disparity)\s+.*?(\d+%)",
        ],
        "governance": [
            r"(fraud|embezzl\w+|scandal|misconduct)\s+.*?(executive|board|officer|management)",
            r"(SEC|regulator|authority)\s+(investigat\w+|fine[sd]?|action)\s+.*?(compan|firm|corpora)",
            r"(related.?party|conflict.?of.?interest)\s+transaction",
            r"(audit|accounting)\s+(irregularit\w+|concern|qualif\w+|restatement)",
        ]
    }

    def __init__(self):
        print("[ContradictionDetectorAgent] Initialized")

    def _extract_claims(self, text: str, category: str) -> List[str]:
        """Extract ESG claims from sustainability reports."""
        claims = []
        for pattern in self.CLAIM_PATTERNS.get(category, []):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get surrounding context (full sentence)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 100)
                sentence = text[start:end].strip()
                claims.append(sentence)
        return list(set(claims))

    def _extract_red_flags(self, text: str, category: str) -> List[str]:
        """Extract red flags from financial reports or news."""
        red_flags = []
        for pattern in self.RED_FLAG_PATTERNS.get(category, []):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 100)
                sentence = text[start:end].strip()
                red_flags.append(sentence)
        return list(set(red_flags))

    def _calculate_contradiction_severity(
        self, claim: str, red_flag: str, category: str
    ) -> Tuple[str, float, str]:
        """Determine severity and confidence of a contradiction."""

        # Keyword overlap heuristic for confidence
        claim_words = set(re.findall(r'\b\w+\b', claim.lower()))
        flag_words = set(re.findall(r'\b\w+\b', red_flag.lower()))
        
        esg_terms = set(["emission", "carbon", "energy", "worker", "board", 
                         "governance", "social", "environmental", "sustainability"])
        overlap = len(claim_words & flag_words & esg_terms)
        confidence = min(0.95, 0.45 + (overlap * 0.15))

        # Severity based on category and keyword presence
        critical_words = ["fraud", "violation", "fine", "lawsuit", "scandal", 
                         "restatement", "SEC", "increasing emission"]
        high_words = ["increased", "expanding", "failed", "missed", "layoff"]

        combined = (claim + " " + red_flag).lower()
        
        if any(w in combined for w in critical_words):
            severity = "CRITICAL"
        elif any(w in combined for w in high_words):
            severity = "HIGH"
        elif confidence > 0.65:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        explanation = (
            f"Company claims '{claim[:80]}...' but evidence suggests '{red_flag[:80]}...'. "
            f"This {category} contradiction indicates potential greenwashing."
        )
        
        return severity, round(confidence, 2), explanation

    def detect(
        self,
        sustainability_report: str,
        financial_report: str,
        news_text: str = "",
        company_name: str = "Unknown"
    ) -> Dict:
        """
        Core contradiction detection across all ESG dimensions.
        
        Args:
            sustainability_report: Text from ESG/sustainability report
            financial_report: Text from annual report / 10-K
            news_text: Recent news articles about the company
            company_name: Company name for context
        
        Returns:
            Dict with all detected contradictions, risk score, and summary
        """
        combined_evidence = financial_report + " " + news_text
        
        all_contradictions = []
        category_summary = {}

        for category in ["environmental", "social", "governance"]:
            claims = self._extract_claims(sustainability_report, category)
            red_flags = self._extract_red_flags(combined_evidence, category)

            contradictions = []
            for claim in claims:
                for red_flag in red_flags:
                    severity, confidence, explanation = self._calculate_contradiction_severity(
                        claim, red_flag, category
                    )
                    if confidence > 0.4:  # Only include meaningful contradictions
                        contradictions.append(Contradiction(
                            claim=claim,
                            evidence=red_flag,
                            contradiction_type=category,
                            severity=severity,
                            confidence=confidence,
                            explanation=explanation
                        ))

            # Deduplicate and take top contradictions by confidence
            contradictions.sort(key=lambda x: x.confidence, reverse=True)
            top_contradictions = contradictions[:5]
            
            all_contradictions.extend(top_contradictions)
            category_summary[category] = {
                "claims_found": len(claims),
                "red_flags_found": len(red_flags),
                "contradictions": len(top_contradictions),
                "max_severity": max(
                    [c.severity for c in top_contradictions],
                    key=lambda s: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(s),
                    default="NONE"
                ) if top_contradictions else "NONE"
            }

        # Overall greenwashing risk score
        severity_weights = {"CRITICAL": 1.0, "HIGH": 0.75, "MEDIUM": 0.5, "LOW": 0.25, "NONE": 0}
        if all_contradictions:
            risk_score = min(100, sum(
                severity_weights[c.severity] * c.confidence * 25
                for c in all_contradictions
            ))
        else:
            risk_score = 0.0

        return {
            "company": company_name,
            "total_contradictions": len(all_contradictions),
            "greenwashing_risk_score": round(risk_score, 2),
            "greenwashing_risk_level": (
                "CRITICAL" if risk_score > 75 else
                "HIGH" if risk_score > 50 else
                "MEDIUM" if risk_score > 25 else
                "LOW"
            ),
            "category_summary": category_summary,
            "contradictions": [
                {
                    "claim": c.claim,
                    "evidence": c.evidence,
                    "type": c.contradiction_type,
                    "severity": c.severity,
                    "confidence": c.confidence,
                    "explanation": c.explanation
                }
                for c in all_contradictions
            ]
        }

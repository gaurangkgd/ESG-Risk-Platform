"""
Sample ESG reports and financial data for demonstration purposes.
In production, replace with real PDF parsing from actual company filings.
"""

SAMPLE_COMPANIES = {
    "GreenTech Corp": {
        "sustainability_report": """
        GreenTech Corp Sustainability Report 2024
        
        Environmental Commitment:
        We are fully committed to achieving net-zero carbon emissions by 2030. 
        Our renewable energy transition is on track, with a target to use 100% renewable electricity across all operations.
        We pledge to reduce our carbon emissions by 50% by 2025 and eliminate fossil fuel dependency entirely by 2030.
        Our carbon neutral certification demonstrates our commitment to environmental stewardship.
        We have pledged to reduce water usage intensity by 30% and achieve zero waste to landfill by 2026.
        
        Social Responsibility:
        GreenTech Corp is dedicated to zero discrimination in all forms across our global operations.
        We maintain living wages for all employees and contractors throughout our supply chain.
        Employee health and safety is our highest priority with zero tolerance for workplace incidents.
        We are committed to 50% gender diversity at all leadership levels by 2025.
        
        Governance:
        We maintain complete transparency in all financial and ESG disclosures.
        Our independent board ensures the highest ethical standards in corporate governance.
        Zero tolerance for corruption, bribery, or fraud across all operations.
        We adhere to full disclosure requirements under all applicable regulations.
        """,
        
        "financial_report": """
        GreenTech Corp Annual Report 2024 - Risk Factors and Operations
        
        Carbon and Energy:
        Due to expanded manufacturing operations in Southeast Asia, total carbon emissions 
        increased by 23% year-over-year in FY2024. Energy consumption grew significantly
        as we brought new fossil fuel-powered facilities online in Vietnam and Indonesia.
        Our new coal-powered plant in Malaysia began operations in Q3 2024, adding 
        approximately 180,000 tCO2e to our annual footprint.
        
        Labor and Social:
        The company faced a class-action lawsuit from 2,400 workers in our supply chain 
        regarding labor violations and below-minimum wage payments in FY2024.
        Employee turnover increased to 31% in our manufacturing divisions.
        A gender pay gap audit revealed an 18% disparity between male and female employees
        at management levels.
        
        Governance:
        The SEC initiated an investigation into accounting irregularities in our ESG 
        reporting metrics in September 2024.
        Our CFO resigned amid allegations of related-party transactions in Q2 2024.
        """,
        
        "news": """
        Reuters, October 2024: GreenTech Corp faces regulatory scrutiny as emissions data 
        contradicts sustainability pledges. Internal documents reveal carbon emissions 
        increased despite public net-zero commitments.
        
        Bloomberg, November 2024: GreenTech Corp fined $45M for environmental violations
        at its new Malaysian facility. Regulators cited excessive pollution levels far 
        exceeding permitted limits.
        
        FT, December 2024: GreenTech Corp's ESG rating downgraded by MSCI following 
        revelations of labor violations in supply chain and audit concerns over 
        disclosed sustainability metrics.
        """
    },
    
    "CleanFuture Energy": {
        "sustainability_report": """
        CleanFuture Energy ESG Report 2024
        
        Environmental Leadership:
        CleanFuture Energy has achieved a 35% reduction in carbon emissions since 2019 baseline.
        Our wind and solar portfolio now generates 68% of our total electricity production.
        We are on track to meet our 2025 target of 80% renewable generation.
        Total Scope 1 and 2 emissions fell from 4.2M tCO2e in 2020 to 2.7M tCO2e in 2024.
        
        Social Impact:
        Employee safety: 0 fatalities, TRIR of 0.45 (industry average: 1.2) in 2024.
        Gender pay gap closed to 2.1%, down from 11% in 2019.
        92% of employees rate us as a great place to work in annual survey.
        
        Governance:
        Board is 62% independent with 45% gender diversity.
        No regulatory actions, fines, or material litigation in 2024.
        Achieved GRI Standards and TCFD-aligned reporting for third consecutive year.
        """,
        
        "financial_report": """
        CleanFuture Energy Annual Report 2024
        
        Strong operational performance with revenue up 18% driven by renewable energy 
        capacity additions. Capital expenditure of $2.1B with 78% allocated to wind, 
        solar and battery storage projects.
        
        Risk Factors:
        Grid integration challenges for variable renewable energy remain a key risk.
        Commodity exposure to lithium and rare earth metals for battery storage.
        Regulatory risk from potential changes to renewable energy incentives.
        
        No material legal proceedings, regulatory investigations, or ESG-related 
        controversies were reported during the fiscal year.
        """,
        
        "news": """
        Bloomberg, Q4 2024: CleanFuture Energy recognized as top ESG performer in 
        energy sector by Sustainalytics. Analysts cite consistent delivery on 
        emission reduction targets.
        
        Reuters, 2024: CleanFuture Energy closes $800M green bond issuance, 
        oversubscribed 3x, reflecting strong investor confidence in ESG credentials.
        """
    }
}


def get_sample_metrics():
    """Return sample ESG financial metrics for anomaly detection demo."""
    return {
        "GreenTech Corp": [
            280,   # carbon_intensity (HIGH - anomalous)
            23,    # energy_consumption_growth (INCREASING - anomalous)
            4.2,   # water_usage_intensity (HIGH)
            35,    # waste_recycling_rate (LOW)
            31,    # employee_turnover_rate (HIGH - anomalous)
            18,    # gender_pay_gap (HIGH)
            40,    # board_independence_ratio (LOW)
            38,    # esg_disclosure_score (LOW - anomalous)
            8,     # controversies_count (HIGH - anomalous)
            5      # capex_green_ratio (LOW)
        ],
        "CleanFuture Energy": [
            120,   # carbon_intensity (normal)
            -12,   # energy_consumption_growth (declining - good)
            2.1,   # water_usage_intensity (normal)
            82,    # waste_recycling_rate (high - good)
            9,     # employee_turnover_rate (normal)
            2,     # gender_pay_gap (low - good)
            62,    # board_independence_ratio (good)
            88,    # esg_disclosure_score (high - good)
            0,     # controversies_count (none - good)
            78     # capex_green_ratio (high - good)
        ]
    }

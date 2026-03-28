# AutoPen_AI

Proof-of-concept security analytics dashboard for adversarial testing of:

- LLM systems under prompt-based attacks
- Image classifiers under epsilon-constrained perturbation attacks

## What this PoC includes

- Interactive Streamlit dashboard with sidebar filters
- LLM attack analysis:
	- Attack Success Rate (ASR) by model
	- ASR by attack category
	- Threat matrix heatmap (model x attack type)
	- Jailbreak volume and ASR trend over time
- Image attack analysis:
	- Accuracy collapse vs epsilon for FGSM and PGD
	- Fooling rate vs epsilon
	- Per-class vulnerability chart
	- Cross-model transferability matrix
- Executive KPI summary and raw data tables

## Run locally

1. Install dependencies

	 pip install streamlit pandas numpy plotly

2. Start the app

	 streamlit run app.py

3. Open the local URL shown in terminal (typically http://localhost:8501)

## Notes

- This project currently uses realistic synthetic benchmark-style data in helper.py for demonstration.
- You can replace generated data functions with real benchmark ingestion later (for example HarmBench and ART outputs).

# Demo UI : AI-Assisted Underwriting Risk Assessment

A lightweight Gradio web app that demonstrates the full pipeline live:
**select a patient / enter data → XGBoost prediction → SHAP → LLM underwriting report.**

## Files
- `app.py` — the Gradio UI (two tabs: Select Patient / Manual Entry).
- `pipeline.py` — inference logic: loads saved artifacts, runs preprocess → predict →
  SHAP → LangGraph agent (Ollama/Qwen).
- `../Code/rag_utils.py` — imported by `pipeline.py` for the RAG approach.

## Report generation approach (selector in the UI)

| Choice | What runs |
|---|---|
| **Tools only (Step 7)** | Agent + 3 clinical tools. No retrieval. Baseline. |
| **Tools + RAG (Step 7a)** | Same agent and tools, but the prompt is pre-loaded with medical-literature passages retrieved from ChromaDB. Report cites them as `[1]`, `[2]`, … |
| **Both — compare side by side** | Runs each in turn and renders the two reports next to each other. Roughly double the wall time. |

Retrieval is **traditional RAG**: it happens in Python before the LLM is invoked, so
it always runs — the model does not get to skip it. The activity log shows a
`Retrieved N literature passages` row with the source filenames, and each report
lists the exact passages (file, page, cosine similarity) that were supplied.

A status line under the selector reports whether the vector index is usable. If it
is empty or `chromadb`/`pypdf` are missing, the RAG option degrades gracefully:
retrieval is skipped, a warning appears in the activity log, and the report is
still generated tools-only.

## Prerequisites
1. Notebooks 03–06 have been run (produces the artifacts below).
2. `pip install gradio` (already installed in the project venv).
3. **Ollama running** with the model pulled: `ollama pull qwen2.5:7b`.
4. *For the RAG approach only:* `pip install chromadb pypdf`,
   `ollama pull nomic-embed-text`, and build the index once by running
   section 7a.2 of `Code/07a_llm_underwriting_agent_with_RAG.ipynb`.

## Run
From the project root **or** this `demo/` folder:
```
python demo/app.py
```
Then open http://127.0.0.1:7860 (it also opens automatically).

## What it uses (loaded once at startup)
- `data/model_ready/preprocessor.joblib` — fitted preprocessing pipeline
- `data/results/xgboost_model.joblib` — the prediction model
- `data/model_ready/preprocessing_metadata.json` — feature lists + log-transform list
- `data/model_ready/raw_feature_defaults.json` — training medians/modes (form defaults)
- `data/model_ready/X_test_raw.parquet` — real patients for the dropdown
- `config/clinical_lookups.json` — clinical ranges + risk categories for the agent tools
- `Code/vectordb/` — ChromaDB store for the RAG approach (built by notebook 07a)

## Notes
- The manual-entry form exposes a curated set of clinically important fields; all other
  model features are filled from training defaults. Derived features (mean BP, pulse
  pressure, hypertension flag, BMI category) are computed automatically.
- The LLM report runs on every assessment (single inference is fast enough for a demo).
  If Ollama is not running, prediction + SHAP still display and the report area shows a
  friendly error.
- For demonstration only — not a validated clinical or underwriting tool.

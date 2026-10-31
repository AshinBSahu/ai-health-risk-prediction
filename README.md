# AI-Assisted Health Risk Prediction for Life Insurance Underwriting

An end-to-end pipeline that predicts mortality risk from public health survey data,
explains each prediction, and uses a local LLM to write a plain-language underwriting
report.

```
NHANES survey + CDC mortality  ->  features  ->  GLM / XGBoost / Neural Net
        ->  SHAP explanations  ->  LLM agent (+ optional RAG)  ->  underwriting report
```

> For education and demonstration only. This is not a validated clinical or
> underwriting tool.

## Results (held-out test set, n = 3,716)

| Model | AUROC | PR-AUC | F1 |
|---|---|---|---|
| Logistic regression (GLM baseline) | 0.904 | 0.649 | 0.579 |
| XGBoost | 0.907 | 0.663 | 0.613 |
| Neural network (MLP) | 0.903 | 0.652 | 0.560 |

Data: 18,578 NHANES participants (2007-2012), 14.1% deceased by the end of 2019.
Small differences from these numbers are normal if you retrain. LLM-written reports
are never identical between runs.

## Project layout

```
Code/
  01_data_download.ipynb               Download NHANES + CDC mortality files
  02_exploratory_data_analysis.ipynb   Profile the data, class imbalance
  03_feature_engineering.ipynb         Clean, select features, train/test split
  04_glm_baseline.ipynb                Logistic regression baseline
  05_ml_models.ipynb                   XGBoost and PyTorch neural network
  06_shap_explainability.ipynb         SHAP explanations per applicant
  07_llm_underwriting_agent.ipynb      LangGraph + Ollama agent (tools only)
  07a_llm_underwriting_agent_with_RAG.ipynb   Same agent plus literature retrieval
  08_results_evaluation.ipynb          Final comparison tables and figures
  fetch_knowledge_base.py              Downloads the medical literature for RAG
  rag_utils.py                         Chunking, embedding, retrieval helpers
  knowledge_base/                      Documents used for RAG (see its README)
config/clinical_lookups.json           Clinical ranges and risk categories for agent tools
demo/                                  Gradio web app (see demo/README.md)
data/                                  Created by the notebooks (raw, processed, results, ...)
AI_In_Healthcare.pdf                   Project paper
Sahu_A_AI_Underwriting.pptx            Project slides
```

## Prerequisites

- Python 3.13 (developed and tested on 3.13.5; other versions are untested)
- Internet access for Step 1 (downloads about 80 MB from CDC)
- About 6 GB free disk space if you also pull the Ollama models
- [Ollama](https://ollama.com/download) for Steps 7, 7a and the demo. Steps 1 to 6 and
  Step 8 (for the model comparison) do not need it.

## Setup

From the project root:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

If PyTorch tries to download a large CUDA build, see the note at the top of
`requirements.txt` for the CPU-only install.

Register the environment as a Jupyter kernel (optional, helps in VS Code and JupyterLab):

```bash
python -m ipykernel install --user --name health-risk --display-name "health-risk"
```

For the LLM steps, install the two Ollama models:

```bash
ollama pull qwen2.5:7b          # chat model used by the agent
ollama pull nomic-embed-text    # embedding model used for RAG
```

Make sure the Ollama app is running (it listens on `http://localhost:11434`). To use a
smaller chat model, change `MODEL_NAME` near the top of notebooks 07 and 07a and in
`demo/pipeline.py`.

## How to run

Open the notebooks from the `Code/` folder and run them in order. Each step reads the
files written by the one before it. Paths are resolved relative to the project, so
nothing needs editing as long as the folder layout above is kept.

```bash
cd Code
jupyter lab
```

| Step | Notebook | Needs Ollama | Produces |
|---|---|---|---|
| 1 | `01_data_download.ipynb` | No | `data/raw/`, `data/processed/` (merged dataset) |
| 2 | `02_exploratory_data_analysis.ipynb` | No | EDA figures, `data/processed/eda_findings.json` |
| 3 | `03_feature_engineering.ipynb` | No | `data/model_ready/` (train/test sets, preprocessor) |
| 4 | `04_glm_baseline.ipynb` | No | GLM model and metrics in `data/results/` |
| 5 | `05_ml_models.ipynb` | No | XGBoost and neural network models and metrics |
| 6 | `06_shap_explainability.ipynb` | No | `data/results/shap_explanations.json` |
| 7 | `07_llm_underwriting_agent.ipynb` | Yes | `data/results/underwriting_reports.json` |
| 7a | `07a_llm_underwriting_agent_with_RAG.ipynb` | Yes (chat + embeddings) | RAG reports and a comparison with Step 7 |
| 8 | `08_results_evaluation.ipynb` | No | Final tables and figures (needs Step 7 output) |

Step 7a needs the knowledge base and a vector index. See the next section.

### Knowledge base for RAG (Step 7a)

The retrieval step searches medical literature stored in `Code/knowledge_base/`. If that
folder is empty or missing files, rebuild it:

```bash
python Code/fetch_knowledge_base.py
```

This downloads open-access guideline PDFs and Europe PMC papers. A few publishers block
scripted downloads, so those documents have to be saved by hand. The links are listed in
`Code/knowledge_base/README.md`. Then run section 7a.2 of the 7a notebook to build the
ChromaDB index (written to `Code/vectordb/`). Rebuild the index whenever you add or
remove documents.

## Demo app

After Steps 3 to 6 have run (and Ollama is up), start the web UI from the project root:

```bash
python demo/app.py
```

Open http://127.0.0.1:7860. Pick a patient or enter values by hand, and the app shows
the XGBoost risk, the SHAP drivers and an LLM-written underwriting report. More detail is
in `demo/README.md`.

## Data sources

- [NHANES](https://wwwn.cdc.gov/nchs/nhanes/) survey cycles 2007-2008, 2009-2010 and
  2011-2012 (demographics, examination, laboratory, questionnaire files)
- [CDC/NCHS Public-Use Linked Mortality Files](https://www.cdc.gov/nchs/linked-data/mortality-files.htm),
  follow-up through 31 December 2019

Both are public and are downloaded by Step 1, so no data files need to be shared.
Medical literature in `Code/knowledge_base/` is fetched from public sources. Check each
publisher's terms before redistributing those documents.

## What is and is not in this repository

Included: the notebooks and scripts, config, the demo, the trained model files and
result tables and figures in `data/model_ready/`, `data/results/` and `data/reports/`
(so the demo can run without retraining), the paper and the slides.

Not included, because it is re-created by the steps above: `data/raw/` and
`data/processed/` (Step 1), `Code/vectordb/` (Step 7a), and the downloaded literature in
`Code/knowledge_base/` (`fetch_knowledge_base.py`). The saved `.joblib` model files are
Python pickles, so only load files from sources you trust.

## License

The code in this repository is released under the [MIT License](LICENSE). It does not
cover the NHANES or CDC data or the downloaded medical literature, which keep their
original terms.

## Troubleshooting

- **"Could not locate data / config" or file not found:** run the earlier notebooks
  first, and start Jupyter from `Code/` (or the project root).
- **"Could not reach Ollama at localhost:11434":** start the Ollama app and check with
  `ollama list`.
- **Model not found:** run `ollama pull qwen2.5:7b` (and `nomic-embed-text` for RAG).
- **RAG shows no passages:** build the index (section 7a.2) after the knowledge base is
  populated; `chromadb` and `pypdf` must be installed.
- **Slow LLM steps:** the 7B model runs on CPU if you have no supported GPU. Use a
  smaller model or run fewer applicants.

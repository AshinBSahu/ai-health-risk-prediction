# Knowledge Base — Medical Literature for RAG (Step 7a)

Drop clinical / actuarial documents in **this folder**, then re-run the index
cell in `Code/07a_llm_underwriting_agent_with_RAG.ipynb` (section 7a.2).

Supported file types: `.pdf`, `.txt`, `.md` (this README is skipped by the loader).
Sub-folders are scanned too, so you can organise into `cardiovascular/`,
`renal/`, `actuarial/`, etc. if you prefer.

## Automated sourcing

Most of this folder is populated by a script, so the corpus is reproducible:

```
python Code/fetch_knowledge_base.py
```

It fetches two kinds of source:

1. **Direct guideline / government PDFs** at stable URLs — NHLBI ATP III lipid quick
   reference, KDIGO 2024 CKD guideline (full + summary), CDC/NCHS United States Life
   Tables 2021.
2. **Europe PMC open-access full texts** via the official REST API, two articles per
   clinical topic (HbA1c, blood pressure, HDL, creatinine/eGFR, BMI, liver enzymes,
   smoking, heart failure, stroke, cancer survivorship, albuminuria, serum albumin).
   Saved as `.txt` with a citation header (title, journal, authors, DOI, URL).
   Only records Europe PMC flags as open access are downloaded, and duplicates
   (one paper matching several topics) are skipped.

Re-running is idempotent for files already present.

### Publisher-blocked — save these by hand

These return HTTP 403 to scripts. Open in a browser and save into this folder:

- ADA *Standards of Care in Diabetes*, "Diagnosis and Classification" chapter —
  https://diabetesjournals.org/care/issue/47/Supplement_1
- ACC/AHA 2017 Hypertension Guideline —
  https://www.ahajournals.org/doi/10.1161/HYP.0000000000000065
- ACC/AHA 2018 Blood Cholesterol Guideline —
  https://www.ahajournals.org/doi/10.1161/CIR.0000000000000625
- WHO obesity and overweight fact sheet (a web page, so copy the text into a `.txt`) —
  https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight

## Adding more documents later

**Option A — manual:** Just drop files (`.pdf`, `.txt`, `.md`) into this folder.

**Option B — automated:** Add a topic to `EPMC_TOPICS` in
`Code/fetch_knowledge_base.py` and re-run it — that downloads new open-access
articles from Europe PMC.

**Either way, after adding new files:** re-run the "Build / rebuild the vector
index" cell (section 7a.2 of the notebook). The indexer scans this folder and
processes *everything* in it — it doesn't distinguish between manually-placed files
and script-downloaded ones.

The ChromaDB collection is recreated from scratch each time, so the index always
matches exactly what is in this folder — no stale chunks left behind.

## Where to get free, citable source documents

| Topic | Source | Link |
|---|---|---|
| Diabetes / HbA1c thresholds | ADA *Standards of Care in Diabetes* | https://diabetesjournals.org/care |
| Blood pressure / hypertension | ACC/AHA Hypertension Guideline | https://www.ahajournals.org |
| Cholesterol / lipids | ACC/AHA Blood Cholesterol Guideline; NCEP ATP III | https://www.nhlbi.nih.gov/resources/atp-iii-guidelines-glance-quick-desk-reference |
| Kidney function / creatinine | KDIGO CKD Evaluation & Management | https://kdigo.org/guidelines/ckd-evaluation-and-management/ |
| Obesity / BMI | WHO obesity and overweight fact sheet | https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight |
| Liver enzymes (ALT/AST) | ACG Clinical Guideline: Abnormal Liver Chemistries | https://journals.lww.com/ajg |
| Mortality / life tables | CDC/NCHS United States Life Tables | https://www.cdc.gov/nchs/products/life_tables.htm |
| Life-insurance mortality studies | Society of Actuaries research | https://www.soa.org/research/ |
| Peer-reviewed papers (free full text) | PubMed Central | https://www.ncbi.nlm.nih.gov/pmc/ |
| NHANES variable documentation | CDC NHANES | https://wwwn.cdc.gov/nchs/nhanes/ |

### Suggested starter set (~6–8 documents)

Target the features the XGBoost model actually leans on: age, HbA1c,
creatinine, HDL, total cholesterol, BMI, blood pressure, ALT, plus history of
heart failure / stroke / cancer.

## Notes

- Prefer **text-based PDFs**. Scanned/image-only PDFs yield no extractable text
  (the loader warns and skips those pages); they would need OCR first.
- Keep the collection focused. 6–15 authoritative guideline documents retrieve
  better than hundreds of loosely related papers.
- Respect each publisher's terms. Everything listed above is freely downloadable
  for academic use; keep the PDFs local and cite them in the ACM report.
- `00_seed_reference_thresholds.md` is a **placeholder** so the pipeline is
  runnable before you download anything. Replace or supplement it with real
  guideline PDFs before quoting generated reports as evidence.
- Consider git-ignoring `*.pdf` here and `Code/vectordb/` — both get large.

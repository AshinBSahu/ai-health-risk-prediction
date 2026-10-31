"""
Reproducible sourcing of the RAG knowledge base (Step 7a).

Run once (or whenever you want to refresh):
    python Code/fetch_knowledge_base.py

Two kinds of source:

1. DIRECT_PDFS -- authoritative guideline / government PDFs at stable URLs.
   Some publishers (ADA, AHA/Lippincott) block automated requests with HTTP 403;
   those are listed in MANUAL_DOWNLOADS and must be saved by hand from a browser.

2. Europe PMC open-access full texts -- peer-reviewed articles fetched through the
   official REST API and saved as plain text. Only articles Europe PMC flags as
   open access are downloaded, and the source URL + licence are written into the
   header of each file so they can be cited.

Everything lands in Code/knowledge_base/. Re-run section 7a.2 of
Code/07a_llm_underwriting_agent_with_RAG.ipynb afterwards to rebuild the index.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from xml.etree import ElementTree

import requests

KB_DIR = Path(__file__).resolve().parent / 'knowledge_base'
UA = {'User-Agent': 'Mozilla/5.0 (academic-research; MS-AI course project)'}
EPMC = 'https://www.ebi.ac.uk/europepmc/webservices/rest'

# ------------------------------------------------------------------ 1. direct PDFs
DIRECT_PDFS = {
    'nhlbi_atp3_lipids_quickref.pdf':
        'https://www.nhlbi.nih.gov/files/docs/guidelines/atglance.pdf',
    'kdigo_2024_ckd_full_guideline.pdf':
        'https://kdigo.org/wp-content/uploads/2024/07/07232024-KDIGO-CKD.pdf',
    'kdigo_2024_ckd_summary_recs.pdf':
        'https://kdigo.org/wp-content/uploads/2026/05/'
        'KDIGO-2024-CKD-Guideline-Summary-Recommendations-and-Practice-Points.pdf',
    'cdc_us_life_tables_2021.pdf':
        'https://restoredcdc.org/www.cdc.gov/nchs/data/nvsr/nvsr72/nvsr72-12.pdf',
}

# Publisher-blocked (HTTP 403 to scripts). Save these manually into knowledge_base/.
MANUAL_DOWNLOADS = {
    'ADA Standards of Care in Diabetes - "Diagnosis and Classification" chapter':
        'https://diabetesjournals.org/care/issue/47/Supplement_1',
    'ACC/AHA 2017 Hypertension Guideline':
        'https://www.ahajournals.org/doi/10.1161/HYP.0000000000000065',
    'ACC/AHA 2018 Blood Cholesterol Guideline':
        'https://www.ahajournals.org/doi/10.1161/CIR.0000000000000625',
    'WHO obesity and overweight fact sheet (copy text into a .txt file)':
        'https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight',
}

# ------------------------------------------------------------------ 2. Europe PMC topics
# One query per clinical driver the XGBoost model actually leans on.
EPMC_TOPICS = {
    'hba1c_diabetes_mortality': 'HbA1c AND diabetes AND all-cause mortality',
    'blood_pressure_mortality': 'hypertension AND blood pressure AND cardiovascular mortality',
    'hdl_cholesterol_risk': 'HDL cholesterol AND coronary heart disease risk',
    'creatinine_ckd_mortality': 'serum creatinine AND estimated glomerular filtration rate AND mortality',
    'bmi_obesity_mortality': 'body mass index AND all-cause mortality AND cohort',
    'liver_enzymes_mortality': 'alanine aminotransferase AND mortality AND population',
    'smoking_mortality': 'cigarette smoking AND all-cause mortality AND cohort',
    'heart_failure_prognosis': 'congestive heart failure AND prognosis AND survival',
    'stroke_prognosis': 'stroke AND long-term survival AND prognosis',
    'cancer_survivor_mortality': 'cancer survivors AND long-term all-cause mortality',
    'albuminuria_risk': 'albuminuria AND cardiovascular risk AND mortality',
    'serum_albumin_mortality': 'serum albumin AND mortality AND older adults',
}

ARTICLES_PER_TOPIC = 2
MIN_CHARS = 4000          # skip stubs / abstract-only records


def _get(url, **kw):
    return requests.get(url, headers=UA, timeout=120, **kw)


def download_direct_pdfs() -> tuple:
    ok, failed = [], []
    for name, url in DIRECT_PDFS.items():
        dest = KB_DIR / name
        if dest.exists() and dest.stat().st_size > 20_000:
            print(f'  skip (already have) {name}')
            ok.append(name)
            continue
        try:
            r = _get(url)
            r.raise_for_status()
            if not r.content.startswith(b'%PDF'):
                raise ValueError('response is not a PDF')
            dest.write_bytes(r.content)
            print(f'  OK   {name}  {len(r.content) // 1024} KB')
            ok.append(name)
        except Exception as e:
            print(f'  FAIL {name} -> {e}')
            failed.append((name, url, str(e)))
    return ok, failed


def _search_open_access(query: str, n: int) -> list:
    """Return open-access PMC records for a query, newest first."""
    params = {'query': f'({query}) AND OPEN_ACCESS:Y AND HAS_FT:Y',
              'format': 'json', 'pageSize': n * 4, 'sort': 'P_PDATE_D desc'}
    r = _get(f'{EPMC}/search', params=params)
    r.raise_for_status()
    out = []
    for it in r.json().get('resultList', {}).get('result', []):
        if it.get('pmcid') and it.get('isOpenAccess') == 'Y':
            out.append(it)
        if len(out) >= n:
            break
    return out


def _fulltext(pmcid: str) -> str:
    """Fetch JATS XML for an OA article and flatten it to readable text."""
    r = _get(f'{EPMC}/{pmcid}/fullTextXML')
    r.raise_for_status()
    root = ElementTree.fromstring(r.content)

    # Drop references, tables and figures -- noisy and unhelpful for retrieval.
    for tag in ('ref-list', 'table-wrap', 'fig', 'graphic', 'inline-formula',
                'disp-formula', 'back'):
        for parent in root.iter():
            for child in list(parent):
                if child.tag.endswith(tag):
                    parent.remove(child)

    parts = []
    for el in root.iter():
        if el.tag.split('}')[-1] in ('title', 'p'):
            txt = ''.join(el.itertext()).strip()
            if txt:
                parts.append(txt)
    text = '\n\n'.join(parts)
    return re.sub(r'\n{3,}', '\n\n', text)


def download_epmc() -> tuple:
    ok, failed = [], []
    seen = set()   # a paper can match several topics; index it only once
    # Pre-seed from files already on disk so re-runs stay idempotent.
    for p in KB_DIR.glob('epmc_*_PMC*.txt'):
        m = re.search(r'(PMC\d+)', p.name)
        if m:
            seen.add(m.group(1))

    for slug, query in EPMC_TOPICS.items():
        try:
            # over-fetch so duplicates can be skipped without starving a topic
            hits = _search_open_access(query, ARTICLES_PER_TOPIC * 3)
        except Exception as e:
            print(f'  FAIL search "{slug}" -> {e}')
            failed.append((slug, query, str(e)))
            continue
        if not hits:
            print(f'  none  {slug} (no open-access full text found)')
            continue

        taken = 0
        for hit in hits:
            if taken >= ARTICLES_PER_TOPIC:
                break
            pmcid = hit['pmcid']
            if pmcid in seen:
                continue
            seen.add(pmcid)
            i = taken + 1
            name = f'epmc_{slug}_{i}_{pmcid}.txt'
            dest = KB_DIR / name
            if dest.exists() and dest.stat().st_size > MIN_CHARS:
                print(f'  skip (already have) {name}')
                ok.append(name)
                taken += 1
                continue
            try:
                body = _fulltext(pmcid)
                if len(body) < MIN_CHARS:
                    print(f'  thin  {name} ({len(body)} chars) - skipped')
                    continue
                header = (
                    f"# {hit.get('title', '').strip()}\n\n"
                    f"Source: Europe PMC {pmcid} (open access)\n"
                    f"Journal: {hit.get('journalTitle', 'n/a')}"
                    f" {hit.get('pubYear', '')}\n"
                    f"Authors: {hit.get('authorString', 'n/a')}\n"
                    f"DOI: {hit.get('doi', 'n/a')}\n"
                    f"URL: https://europepmc.org/article/PMC/{pmcid}\n"
                    f"Retrieved by Code/fetch_knowledge_base.py for academic use.\n\n"
                    f"---\n\n")
                dest.write_text(header + body, encoding='utf-8')
                print(f'  OK   {name}  {len(body) // 1024} KB')
                ok.append(name)
                taken += 1
                time.sleep(0.4)      # be polite to the API
            except Exception as e:
                print(f'  FAIL {name} -> {e}')
                failed.append((name, pmcid, str(e)))
    return ok, failed


def main():
    KB_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Knowledge base: {KB_DIR}\n')

    print('1) Direct guideline / government PDFs')
    pdf_ok, pdf_failed = download_direct_pdfs()

    print('\n2) Europe PMC open-access full texts')
    epmc_ok, epmc_failed = download_epmc()

    print('\n' + '=' * 68)
    print(f'Downloaded/verified: {len(pdf_ok)} PDF(s), {len(epmc_ok)} article(s)')
    if pdf_failed or epmc_failed:
        print('\nFailures:')
        for name, src, err in pdf_failed + epmc_failed:
            print(f'  - {name}: {err}')
    print('\nPublisher-blocked sources (save manually from a browser):')
    for label, url in MANUAL_DOWNLOADS.items():
        print(f'  - {label}\n      {url}')
    print('\nNext: re-run section 7a.2 of '
          'Code/07a_llm_underwriting_agent_with_RAG.ipynb to rebuild the index.')


if __name__ == '__main__':
    sys.exit(main())

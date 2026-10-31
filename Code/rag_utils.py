"""
Traditional RAG helper for the underwriting pipeline (Step 7a).

Shared by:
  - Code/07a_llm_underwriting_agent_with_RAG.ipynb  (batch report generation)
  - demo/pipeline.py                                (live demo UI)

Design: classic retrieve-then-generate. Retrieval happens in Python BEFORE the
LLM is called and the passages are injected into the prompt as context. The LLM
does not decide whether to retrieve -- that would be "agentic RAG". This keeps
Step 7a a clean A/B against the tools-only Step 7 baseline.

Pipeline:
    Code/knowledge_base/*.pdf|*.txt|*.md
        -> page text (pypdf)
        -> overlapping character chunks
        -> embeddings (Ollama: nomic-embed-text)
        -> ChromaDB persistent collection at Code/vectordb/

Requires:
    pip install chromadb pypdf
    ollama pull nomic-embed-text
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# ------------------------------------------------------------------ config
EMBED_MODEL = 'nomic-embed-text'
COLLECTION_NAME = 'medical_literature'
CHUNK_SIZE = 900          # characters per chunk
CHUNK_OVERLAP = 150       # characters shared between neighbouring chunks
DEFAULT_TOP_K = 4         # passages injected into the prompt

# Everything RAG-related lives next to this file, under Code/.
CODE_DIR = Path(__file__).resolve().parent
KB_DIR = CODE_DIR / 'knowledge_base'
VECTORDB_DIR = CODE_DIR / 'vectordb'


def _find_project_dir(name: str) -> Path:
    """Locate a sibling project folder (config/, data/) from any working dir."""
    for base in [CODE_DIR.parent, Path('.'), Path('..'), Path('..') / '..']:
        c = base / name
        if c.exists():
            return c.resolve()
    return (CODE_DIR.parent / name).resolve()


CONFIG_DIR = _find_project_dir('config')
DATA_DIR = _find_project_dir('data')

SUPPORTED_EXTS = ('.pdf', '.txt', '.md')


# ------------------------------------------------------------------ feature -> search topic
# SHAP gives us NHANES codes. To build a meaningful semantic query we translate
# them into the clinical language the literature actually uses.
_EXTRA_TOPICS = {
    'AGE': 'advancing age and all-cause mortality risk',
    'GENDER': 'sex differences in mortality risk',
    'RIAGENDR': 'sex differences in mortality risk',
    'RACE_ETHNICITY': 'race and ethnicity disparities in mortality',
    'SMQ020': 'cigarette smoking history and mortality',
    'SMQ040': 'current smoking status and mortality',
    'DIQ010': 'diagnosed diabetes mellitus and mortality',
    'MCQ160B': 'congestive heart failure prognosis and mortality',
    'MCQ160C': 'coronary heart disease and mortality',
    'MCQ160D': 'angina pectoris and cardiovascular mortality',
    'MCQ160E': 'myocardial infarction heart attack and mortality',
    'MCQ160F': 'stroke and cerebrovascular disease mortality',
    'MCQ160L': 'chronic liver disease and mortality',
    'MCQ160P': 'chronic obstructive pulmonary disease and mortality',
    'MCQ220': 'cancer history and life expectancy',
    'BMI_CATEGORY': 'body mass index category and mortality',
    'HYPERTENSION_FLAG': 'hypertension and cardiovascular mortality',
    'PULSE_PRESSURE': 'wide pulse pressure arterial stiffness and cardiovascular risk',
    'LBXSATSI': 'alanine aminotransferase ALT liver enzyme elevation',
    'LBXSASSI': 'aspartate aminotransferase AST liver enzyme elevation',
    'LBXSAPSI': 'alkaline phosphatase elevation clinical significance',
    'LBXSBU': 'blood urea nitrogen and renal function',
    'LBXSNASI': 'serum sodium abnormality and outcomes',
    'LBXSKSI': 'serum potassium abnormality and outcomes',
    'LBXCRP': 'C-reactive protein inflammation and mortality',
    'LBXWBCSI': 'white blood cell count and mortality',
    'LBXHGB': 'hemoglobin anemia and mortality',
    'LBXRBCSI': 'red blood cell count and health outcomes',
    'BMXWAIST': 'waist circumference central obesity and mortality',
    'BMXWT': 'body weight and mortality risk',
    'PAQ650': 'physical activity and mortality',
    'ALQ130': 'alcohol consumption and mortality',
    'EDUCATION': 'educational attainment socioeconomic status and mortality',
    'MARITAL_STATUS': 'marital status and mortality risk',
}


def _clinical_names() -> dict:
    """feature_code -> friendly clinical name, from config/clinical_lookups.json."""
    path = CONFIG_DIR / 'clinical_lookups.json'
    if not path.exists():
        return {}
    with open(path) as f:
        lookups = json.load(f)
    return {code: info['name']
            for code, info in lookups.get('clinical_ranges', {}).items()}


_CLINICAL_NAMES = _clinical_names()


def base_code(feature: str) -> str:
    """'MCQ160B_1.0' -> 'MCQ160B';  'RACE_ETHNICITY_3.0' -> 'RACE_ETHNICITY'."""
    return re.sub(r'_-?\d+(\.\d+)?$', '', feature)


# ------------------------------------------------------------------ faithfulness matcher
# Kept here so notebook 07a and notebook 08 score report faithfulness with the
# SAME rule. An exact NHANES-code match is too strict: the LLM writes in plain
# English ('BMXWAIST' -> 'waist circumference'), so each code lists the
# alternative phrasings a report may legitimately use.
FEATURE_SYNONYMS = {
    'AGE': ['age', 'elderly', 'older adult', 'advanced age'],
    'BMXBMI': ['bmi', 'body mass index'],
    'BMXWAIST': ['waist', 'waist circumference', 'abdominal'],
    'BMXHT': ['height'],
    'BMXWT': ['weight', 'body weight'],
    'BP_SYS_MEAN': ['systolic', 'blood pressure'],
    'BP_DIA_MEAN': ['diastolic', 'blood pressure'],
    'LBXTC': ['total cholesterol', 'cholesterol'],
    'LBDHDD': ['hdl', 'hdl cholesterol', 'good cholesterol'],
    'LBXGH': ['hba1c', 'hemoglobin a1c', 'glycated hemoglobin', 'a1c'],
    'LBXSGL': ['glucose', 'blood sugar', 'fasting glucose'],
    'LBXSCR': ['creatinine', 'renal', 'kidney'],
    'LBXSUA': ['uric acid'],
    'LBXSAL': ['albumin', 'serum albumin'],
    'LBXSASSI': ['ast', 'aspartate aminotransferase', 'liver enzyme'],
    'LBXSATSI': ['alt', 'alanine aminotransferase', 'liver enzyme'],
    'LBXSGB': ['globulin'],
    'LBXSTR': ['triglyceride'],
    'LBXWBCSI': ['white blood cell', 'wbc', 'leukocyte'],
    'LBXPLTSI': ['platelet'],
    'URXUMA': ['urine albumin', 'albuminuria', 'albumin'],
    'URXUCR': ['urine creatinine'],
    'URDACT': ['albumin-creatinine ratio', 'albumin creatinine'],
    'POVERTY_INCOME_RATIO': ['poverty', 'income', 'socioeconomic'],
    'GENDER': ['gender', 'male', 'female', 'sex'],
    'RACE_ETHNICITY': ['race', 'ethnicity'],
    'MARITAL_STATUS': ['marital', 'married', 'widowed', 'divorced'],
    'SMQ020': ['smoking', 'smoker', 'cigarette'],
    'MCQ160B': ['congestive heart failure', 'heart failure', 'chf'],
    'MCQ160F': ['stroke'],
    'MCQ220': ['cancer', 'malignancy'],
    'PAQ650': ['physical activity', 'exercise', 'vigorous activity'],
    'CYCLE': ['survey cycle'],
}


def synonym_base_code(feature: str) -> str:
    """Resolve a (possibly one-hot) feature name against FEATURE_SYNONYMS keys.

    Prefix-based rather than regex-based so codes such as 'CYCLE_2011-2012' also
    resolve, matching notebook 08's behaviour.
    """
    for code in FEATURE_SYNONYMS:
        if feature == code or feature.startswith(code + '_'):
            return code
    return base_code(feature)


def mentions_feature(report_text: str, feature: str) -> bool:
    """True if the report refers to `feature` by code, clinical name, or synonym."""
    low = (report_text or '').lower()
    code = synonym_base_code(feature)
    terms = [code.lower()] + FEATURE_SYNONYMS.get(code, [])
    friendly = _CLINICAL_NAMES.get(code)
    if friendly:
        terms.append(friendly.lower())
    return any(t in low for t in terms)


def feature_topic(feature: str) -> str:
    """Turn a (possibly one-hot encoded) feature name into a search phrase."""
    code = base_code(feature)
    if code in _EXTRA_TOPICS:
        return _EXTRA_TOPICS[code]
    if code in _CLINICAL_NAMES:
        return f'{_CLINICAL_NAMES[code]} reference range and mortality risk'
    return f'{code} clinical significance and mortality risk'


# ------------------------------------------------------------------ chunking
_SPLIT_PATTERNS = ['\n\n', '\n', '. ', ' ']


def chunk_text(text: str, size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list:
    """Recursive-style character splitter (no langchain-text-splitters needed).

    Prefers to break on paragraph, then line, then sentence, then word
    boundaries so chunks stay readable.
    """
    text = re.sub(r'[ \t]+', ' ', text or '').strip()
    if not text:
        return []

    chunks, start = [], 0
    while start < len(text):
        end = start + size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        window = text[start:end]
        cut = -1
        for pat in _SPLIT_PATTERNS:
            idx = window.rfind(pat)
            if idx > size * 0.4:          # only accept a break in the back 60%
                cut = idx + len(pat)
                break
        if cut == -1:
            cut = size
        chunks.append(text[start:start + cut].strip())
        start += max(cut - overlap, 1)

    return [c for c in chunks if len(c) > 60]   # drop tiny fragments


# ------------------------------------------------------------------ document loading
def load_documents(kb_dir=None) -> list:
    """Read every supported file in the knowledge-base folder (recursively).

    Returns records: {'text', 'source', 'page'}
    """
    kb = Path(kb_dir) if kb_dir else KB_DIR
    kb.mkdir(parents=True, exist_ok=True)

    records = []
    files = sorted(p for p in kb.rglob('*')
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
                   and p.name.lower() != 'readme.md')

    for path in files:
        if path.suffix.lower() == '.pdf':
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                for i, page in enumerate(reader.pages):
                    txt = page.extract_text() or ''
                    if txt.strip():
                        records.append({'text': txt, 'source': path.name,
                                        'page': i + 1})
            except Exception as e:
                print(f'  [warn] could not read {path.name}: {e}')
        else:
            try:
                txt = path.read_text(encoding='utf-8', errors='ignore')
                if txt.strip():
                    records.append({'text': txt, 'source': path.name, 'page': 0})
            except Exception as e:
                print(f'  [warn] could not read {path.name}: {e}')

    return records


# ------------------------------------------------------------------ embeddings
_EMBEDDER = None


def _embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from langchain_ollama import OllamaEmbeddings
        _EMBEDDER = OllamaEmbeddings(model=EMBED_MODEL)
    return _EMBEDDER


# ------------------------------------------------------------------ chroma client
_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        import chromadb
        VECTORDB_DIR.mkdir(parents=True, exist_ok=True)
        _CLIENT = chromadb.PersistentClient(path=str(VECTORDB_DIR))
    return _CLIENT


def _collection_names(client) -> list:
    """chromadb changed list_collections() return type across versions."""
    return [c if isinstance(c, str) else c.name for c in client.list_collections()]


# ------------------------------------------------------------------ index build
def build_index(rebuild: bool = True, batch_size: int = 32,
                verbose: bool = True) -> dict:
    """Chunk + embed everything in Code/knowledge_base and store in ChromaDB.

    Re-run after dropping new PDFs into the folder. With rebuild=True the
    collection is recreated so the index always mirrors the folder exactly.
    """
    docs = load_documents()
    if not docs:
        raise FileNotFoundError(
            f'No documents found in {KB_DIR}. Add PDFs (see the README in that '
            f'folder) and re-run this cell.')

    chunks, metas = [], []
    for rec in docs:
        for piece in chunk_text(rec['text']):
            chunks.append(piece)
            metas.append({'source': rec['source'], 'page': rec['page']})

    if verbose:
        n_files = len({d['source'] for d in docs})
        print(f'Loaded {len(docs)} page(s) from {n_files} file(s) in {KB_DIR}')
        print(f'Split into {len(chunks)} chunks '
              f'(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})')

    client = _client()
    if rebuild and COLLECTION_NAME in _collection_names(client):
        client.delete_collection(COLLECTION_NAME)
    collection = client.get_or_create_collection(
        COLLECTION_NAME, metadata={'hnsw:space': 'cosine'})

    emb = _embedder()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectors = emb.embed_documents(batch)
        collection.add(
            ids=[f'chunk_{i + j}' for j in range(len(batch))],
            embeddings=vectors,
            documents=batch,
            metadatas=metas[i:i + batch_size],
        )
        if verbose:
            print(f'  embedded {min(i + batch_size, len(chunks))}/{len(chunks)}',
                  end='\r')

    if verbose:
        print(f'\nIndexed {len(chunks)} chunks -> {VECTORDB_DIR}')

    return {'files': sorted({d['source'] for d in docs}),
            'pages': len(docs), 'chunks': len(chunks)}


def index_stats() -> dict:
    """{'available', 'chunks', 'sources'} -- never raises, so the demo UI can
    degrade gracefully when the index has not been built yet."""
    try:
        client = _client()
        if COLLECTION_NAME not in _collection_names(client):
            return {'available': False, 'chunks': 0, 'sources': []}
        col = client.get_collection(COLLECTION_NAME)
        n = col.count()
        if n == 0:
            return {'available': False, 'chunks': 0, 'sources': []}
        # Page through ALL metadata. A capped single get() silently under-reports
        # the source list once the collection exceeds the cap.
        sources, offset, page = set(), 0, 2000
        while offset < n:
            batch = col.get(limit=page, offset=offset, include=['metadatas'])
            metas = batch.get('metadatas') or []
            if not metas:
                break
            sources.update((m or {}).get('source', '?') for m in metas)
            offset += len(metas)
        return {'available': True, 'chunks': n, 'sources': sorted(sources)}
    except Exception as e:
        return {'available': False, 'chunks': 0, 'sources': [], 'error': str(e)}


def is_available() -> bool:
    return index_stats()['available']


# ------------------------------------------------------------------ retrieval
def retrieve(query: str, k: int = DEFAULT_TOP_K) -> list:
    """Embed the query and return the k nearest chunks.

    Each result: {'text', 'source', 'page', 'score'} where score is cosine
    similarity (1.0 = identical).
    """
    client = _client()
    if COLLECTION_NAME not in _collection_names(client):
        return []
    col = client.get_collection(COLLECTION_NAME)
    n = col.count()
    if n == 0:
        return []

    qv = _embedder().embed_query(query)
    res = col.query(query_embeddings=[qv], n_results=min(k, n),
                    include=['documents', 'metadatas', 'distances'])

    out = []
    docs = (res.get('documents') or [[]])[0]
    mets = (res.get('metadatas') or [[]])[0]
    dists = (res.get('distances') or [[]])[0]
    for doc, meta, dist in zip(docs, mets, dists):
        out.append({'text': doc,
                    'source': (meta or {}).get('source', 'unknown'),
                    'page': (meta or {}).get('page', 0),
                    'score': round(1.0 - float(dist), 3)})
    return out


GENERAL_QUERY = ('life insurance underwriting mortality risk assessment of '
                 'laboratory and clinical findings')


def applicant_queries(top_factors: list, n_factors: int = 4) -> list:
    """The semantic queries that will be issued for this applicant.

    Exposed separately so the UI can show *what was searched for*, not just what
    came back. Returns [{'feature', 'query'}], last entry is the general query.
    """
    out = [{'feature': f['feature'], 'query': feature_topic(f['feature'])}
           for f in top_factors[:n_factors]]
    out.append({'feature': '(overall)', 'query': GENERAL_QUERY})
    return out


def retrieve_for_applicant(top_factors: list, k: int = DEFAULT_TOP_K,
                           per_factor: int = 2, n_factors: int = 4) -> list:
    """Multi-query retrieval driven by the applicant's top SHAP drivers.

    One semantic query per important factor plus one overall underwriting query;
    results merged, de-duplicated, ranked by similarity, trimmed to the k best.
    `top_factors` items need a 'feature' key (notebook and demo both provide it).
    """
    queries = [q['query'] for q in applicant_queries(top_factors, n_factors)]

    merged = {}
    for q in queries:
        for hit in retrieve(q, k=per_factor):
            key = hit['text'][:120]
            if key not in merged or hit['score'] > merged[key]['score']:
                merged[key] = hit

    ranked = sorted(merged.values(), key=lambda h: h['score'], reverse=True)
    return ranked[:k]


def format_context(passages: list, max_chars: int = 3200) -> str:
    """Render retrieved passages as a numbered, citable block for the prompt."""
    if not passages:
        return '(no relevant literature retrieved)'
    lines, used = [], 0
    for i, p in enumerate(passages, 1):
        page = f", p.{p['page']}" if p.get('page') else ''
        head = f"[{i}] {p['source']}{page} (similarity {p['score']:.2f})"
        body = re.sub(r'\s+', ' ', p['text']).strip()
        if used + len(body) > max_chars:
            body = body[:max(max_chars - used, 0)].rstrip() + '...'
        if not body:
            break
        lines.append(f'{head}\n{body}')
        used += len(body)
    return '\n\n'.join(lines)


def build_rag_block(top_factors: list, k: int = DEFAULT_TOP_K):
    """Retrieve for an applicant; return (prompt_block, passages)."""
    passages = retrieve_for_applicant(top_factors, k=k)
    block = ('--- Retrieved Medical Literature (cite as [1], [2], ...) ---\n'
             + format_context(passages)
             + '\n--- End of Retrieved Literature ---')
    return block, passages

"""
Inference pipeline for the demo UI.

Loads the artifacts produced by the notebooks and runs the full flow for a
single applicant:
    raw features -> log1p (on skewed cols) -> preprocessor -> XGBoost -> SHAP
    -> LangGraph agent (Ollama/Qwen) -> underwriting report.

Two report-generation approaches are selectable (see APPROACHES):
    'tools'      Step 7  — agent + 3 clinical tools, no retrieval
    'tools+rag'  Step 7a — same agent, prompt pre-loaded with retrieved
                           medical literature (traditional RAG)

All heavy objects are loaded once at import time.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap


# ------------------------------------------------------------------ paths
def _find_dir(name: str) -> Path:
    for base in [Path('.'), Path('..'), Path('..') / '..']:
        c = base / name
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f'Could not locate {name} (run the notebooks first).')


DATA_DIR = _find_dir('data')
CONFIG_DIR = _find_dir('config')
CODE_DIR = _find_dir('Code')
MODEL_DIR = DATA_DIR / 'model_ready'
RESULTS_DIR = DATA_DIR / 'results'

# ------------------------------------------------------------------ RAG (Step 7a)
# rag_utils.py lives in Code/ next to notebook 07a. Import is best-effort so the
# demo still runs (tools-only) if chromadb/pypdf are missing.
sys.path.insert(0, str(CODE_DIR))
try:
    import rag_utils as rag
    RAG_IMPORT_ERROR = None
except Exception as _e:                                  # pragma: no cover
    rag = None
    RAG_IMPORT_ERROR = str(_e)

APPROACHES = {
    'tools': 'Tools only (Step 7)',
    'tools+rag': 'Tools + RAG (Step 7a)',
}


def rag_status() -> dict:
    """{'available', 'chunks', 'sources', 'message'} for the UI banner."""
    if rag is None:
        return {'available': False, 'chunks': 0, 'sources': [],
                'message': f'RAG unavailable — import failed: {RAG_IMPORT_ERROR}. '
                           f'Run: pip install chromadb pypdf'}
    info = rag.index_stats()
    if not info['available']:
        return {**info,
                'message': 'RAG index is empty. Add documents to '
                           'Code/knowledge_base/ and run section 7a.2 of '
                           'Code/07a_llm_underwriting_agent_with_RAG.ipynb.'}
    return {**info,
            'message': f"RAG index ready — {info['chunks']} chunks from "
                       f"{len(info['sources'])} source(s): "
                       f"{', '.join(info['sources'])}"}


# ------------------------------------------------------------------ artifacts
PREPROCESSOR = joblib.load(MODEL_DIR / 'preprocessor.joblib')
XGB_MODEL = joblib.load(RESULTS_DIR / 'xgboost_model.joblib')

with open(MODEL_DIR / 'preprocessing_metadata.json') as f:
    META = json.load(f)
with open(MODEL_DIR / 'raw_feature_defaults.json') as f:
    DEFAULTS = json.load(f)
with open(CONFIG_DIR / 'clinical_lookups.json') as f:
    LOOKUPS = json.load(f)

CONTINUOUS = META['continuous_features']
CATEGORICAL = META['categorical_features']
FEATURE_COLS = CONTINUOUS + CATEGORICAL
LOG_COLS = META['log_transformed']
CLINICAL_RANGES = LOOKUPS['clinical_ranges']
RISK_CATEGORIES = LOOKUPS['risk_categories']

# Sample patients (raw, human-readable) for the dropdown
X_TEST_RAW = pd.read_parquet(MODEL_DIR / 'X_test_raw.parquet')

# Scaler stats to de-standardize SHAP values back to clinical units
_num_pipe = PREPROCESSOR.named_transformers_['num']
_scaler = _num_pipe.named_steps['scale']
SCALER_PARAMS = {feat: (float(_scaler.mean_[i]), float(_scaler.scale_[i]))
                 for i, feat in enumerate(CONTINUOUS)}

# SHAP explainer (built once)
_EXPLAINER = shap.TreeExplainer(XGB_MODEL)


def clean_name(n: str) -> str:
    return n.replace('num__', '').replace('cat__', '')


# ------------------------------------------------------------------ feature assembly
def build_feature_row(user_inputs: dict) -> pd.DataFrame:
    """Start from training defaults, override with user-provided values, and
    compute derived features. Returns a single-row DataFrame with FEATURE_COLS."""
    row = dict(DEFAULTS)  # copy defaults for every model feature
    row.update({k: v for k, v in user_inputs.items() if v is not None})

    # Derived features (mirror notebook 03 logic)
    if 'BP_SYS_MEAN' in row and 'BP_DIA_MEAN' in row:
        row['PULSE_PRESSURE'] = row['BP_SYS_MEAN'] - row['BP_DIA_MEAN']
        row['HYPERTENSION_FLAG'] = float(
            (row['BP_SYS_MEAN'] >= 130) or (row['BP_DIA_MEAN'] >= 80))
    bmi = row.get('BMXBMI')
    if bmi is not None:
        if bmi < 18.5:
            row['BMI_CATEGORY'] = 0.0
        elif bmi < 25:
            row['BMI_CATEGORY'] = 1.0
        elif bmi < 30:
            row['BMI_CATEGORY'] = 2.0
        else:
            row['BMI_CATEGORY'] = 3.0

    return pd.DataFrame([{c: row.get(c, DEFAULTS.get(c)) for c in FEATURE_COLS}])


def predict(feature_row: pd.DataFrame):
    """Apply log1p + preprocessor, then predict with XGBoost. Returns probability."""
    X = feature_row.copy()
    for c in LOG_COLS:
        if c in X.columns:
            X[c] = np.log1p(X[c].astype(float))
    X_proc = PREPROCESSOR.transform(X)
    prob = float(XGB_MODEL.predict_proba(X_proc)[:, 1][0])
    return prob, X_proc


def get_risk_category(prob: float) -> dict:
    for b in RISK_CATEGORIES['bands']:
        if b['min'] <= prob < b['max']:
            return b
    return {'category': 'Unknown', 'action': 'n/a', 'table_rating': 'n/a'}


def top_shap_factors(X_proc, feature_row: pd.DataFrame, k: int = 6):
    """Compute SHAP for this applicant; return top-k contributions with
    de-standardized (clinical-unit) values for display."""
    shap_vals = _EXPLAINER.shap_values(X_proc)[0]
    names = [clean_name(n) for n in PREPROCESSOR.get_feature_names_out()]

    contribs = []
    for name, sv in zip(names, shap_vals):
        contribs.append({'feature': name, 'shap': float(sv)})
    contribs.sort(key=lambda d: abs(d['shap']), reverse=True)

    def _raw_value(feat):
        # exact continuous/categorical match
        if feat in feature_row.columns:
            v = feature_row[feat].iloc[0]
            return v if isinstance(v, str) else float(v)
        # one-hot name like 'MCQ160B_1.0' -> base column 'MCQ160B'
        for base in feature_row.columns:
            if feat.startswith(base + '_'):
                v = feature_row[base].iloc[0]
                return v if isinstance(v, str) else float(v)
        return None

    top = []
    for c in contribs[:k]:
        feat = c['feature']
        top.append({
            'feature': feat,
            'shap': c['shap'],
            'direction': 'increases risk' if c['shap'] > 0 else 'decreases risk',
            'value': _raw_value(feat),
        })
    return top


# ------------------------------------------------------------------ LLM agent
MODEL_NAME = 'qwen2.5:7b'
_AGENT = None  # lazily built on first use


def _build_agent():
    from langchain_ollama import ChatOllama
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    @tool
    def lookup_clinical_ranges(feature_code: str, value: float) -> str:
        """Interpret a clinical value against published reference ranges. Returns
        the matching band (e.g. 'diabetic') and the full ranges."""
        info = CLINICAL_RANGES.get(feature_code)
        if not info:
            return f'No clinical reference range available for {feature_code}.'
        band = 'unknown'
        for b in info['bands']:
            lo = b['min'] if b['min'] is not None else float('-inf')
            hi = b['max'] if b['max'] is not None else float('inf')
            if lo <= value < hi:
                band = b['label']
                break
        return (f"{info['name']} = {value} {info['unit']} -> '{band}'. "
                f"Source: {info['source']}.")

    @tool
    def get_risk_category_tool(predicted_risk: float) -> str:
        """Map predicted mortality probability (0..1) to an underwriting decision."""
        b = get_risk_category(predicted_risk)
        return (f"Predicted risk {predicted_risk:.1%} -> category '{b['category']}'. "
                f"Action: {b['action']}. Rating: {b['table_rating']}.")

    @tool
    def explain_shap_contribution(feature_code: str, shap_value: float, direction: str) -> str:
        """Turn a SHAP contribution into a plain-English risk statement."""
        info = CLINICAL_RANGES.get(feature_code)
        friendly = info['name'] if info else feature_code
        verb = 'increased' if 'increase' in direction else 'decreased'
        strength = ('strongly' if abs(shap_value) > 1.0
                    else 'moderately' if abs(shap_value) > 0.3 else 'slightly')
        return (f"{friendly} {strength} {verb} the applicant's mortality risk "
                f"(SHAP {shap_value:+.2f}).")

    # num_ctx=8192: Ollama's default 4096-token context overflows on the RAG path
    # (retrieved block + system prompt + tool schemas + tool results), after which
    # Ollama context-shifts and effectively stalls. client_kwargs sets an HTTP read
    # timeout so a stall surfaces as an error instead of hanging the UI forever.
    llm = ChatOllama(model=MODEL_NAME, temperature=0.2, num_ctx=8192,
                     client_kwargs={'timeout': 900})
    return create_react_agent(llm, tools=[lookup_clinical_ranges,
                                          get_risk_category_tool,
                                          explain_shap_contribution])


SYSTEM_PROMPT = (
    "You are an AI underwriting assistant for a life insurance company. You receive "
    "a mortality-risk prediction (already produced by a machine-learning model) and "
    "SHAP explanations of the factors that drove it. Write a clear, professional "
    "underwriting summary. Use get_risk_category_tool once for the overall decision; "
    "use explain_shap_contribution for the most important factors; use "
    "lookup_clinical_ranges to interpret specific clinical values. Never invent "
    "numbers. NEVER change the prediction; you only explain it. Structure the report "
    "as: 1) Underwriting Decision, 2) Key Risk Factors, 3) Protective Factors, "
    "4) Summary (2-3 sentences)."
)

# Step 7a: same tools, but authoritative literature is already in context and
# must be cited. Retrieval is done in Python, not chosen by the model.
SYSTEM_PROMPT_RAG = (
    "You are an AI underwriting assistant for a life insurance company. You receive "
    "a mortality-risk prediction (already produced by a machine-learning model), "
    "SHAP explanations of the factors that drove it, and excerpts from authoritative "
    "medical literature retrieved for this applicant. Write a clear, professional "
    "underwriting summary. Use get_risk_category_tool once for the overall decision; "
    "use explain_shap_contribution for the most important factors; use "
    "lookup_clinical_ranges to interpret specific clinical values. Ground your "
    "clinical reasoning in the RETRIEVED LITERATURE and cite it inline by bracket "
    "number, e.g. [1], [2]. Only cite numbers that actually appear in that block; if "
    "the literature does not cover a factor, say so instead of inventing support. "
    "Never invent numbers. NEVER change the prediction; you only explain it. "
    "Structure the report as: 1) Underwriting Decision, 2) Key Risk Factors (with "
    "citations), 3) Protective Factors, 4) Clinical Context from Literature "
    "(2-4 bullets, each cited), 5) Summary (2-3 sentences)."
)

RAG_TOP_K = 4


def _system_prompt(approach: str) -> str:
    return SYSTEM_PROMPT_RAG if approach == 'tools+rag' else SYSTEM_PROMPT


def build_agent_prompt(prob: float, top_factors: list, approach: str = 'tools'):
    """Format the applicant data into the agent prompt (shared by invoke/stream).

    Returns (prompt, passages). `passages` is [] for the tools-only approach and
    for a RAG run where the index is unavailable.
    """
    lines = [f'Predicted mortality risk: {prob:.4f}', '',
             'Top contributing factors (SHAP):']
    for f in top_factors:
        val = f['value']
        val_s = f'{val:.2f}' if isinstance(val, (int, float)) else str(val)
        lines.append(f"  - {f['feature']}: SHAP={f['shap']:+.3f}, "
                     f"value={val_s}, {f['direction']}")
    base = '\n'.join(lines)

    passages = []
    if approach == 'tools+rag' and rag is not None and rag.is_available():
        block, passages = rag.build_rag_block(top_factors, k=RAG_TOP_K)
        prompt = (f'{base}\n\n{block}\n\n'
                  'Write the underwriting summary using your tools. Support your '
                  'clinical reasoning with the retrieved literature and cite it as '
                  '[1], [2], etc.')
    else:
        prompt = f'{base}\n\nWrite the underwriting summary using your tools.'

    return prompt, passages


def generate_report(prob: float, top_factors: list, approach: str = 'tools') -> str:
    """Run the LangGraph agent to produce the underwriting narrative."""
    global _AGENT
    if _AGENT is None:
        _AGENT = _build_agent()

    from langchain_core.messages import HumanMessage, SystemMessage

    prompt, _ = build_agent_prompt(prob, top_factors, approach)
    result = _AGENT.invoke({'messages': [
        SystemMessage(content=_system_prompt(approach)),
        HumanMessage(content=prompt),
    ]})

    # Log how many LLM round-trips / tool calls happened (main time driver)
    msgs = result['messages']
    n_tool_calls = sum(len(getattr(m, 'tool_calls', []) or []) for m in msgs)
    n_ai_turns = sum(1 for m in msgs if type(m).__name__ == 'AIMessage')
    print(f"  [agent:{approach}] messages={len(msgs)}  AI turns={n_ai_turns}  "
          f"tool calls={n_tool_calls}")

    return msgs[-1].content


def stream_report(prob: float, top_factors: list, approach: str = 'tools'):
    """Yield agent events as they happen so the UI can show live progress.

    Yields dicts:
      {'type': 'retrieval_start', 'queries': list, 'chunks': int, ...}
      {'type': 'retrieval', 'passages': list, 'prompt_chars': int}
      {'type': 'retrieval_skipped', 'reason': str}
      {'type': 'ai_turn',  'n': int}
      {'type': 'tool',     'name': str, 'args': dict}
      {'type': 'tool_done','name': str, 'result': str}
      {'type': 'final',    'text': str}
    """
    global _AGENT
    if _AGENT is None:
        _AGENT = _build_agent()

    from langchain_core.messages import HumanMessage, SystemMessage

    # Traditional RAG: retrieval happens HERE, in Python, before the LLM runs.
    if approach == 'tools+rag':
        status = rag_status()
        if not status['available']:
            yield {'type': 'retrieval_skipped', 'reason': status['message']}
        else:
            # Announce the queries BEFORE searching, so the UI can show the step
            # in progress rather than only its result.
            yield {'type': 'retrieval_start',
                   'queries': rag.applicant_queries(top_factors),
                   'chunks': status['chunks'],
                   'sources': len(status['sources']),
                   'embed_model': rag.EMBED_MODEL,
                   'top_k': RAG_TOP_K}

    prompt, passages = build_agent_prompt(prob, top_factors, approach)
    if passages:
        yield {'type': 'retrieval', 'passages': passages,
               'prompt_chars': len(prompt)}

    state = {'messages': [SystemMessage(content=_system_prompt(approach)),
                          HumanMessage(content=prompt)]}

    ai_turns = 0
    final_text = ''
    pending_for = None   # guards against emitting duplicate pending markers
    # stream_mode='updates' emits one event per graph node execution
    for chunk in _AGENT.stream(state, stream_mode='updates'):
        for node_name, node_out in chunk.items():
            for m in node_out.get('messages', []):
                kind = type(m).__name__
                if kind == 'AIMessage':
                    # EVERY AIMessage is an LLM turn (with or without tool calls)
                    ai_turns += 1
                    tcs = getattr(m, 'tool_calls', None) or []
                    yield {'type': 'ai_turn', 'n': ai_turns,
                           'kind': 'tool selection' if tcs else 'writing report'}
                    for tc in tcs:
                        yield {'type': 'tool', 'name': tc.get('name', '?'),
                               'args': tc.get('args', {})}
                    if not tcs and m.content:
                        final_text = m.content
                        yield {'type': 'final', 'text': final_text}
                elif kind == 'ToolMessage':
                    yield {'type': 'tool_done', 'name': getattr(m, 'name', '?'),
                           'result': str(m.content)[:160]}
            # After ALL tool results for this step return, the agent starts another
            # LLM turn. Emit ONE pending marker so the UI shows activity during
            # the long wait (guard against duplicates per tool message).
            if node_name == 'tools' and pending_for != ai_turns + 1:
                pending_for = ai_turns + 1
                yield {'type': 'ai_pending', 'n': pending_for}

    if not final_text:
        yield {'type': 'final', 'text': '_(no final report produced)_'}

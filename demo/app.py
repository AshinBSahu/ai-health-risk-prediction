"""
Demo UI — AI-Assisted Health Risk Assessment for Life Insurance Underwriting.

Run from the project root or the demo/ folder:
    python demo/app.py
Then open http://localhost:7860

Requires: gradio, and Ollama running with qwen2.5:7b for the LLM report.
"""
import gradio as gr
import pandas as pd
import pipeline as pl


# ------------------------------------------------------------------ theme
# UT Austin burnt orange (#BF5700) with a supporting neutral ramp.
UT_ORANGE = '#BF5700'
UT_ORANGE_DARK = '#8C3F00'
UT_ORANGE_LIGHT = '#F2E4D8'

_burnt = gr.themes.Color(
    c50='#FDF5EF', c100='#FAE6D6', c200='#F3C7A6', c300='#EBA372',
    c400='#DC8140', c500=UT_ORANGE, c600='#A64A00', c700=UT_ORANGE_DARK,
    c800='#733400', c900='#5A2900', c950='#3F1D00',
)

UT_THEME = gr.themes.Soft(
    primary_hue=_burnt,
    secondary_hue=_burnt,
    neutral_hue='stone',
).set(
    button_primary_background_fill=UT_ORANGE,
    button_primary_background_fill_hover=UT_ORANGE_DARK,
    button_primary_text_color='#FFFFFF',
    block_title_text_color=UT_ORANGE_DARK,
    block_label_text_color=UT_ORANGE_DARK,
    slider_color=UT_ORANGE,
    checkbox_background_color_selected=UT_ORANGE,
    checkbox_border_color_selected=UT_ORANGE,
)


# ------------------------------------------------------------------ helpers
def _risk_html(prob, cat):
    pct = prob * 100
    color = ('#c0392b' if prob >= 0.40 else '#e67e22' if prob >= 0.15
             else '#27ae60')
    return f"""
    <div style='font-family:system-ui;'>
      <div style='font-size:13px;color:#666;'>Predicted mortality risk</div>
      <div style='font-size:40px;font-weight:700;color:{color};'>{pct:.1f}%</div>
      <div style='background:#eee;border-radius:8px;height:16px;width:100%;margin:6px 0;'>
        <div style='background:{color};height:16px;border-radius:8px;width:{min(pct,100):.1f}%;'></div>
      </div>
      <div style='font-size:16px;'><b>Category:</b> {cat['category']}</div>
      <div style='font-size:13px;color:#444;'>{cat['action']}</div>
    </div>"""


def _factors_html(top):
    rows = ''
    for f in top:
        arrow = '▲' if f['shap'] > 0 else '▼'
        color = '#c0392b' if f['shap'] > 0 else '#27ae60'
        val = f['value']
        val_s = f'{val:.2f}' if isinstance(val, (int, float)) else str(val)
        rows += (f"<tr><td style='padding:4px 8px;color:{color};'>{arrow}</td>"
                 f"<td style='padding:4px 8px;'>{f['feature']}</td>"
                 f"<td style='padding:4px 8px;color:#555;'>{val_s}</td>"
                 f"<td style='padding:4px 8px;color:{color};'>{f['shap']:+.2f}</td></tr>")
    legend = ("<div style='font-family:system-ui;font-size:11px;color:#666;margin-top:6px;'>"
              "<span style='color:#c0392b;'>▲ raises</span> / "
              "<span style='color:#27ae60;'>▼ lowers</span> this applicant's predicted "
              "mortality risk. <b>SHAP</b> = size of that push (log-odds); larger "
              "magnitude means more influence on the prediction.</div>")
    return (f"<table style='font-family:system-ui;border-collapse:collapse;width:100%;'>"
            f"<tr style='border-bottom:1px solid #ddd;color:#666;font-size:12px;'>"
            f"<th></th><th style='text-align:left;'>Factor</th>"
            f"<th style='text-align:left;'>Value</th><th style='text-align:left;'>SHAP</th></tr>"
            f"{rows}</table>{legend}")


import time

_TOOL_LABELS = {
    'lookup_clinical_ranges': 'Looking up clinical reference ranges',
    'get_risk_category_tool': 'Mapping risk to underwriting category',
    'explain_shap_contribution': 'Explaining a SHAP contribution',
}

# --- approach selector (Step 7 vs Step 7a) ---
APPROACH_TOOLS = 'Tools only (Step 7)'
APPROACH_RAG = 'Tools + RAG (Step 7a)'
APPROACH_BOTH = 'Both — compare side by side'

_APPROACH_KEY = {
    APPROACH_TOOLS: 'tools',
    APPROACH_RAG: 'tools+rag',
    APPROACH_BOTH: 'both',
}


_APPROACH_DESC = [
    (APPROACH_TOOLS, [
        'Step 7 baseline. The LangGraph agent reasons over the SHAP output and calls '
        'three clinical tools.',
        'No document retrieval at all — clinical thresholds come only from '
        'config/clinical_lookups.json.',
        'Fastest option. The narrative carries no source citations.',
    ]),
    (APPROACH_RAG, [
        'Step 7a. Identical agent and identical three tools, so retrieval is the only '
        'difference.',
        'Before the LLM runs, Python embeds one query per top SHAP driver and pulls the '
        'best-matching passages from ChromaDB.',
        'Those passages are injected into the prompt and cited as [1], [2] — the model '
        'never gets to skip retrieval.',
    ]),
    (APPROACH_BOTH, [
        'Runs the two approaches one after the other on the same prediction and the same '
        'SHAP values.',
        'Renders both reports side by side so the effect of retrieval is directly '
        'readable.',
        'Roughly double the wall time — best for a demo, not for routine use.',
    ]),
]


def _approach_help_html():
    """Three-line explanation of each approach, always visible."""
    cards = ''
    for title, lines in _APPROACH_DESC:
        body = ''.join(
            f"<div style='color:#44403c;margin:1px 0;'>{i}. {ln}</div>"
            for i, ln in enumerate(lines, 1))
        cards += (
            f"<div style='flex:1;min-width:210px;border:1px solid #E7E0D8;"
            f"border-left:4px solid {UT_ORANGE};border-radius:6px;padding:8px 10px;"
            f"background:#FDFBF9;'>"
            f"<div style='font-weight:700;color:{UT_ORANGE_DARK};margin-bottom:4px;'>"
            f"{title}</div>{body}</div>")
    return (f"<div style='font-family:system-ui;font-size:11.5px;display:flex;"
            f"gap:8px;flex-wrap:wrap;'>{cards}</div>")


def _rag_banner():
    """Compact status line. The full source list is collapsed — 47 filenames
    inline made the panel unreadable."""
    st = pl.rag_status()
    if not st['available']:
        return (f"<div style='font-family:system-ui;font-size:12px;color:#b45309;'>"
                f"! {st['message']}</div>")

    srcs = st['sources']
    pdfs = [s for s in srcs if s.lower().endswith('.pdf')]
    head = (f"<span style='color:#16a34a;font-weight:600;'>✓ RAG index ready</span>"
            f" — <b>{st['chunks']:,}</b> chunks from <b>{len(srcs)}</b> source(s)"
            f" ({len(pdfs)} PDF, {len(srcs) - len(pdfs)} text)"
            f" · embeddings <code>{pl.rag.EMBED_MODEL}</code>"
            if pl.rag else '')
    listing = ''.join(f'<li>{s}</li>' for s in srcs)
    return (f"<div style='font-family:system-ui;font-size:12px;color:#44403c;'>{head}"
            f"<details style='margin-top:4px;'>"
            f"<summary style='cursor:pointer;color:{UT_ORANGE_DARK};'>"
            f"show indexed sources</summary>"
            f"<ul style='margin:4px 0 0 16px;padding:0;font-size:11px;"
            f"max-height:170px;overflow:auto;'>{listing}</ul></details></div>")


def _retrieval_html(state: dict) -> str:
    """Dedicated RAG panel: what was searched, what came back, what got injected.

    `state` keys: status ('idle'|'searching'|'done'|'skipped'), queries, passages,
    chunks, sources, embed_model, top_k, prompt_chars, reason.
    """
    st = state.get('status', 'idle')
    if st == 'idle':
        return ("<div style='font-family:system-ui;font-size:12px;color:#78716c;'>"
                "Retrieval runs only for the <b>Tools + RAG</b> and <b>Both</b> "
                "approaches. Pick one and run an assessment to see the queries, the "
                "matched passages and their similarity scores.</div>")
    if st == 'skipped':
        return (f"<div style='font-family:system-ui;font-size:12px;color:#b45309;'>"
                f"<b>Retrieval skipped.</b> {state.get('reason', '')}<br>"
                f"The report was generated tools-only.</div>")

    hdr = (f"<div style='font-family:system-ui;font-size:12px;color:#44403c;'>"
           f"<b>Step 1 — embed queries.</b> One semantic query per top SHAP driver "
           f"plus one overall underwriting query. Model "
           f"<code>{state.get('embed_model', '?')}</code>, searching "
           f"<b>{state.get('chunks', 0):,}</b> chunks across "
           f"<b>{state.get('sources', 0)}</b> sources, keeping the top "
           f"<b>{state.get('top_k', 0)}</b>.</div>")

    qrows = ''
    for q in state.get('queries', []):
        qrows += (f"<tr>"
                  f"<td style='padding:3px 8px;white-space:nowrap;color:"
                  f"{UT_ORANGE_DARK};font-family:ui-monospace,monospace;'>"
                  f"{q['feature']}</td>"
                  f"<td style='padding:3px 8px;color:#57534e;'>“{q['query']}”</td>"
                  f"</tr>")
    qtable = (f"<table style='font-family:system-ui;font-size:11.5px;"
              f"border-collapse:collapse;width:100%;margin:6px 0;'>"
              f"<tr style='border-bottom:1px solid #E7E0D8;color:#78716c;'>"
              f"<th style='text-align:left;padding:2px 8px;'>SHAP driver</th>"
              f"<th style='text-align:left;padding:2px 8px;'>Search query sent to the "
              f"vector store</th></tr>{qrows}</table>")

    if st == 'searching':
        pending = ("<div style='font-family:system-ui;font-size:12px;color:"
                   f"{UT_ORANGE};'>Searching the vector store…</div>")
        return hdr + qtable + pending

    passages = state.get('passages', [])
    prows = ''
    for i, p in enumerate(passages, 1):
        pct = max(min(p['score'], 1.0), 0.0) * 100
        snippet = (p.get('text') or '')[:260].replace('<', '&lt;')
        page = f", p.{p['page']}" if p.get('page') else ''
        prows += (
            f"<div style='border:1px solid #E7E0D8;border-left:4px solid {UT_ORANGE};"
            f"border-radius:6px;padding:7px 10px;margin:6px 0;background:#FDFBF9;'>"
            f"<div style='display:flex;justify-content:space-between;gap:10px;"
            f"font-size:11.5px;'>"
            f"<span style='font-weight:700;color:{UT_ORANGE_DARK};'>[{i}] "
            f"{p['source']}{page}</span>"
            f"<span style='color:#78716c;white-space:nowrap;'>cosine "
            f"{p['score']:.3f}</span></div>"
            f"<div style='background:#EFE9E3;border-radius:5px;height:5px;margin:4px 0;'>"
            f"<div style='background:{UT_ORANGE};height:5px;border-radius:5px;"
            f"width:{pct:.1f}%;'></div></div>"
            f"<div style='font-size:11px;color:#57534e;'>{snippet}…</div></div>")

    foot = (f"<div style='font-family:system-ui;font-size:12px;color:#44403c;"
            f"margin-top:6px;'><b>Step 3 — inject.</b> These "
            f"{len(passages)} passages were written into the prompt as a numbered, "
            f"citable block (prompt is now "
            f"<b>{state.get('prompt_chars', 0):,}</b> characters). The LLM is "
            f"instructed to cite them as [1], [2], … </div>")

    mid = (f"<div style='font-family:system-ui;font-size:12px;color:#44403c;"
           f"margin-top:8px;'><b>Step 2 — nearest neighbours.</b> Merged across all "
           f"queries, de-duplicated, ranked by cosine similarity.</div>")

    return hdr + qtable + mid + prows + foot


def _fmt_report(approach_key: str, text: str, retrieved: list) -> str:
    """Render one report with an approach header and its retrieved sources."""
    if not text:
        return ''
    label = pl.APPROACHES.get(approach_key, approach_key)
    out = f'#### {label}\n\n{text}'
    if retrieved:
        lines = '\n'.join(
            f"- `[{i}]` {p['source']}"
            + (f", p.{p['page']}" if p.get('page') else '')
            + f" — similarity {p['score']:.2f}"
            for i, p in enumerate(retrieved, 1))
        out += f'\n\n**Retrieved sources supplied to the model**\n\n{lines}'
    return out


def _activity_html(events, elapsed, running=True):
    """Render the activity log. While running, a client-side JS ticker shows a
    live millisecond timer (Python only updates the DOM on each yield)."""
    rows = ''
    for e in events:
        icon, color, label, detail = e
        rows += (f"<div style='display:flex;gap:8px;padding:3px 0;font-size:13px;'>"
                 f"<span style='color:{color};width:16px;'>{icon}</span>"
                 f"<span style='flex:1;'>{label}"
                 f"<span style='color:#888;'>{detail}</span></span></div>")

    if running:
        bar = (f"<div style='background:#EFE9E3;border-radius:6px;height:6px;"
               f"overflow:hidden;margin:6px 0;'><div style='height:6px;width:40%;"
               f"background:{UT_ORANGE};"
               f"animation:hrppulse 1.2s ease-in-out infinite;'></div></div>"
               "<style>@keyframes hrppulse{0%{margin-left:-40%}100%{margin-left:100%}}</style>")
        status = 'Running…'
    else:
        bar = ("<div style='background:#EFE9E3;border-radius:6px;height:6px;margin:6px 0;'>"
               "<div style='height:6px;width:100%;background:#16a34a;border-radius:6px;'></div></div>")
        status = 'Complete'

    header = (f"<div style='font-family:system-ui;'>"
              f"<div style='font-size:12px;color:#666;'>{status}</div>{bar}")
    return header + rows + "</div>"


def _timer_html(elapsed, running=True):
    """Standalone elapsed-time display, driven by gr.Timer so it ticks live."""
    color = UT_ORANGE if running else '#16a34a'
    return (f"<div style='text-align:right;font-family:system-ui;"
            f"font-variant-numeric:tabular-nums;font-size:20px;font-weight:700;"
            f"color:{color};'>{elapsed:.3f}<span style='font-size:13px;'>s</span></div>")


def run_assessment(inputs: dict, approach_label: str = APPROACH_TOOLS):
    """Generator: streams (risk, factors, activity, report_a, report_b, col_b)
    updates to the UI. `approach_label` picks Step 7, Step 7a, or both."""
    t0 = time.perf_counter()
    ev = []

    def el():
        return time.perf_counter() - t0

    key = _APPROACH_KEY.get(approach_label, 'tools')
    slots = ['tools', 'tools+rag'] if key == 'both' else [key]
    show_b = len(slots) == 2
    reports = {s: '' for s in slots}
    retrieved = {s: [] for s in slots}
    rstate = {'status': 'idle'}          # drives the dedicated RAG panel

    def frames(risk='', factors='', running=True):
        a = _fmt_report(slots[0], reports[slots[0]], retrieved[slots[0]])
        b = _fmt_report(slots[1], reports[slots[1]], retrieved[slots[1]]) if show_b else ''
        return (risk, factors, _activity_html(ev, el(), running), a, b,
                gr.update(visible=show_b), _retrieval_html(rstate))

    # --- Stage 1: features ---
    ev.append(('⏳', UT_ORANGE, 'Assembling features', ''))
    yield frames()
    row = pl.build_feature_row(inputs)
    ev[-1] = ('✓', '#16a34a', 'Assembled features',
              f' — {len(pl.FEATURE_COLS)} inputs · {el():.2f}s')

    # --- Stage 2: preprocess + predict ---
    ev.append(('⏳', UT_ORANGE, 'Preprocess + XGBoost prediction', ''))
    yield frames()
    tA = time.perf_counter()
    prob, X_proc = pl.predict(row)
    cat = pl.get_risk_category(prob)
    ev[-1] = ('✓', '#16a34a', 'XGBoost prediction',
              f" — risk {prob:.1%} ({cat['category']}) · {time.perf_counter()-tA:.2f}s")
    risk_html = _risk_html(prob, cat)
    yield frames(risk_html)

    # --- Stage 3: SHAP ---
    ev.append(('⏳', UT_ORANGE, 'Computing SHAP explanation', ''))
    yield frames(risk_html)
    tB = time.perf_counter()
    top = pl.top_shap_factors(X_proc, row, k=6)
    ev[-1] = ('✓', '#16a34a', 'SHAP explanation',
              f' — top driver {top[0]["feature"]} · {time.perf_counter()-tB:.2f}s')
    factors_html = _factors_html(top)
    yield frames(risk_html, factors_html)

    # --- Stage 4: LLM agent, once per selected approach ---
    tC = time.perf_counter()
    for slot in slots:
        label = pl.APPROACHES[slot]
        ev.append(('⏳', UT_ORANGE_DARK, f'{label}: generating report',
                   ' — starting model…'))
        llm_idx = len(ev) - 1
        yield frames(risk_html, factors_html)
        tS = time.perf_counter()
        pending_idx = None   # index of an in-progress "AI turn N" row, if any
        try:
            for evt in pl.stream_report(prob, top, approach=slot):
                if evt['type'] == 'retrieval_start':
                    rstate.update(status='searching', **{k2: v for k2, v in evt.items()
                                                         if k2 != 'type'})
                    ev.append(('🔎', UT_ORANGE, 'RAG: embedding search queries',
                               f" — {len(evt['queries'])} queries over "
                               f"{evt['chunks']:,} chunks · {el():.1f}s"))
                elif evt['type'] == 'retrieval':
                    retrieved[slot] = evt['passages']
                    rstate.update(status='done', passages=evt['passages'],
                                  prompt_chars=evt.get('prompt_chars', 0))
                    srcs = ', '.join(sorted({p['source'] for p in evt['passages']}))
                    best = max((p['score'] for p in evt['passages']), default=0)
                    ev.append(('📚', '#0891b2',
                               f"RAG: injected {len(evt['passages'])} passages "
                               f"into the prompt",
                               f' — best cosine {best:.3f} · {srcs[:80]} · {el():.1f}s'))
                elif evt['type'] == 'retrieval_skipped':
                    rstate.update(status='skipped', reason=evt['reason'])
                    ev.append(('!', '#b45309', 'RAG: retrieval skipped',
                               f" — {evt['reason'][:110]}"))
                elif evt['type'] == 'ai_pending':
                    # The next LLM turn has started but won't return for a while.
                    # Show it immediately so the UI isn't silent during the wait.
                    ev.append(('⏳', UT_ORANGE_DARK,
                               f"AI turn {evt['n']} — writing report",
                               f' · started at {el():.1f}s (this step is the slowest)'))
                    pending_idx = len(ev) - 1
                elif evt['type'] == 'ai_turn':
                    kind = evt.get('kind', 'reasoning')
                    detail = f' · {el():.1f}s'
                    if pending_idx is not None:
                        # resolve the pending row instead of adding a duplicate
                        ev[pending_idx] = ('🧠', UT_ORANGE_DARK,
                                           f"AI turn {evt['n']} — {kind}", detail)
                        pending_idx = None
                    else:
                        ev.append(('🧠', UT_ORANGE_DARK,
                                   f"AI turn {evt['n']} — {kind}", detail))
                elif evt['type'] == 'tool':
                    tl = _TOOL_LABELS.get(evt['name'], evt['name'])
                    ev.append(('🔧', '#b45309', f"Tool: {evt['name']}",
                               f' — {tl} · {el():.1f}s'))
                elif evt['type'] == 'tool_done':
                    ev.append(('↳', '#6b7280', f"{evt['name']} returned",
                               f" — {evt['result'][:70]}…"))
                elif evt['type'] == 'final':
                    reports[slot] = evt['text']
                yield frames(risk_html, factors_html)
            ev[llm_idx] = ('✓', '#16a34a', f'{label}: report complete',
                           f' — {time.perf_counter()-tS:.1f}s')
        except Exception as e:
            ev[llm_idx] = ('✗', '#dc2626', f'{label}: failed', f' — {e}')
            reports[slot] = (f"_Could not generate the LLM report._\n\n"
                             f"Check that Ollama is running and `{pl.MODEL_NAME}` "
                             f"is pulled.\n\nError: {e}")
        yield frames(risk_html, factors_html)

    total = el()
    print(f"[timing] predict+shap={tC-t0:.2f}s  llm={total-(tC-t0):.2f}s  "
          f"total={total:.2f}s  approach={key}")
    yield frames(risk_html, factors_html, running=False)


# ------------------------------------------------------------------ input fields
# A curated, human-friendly subset for manual entry; the rest use training defaults.
def collect_manual(age, gender, bmi, sys_bp, dia_bp, hba1c, tchol, hdl,
                   creatinine, history):
    """history is a list of checked labels from the CheckboxGroup.
    NHANES yes/no coding: 1 = Yes, 2 = No."""
    hist = set(history or [])

    def yn(label):
        return 1.0 if label in hist else 2.0

    return {
        'AGE': age,
        'GENDER': 1.0 if gender == 'Male' else 2.0,
        'BMXBMI': bmi,
        'BP_SYS_MEAN': sys_bp,
        'BP_DIA_MEAN': dia_bp,
        'LBXGH': hba1c,
        'LBXTC': tchol,
        'LBDHDD': hdl,
        'LBXSCR': creatinine,
        'SMQ020': yn('Ever smoked 100+ cigarettes'),
        'DIQ010': yn('Diagnosed diabetes'),
        'MCQ160B': yn('Congestive heart failure'),
        'MCQ220': yn('Cancer history'),
    }


# ------------------------------------------------------------------ patient dropdown
def _patient_choices():
    df = pl.X_TEST_RAW
    choices = []
    # pick a spread: a few high-risk-looking (older) and low-risk (younger) plus random
    sample = df.sort_values('AGE', ascending=False).head(6)
    sample = pd.concat([sample, df.sort_values('AGE').head(4)])
    for idx, r in sample.iterrows():
        seqn = int(r['SEQN']) if 'SEQN' in r else idx
        outcome = 'Deceased' if r.get('MORTSTAT', 0) == 1 else 'Alive'
        label = f"Patient {seqn} — Age {int(r['AGE'])}, BMI {r['BMXBMI']:.0f} ({outcome})"
        choices.append((label, str(idx)))
    return choices


def load_patient(choice_idx):
    if choice_idx is None:
        return {}
    r = pl.X_TEST_RAW.loc[int(choice_idx)]
    # Return the full raw feature row as a dict (all model features)
    return {c: (r[c] if c in r else pl.DEFAULTS.get(c)) for c in pl.FEATURE_COLS}


# ------------------------------------------------------------------ UI
with gr.Blocks(title="AI Underwriting Risk Assessment") as demo:
    gr.HTML(
        f"<div style='font-family:system-ui;border-left:6px solid {UT_ORANGE};"
        f"padding:8px 14px;background:{UT_ORANGE_LIGHT};border-radius:6px;'>"
        f"<div style='font-size:24px;font-weight:800;color:{UT_ORANGE_DARK};'>"
        f"AI-Assisted Health Risk Prediction for Life Insurance Underwriting</div>"
        f"<div style='font-size:13px;color:#57534e;margin-top:2px;'>"
        f"Mortality-risk prediction (XGBoost) &rarr; SHAP explanation &rarr; "
        f"LLM-generated underwriting report (Qwen 2.5 via Ollama), optionally "
        f"grounded with retrieved medical literature (RAG). "
        f"<i>For academic demonstration only.</i></div></div>")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("Select Patient"):
                    patient_dd = gr.Dropdown(
                        choices=_patient_choices(), label="Test-set patient",
                        info="Pick a real (de-identified) applicant from the test set.")
                    btn_patient = gr.Button("Assess Selected Patient", variant="primary")

                with gr.Tab("Manual Entry"):
                    # Compact 2-column layout: numbers on the left, vitals/labs on
                    # the right, history as inline checkboxes. Keeps everything
                    # visible without scrolling.
                    # Sliders (drag-only, no free numeric entry) laid out in
                    # 2 compact columns. `container=False` removes the tall
                    # wrapper so each slider is a single short row.
                    _S = dict(container=False, interactive=True)
                    with gr.Row():
                        with gr.Column(min_width=180):
                            gr.Markdown("**Demographics & Body**")
                            gr.Markdown("<span style='font-size:11px;color:#666;'>Age</span>")
                            age = gr.Slider(18, 90, value=55, step=1, **_S)
                            gender = gr.Radio(["Male", "Female"], value="Male",
                                              label="Gender", container=False)
                            gr.Markdown("<span style='font-size:11px;color:#666;'>BMI (kg/m²)</span>")
                            bmi = gr.Slider(15, 55, value=27, step=0.5, **_S)
                            gr.Markdown("**Vitals**")
                            gr.Markdown("<span style='font-size:11px;color:#666;'>Systolic BP (mmHg)</span>")
                            sys_bp = gr.Slider(80, 220, value=125, step=1, **_S)
                            gr.Markdown("<span style='font-size:11px;color:#666;'>Diastolic BP (mmHg)</span>")
                            dia_bp = gr.Slider(40, 130, value=75, step=1, **_S)
                        with gr.Column(min_width=180):
                            gr.Markdown("**Laboratory Values**")
                            gr.Markdown("<span style='font-size:11px;color:#666;'>HbA1c (%)</span>")
                            hba1c = gr.Slider(3.5, 15, value=5.5, step=0.1, **_S)
                            gr.Markdown("<span style='font-size:11px;color:#666;'>Total Cholesterol (mg/dL)</span>")
                            tchol = gr.Slider(80, 400, value=190, step=5, **_S)
                            gr.Markdown("<span style='font-size:11px;color:#666;'>HDL Cholesterol (mg/dL)</span>")
                            hdl = gr.Slider(15, 120, value=50, step=1, **_S)
                            gr.Markdown("<span style='font-size:11px;color:#666;'>Creatinine (mg/dL)</span>")
                            creat = gr.Slider(0.3, 8, value=0.9, step=0.1, **_S)

                    history = gr.CheckboxGroup(
                        ["Ever smoked 100+ cigarettes", "Diagnosed diabetes",
                         "Congestive heart failure", "Cancer history"],
                        value=[], label="Medical / lifestyle history (check if yes)")
                    btn_manual = gr.Button("Assess Risk", variant="primary")

            with gr.Accordion("Report generation approach", open=True):
                approach = gr.Radio(
                    [APPROACH_TOOLS, APPROACH_RAG, APPROACH_BOTH],
                    value=APPROACH_TOOLS, label="LLM approach", container=False)
                gr.HTML(_approach_help_html())
                rag_banner = gr.HTML(_rag_banner())
                btn_refresh_rag = gr.Button("Re-check RAG index", size="sm")

        with gr.Column(scale=1):
            risk_out = gr.HTML(label="Risk Score")
            gr.Markdown("### Top Contributing Factors")
            factors_out = gr.HTML()

    with gr.Accordion("Pipeline Activity (live)", open=True):
        with gr.Row():
            activity_out = gr.HTML()
            timer_out = gr.HTML(_timer_html(0.0, running=False))
        # Ticks every 50 ms while an assessment is running, so the elapsed-time
        # display counts up live (Gradio strips <script>, so we use gr.Timer).
        tick = gr.Timer(0.05, active=False)
        run_start = gr.State(0.0)

    with gr.Accordion("RAG Retrieval — what was searched and what was injected",
                      open=True):
        retrieval_out = gr.HTML(_retrieval_html({'status': 'idle'}))

    gr.Markdown("### AI-Generated Underwriting Report")
    with gr.Row():
        with gr.Column():
            report_out = gr.Markdown()
        with gr.Column(visible=False) as report_col_b:
            report_b_out = gr.Markdown()

    ASSESS_OUTPUTS = [risk_out, factors_out, activity_out, report_out,
                      report_b_out, report_col_b, retrieval_out]

    # --- callbacks ---
    def _on_manual(age, gender, bmi, sys_bp, dia_bp, hba1c, tchol, hdl, creat,
                   history, approach_label):
        inputs = collect_manual(age, gender, bmi, sys_bp, dia_bp, hba1c, tchol, hdl,
                                creat, history)
        yield from run_assessment(inputs, approach_label)

    def _on_patient(choice_idx, approach_label):
        inputs = load_patient(choice_idx)
        yield from run_assessment(inputs, approach_label)

    btn_refresh_rag.click(lambda: _rag_banner(), outputs=[rag_banner])

    # --- live timer plumbing ---
    def _start_timer():
        # record wall-clock start, activate the ticker
        return time.time(), gr.Timer(active=True), _timer_html(0.0, running=True)

    def _on_tick(start_ts):
        if not start_ts:
            return gr.skip()
        return _timer_html(time.time() - start_ts, running=True)

    def _stop_timer(start_ts):
        elapsed = (time.time() - start_ts) if start_ts else 0.0
        return gr.Timer(active=False), _timer_html(elapsed, running=False)

    tick.tick(_on_tick, inputs=[run_start], outputs=[timer_out])

    # Manual-entry flow: start timer -> run assessment -> stop timer
    btn_manual.click(
        _start_timer, outputs=[run_start, tick, timer_out]
    ).then(
        _on_manual,
        inputs=[age, gender, bmi, sys_bp, dia_bp, hba1c, tchol, hdl, creat,
                history, approach],
        outputs=ASSESS_OUTPUTS,
    ).then(
        _stop_timer, inputs=[run_start], outputs=[tick, timer_out]
    )

    # Select-patient flow
    btn_patient.click(
        _start_timer, outputs=[run_start, tick, timer_out]
    ).then(
        _on_patient, inputs=[patient_dd, approach],
        outputs=ASSESS_OUTPUTS,
    ).then(
        _stop_timer, inputs=[run_start], outputs=[tick, timer_out]
    )


if __name__ == "__main__":
    # Gradio 6 moved `theme` from the Blocks constructor to launch().
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True,
                theme=UT_THEME)

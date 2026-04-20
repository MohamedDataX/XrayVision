"""
=============================================================================
XRAYVISION - style_css.py
=============================================================================
Variable CSS_STYLE à injecter via st.markdown(..., unsafe_allow_html=True)
Design : industriel / sécurité — sobre, contrasté, sans emojis superflus
"""

CSS_STYLE = """
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

/* ── Variables ────────────────────────────────────────────────────────── */
:root {
    --bg:           #0d0f11;
    --bg-card:      #13161a;
    --bg-sidebar:   #0d0f11;
    --border:       #252a30;
    --border-hover: #3a424d;
    --accent:       #e8f0f7;
    --accent-dim:   #7a8d9e;
    --danger:       #d94f4f;
    --danger-bg:    rgba(217, 79, 79, 0.08);
    --safe:         #3ab07a;
    --safe-bg:      rgba(58, 176, 122, 0.08);
    --text:         #c8d4de;
    --text-muted:   #5a6a7a;
    --font-ui:      'IBM Plex Sans', sans-serif;
    --font-mono:    'IBM Plex Mono', monospace;
    --radius:       4px;
    --radius-lg:    8px;
}

/* ── Global reset ─────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-ui) !important;
}

/* ── Titre principal ──────────────────────────────────────────────────── */
h1 {
    font-family: var(--font-mono) !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    color: var(--accent) !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 0.6rem !important;
    margin-bottom: 0.4rem !important;
}

h2, h3 {
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--accent-dim) !important;
    margin-bottom: 0.8rem !important;
}

p, label, [data-testid="stMarkdownContainer"] p {
    font-size: 0.875rem !important;
    color: var(--text) !important;
    line-height: 1.6 !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    font-family: var(--font-ui) !important;
    color: var(--text) !important;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--accent-dim) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    margin-top: 1.2rem !important;
}

/* ── Divider ──────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.4rem 0 !important;
}

/* ── File uploader ────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.4rem !important;
    transition: border-color 0.2s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--border-hover) !important;
}

/* ── Bouton principal ─────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    background: transparent !important;
    border: 1px solid var(--accent-dim) !important;
    border-radius: var(--radius) !important;
    color: var(--accent) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 2rem !important;
    transition: background 0.2s ease, border-color 0.2s ease !important;
}

[data-testid="stButton"] > button:hover {
    background: rgba(200, 212, 222, 0.06) !important;
    border-color: var(--accent) !important;
}

[data-testid="stButton"] > button[kind="primary"] {
    border-color: var(--accent) !important;
}

/* ── Alertes ──────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--radius) !important;
    border-left-width: 3px !important;
    font-size: 0.85rem !important;
}

/* Alerte danger */
[data-testid="stAlert"][data-baseweb="notification"][kind="error"],
div[data-baseweb="notification"].error {
    background: var(--danger-bg) !important;
    border-left-color: var(--danger) !important;
    color: #f0a0a0 !important;
}

/* Alerte succès */
[data-testid="stAlert"][data-baseweb="notification"][kind="success"],
div[data-baseweb="notification"].success {
    background: var(--safe-bg) !important;
    border-left-color: var(--safe) !important;
    color: #8adab8 !important;
}

/* ── Carte de détection individuelle ─────────────────────────────────── */
.detection-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: border-color 0.15s ease;
}

.detection-card:hover {
    border-color: var(--border-hover);
}

.detection-card.dangerous {
    border-left: 3px solid var(--danger);
    background: var(--danger-bg);
}

.detection-label {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.04em;
}

.detection-label.dangerous {
    color: #e87a7a;
}

.detection-confidence {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--accent-dim);
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 0.15rem 0.5rem;
}

.detection-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Légende latérale ─────────────────────────────────────────────────── */
.legend-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.78rem;
    font-family: var(--font-mono);
    color: var(--text);
    padding: 0.15rem 0;
}

.legend-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Badge statut API ─────────────────────────────────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.3rem 0.8rem;
    border-radius: 2px;
    margin-bottom: 0.8rem;
}

.status-badge.online {
    background: var(--safe-bg);
    border: 1px solid rgba(58,176,122,0.3);
    color: #3ab07a;
}

.status-badge.offline {
    background: var(--danger-bg);
    border: 1px solid rgba(217,79,79,0.3);
    color: #d94f4f;
}

.status-indicator {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
}

/* ── Spinner ──────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] {
    color: var(--accent-dim) !important;
}

/* ── Images ───────────────────────────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
}

/* ── Code blocks ──────────────────────────────────────────────────────── */
code, pre {
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Footer ───────────────────────────────────────────────────────────── */
.xray-footer {
    text-align: center;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 2rem;
}

/* ── Scrollbar ────────────────────────────────────────────────────────── */
::-webkit-scrollbar       { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
"""
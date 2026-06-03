"""
Zanardelli Range Suite — tracking allenamento (range, gioco corto, putting).
UI mobile-first, persistenza Google Sheets, export Diario di Gioco (HTML/PDF).
"""

from __future__ import annotations

import base64
import datetime
import html
import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_gsheets_connection import GSheetsConnection
except ImportError:  # pragma: no cover
    from streamlit_gsheets import GSheetsConnection  # type: ignore

try:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# =============================================================================
# Config pagina
# =============================================================================
APP_NAME = "Zanardelli Range Suite"
APP_TAGLINE = "Range Data Suite · Data over talent"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .stDeployButton {display:none;}
    [data-testid="stToolbar"] {visibility: hidden !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Brand palette — arancione, nero, bianco, ocra
BLACK = "#141414"
BLACK_SOFT = "#2A2A2A"
WHITE = "#FFFFFF"
OFF_WHITE = "#FFFCF7"
OCHRE = "#C9A227"
OCHRE_LIGHT = "#E8D48A"
OCHRE_DARK = "#7A5B12"
ORANGE = "#F08A24"
ORANGE_DEEP = "#D96E0A"
ORANGE_SOFT = "#FFE7CF"
ORANGE_LIGHT = "#FFF1E3"
TEXT = "#1A1A1A"
MUTED = "#6B5B4A"
CARD_BG = "#FFFFFF"
CARD_BORDER = "#E8D4B8"
ACCENT_BLUE = "#3D6BB5"
SUCCESS_GREEN = "#17A673"

PASSWORD_DEFAULT = "supernova.analytics"

CATEGORIES = {
    "RANGE": "Gioco lungo / Range",
    "SHORT": "Gioco corto (<50 m)",
    "PUTT": "Putting",
}

DATA_COLUMNS = [
    "User",
    "Date",
    "SessionName",
    "Time",
    "Category",
    "Club",
    "Impact",
    "Curvature",
    "Trajectory",
    "Lie_Start",
    "Lie_End",
    "Direction_LR",
    "Proximity_Lateral_m",
    "Proximity_Depth_m",
    "Start_Dist_m",
    "End_Dist_m",
    "Hole_Dist_Start_m",
    "Hole_Dist_End_m",
    "Lie_Long",
    "Rating",
    "Mental_Reaction",
    "Strokes_Gained",
]

LONG_IMPACT = ["Centro", "Punta", "Tacco", "Shank", "Top", "Flappa"]
LONG_CURVE = ["Dritta", "Fade", "Draw", "Slice", "Hook", "Push", "Pull"]
LONG_DIR = ["Esattamente in linea", "A destra del bersaglio", "A sinistra del bersaglio"]

SHORT_IMPACT = ["Dritta", "Punta", "Tacco", "Shank", "Top", "Flappa"]
SHORT_LIE_START = [
    "Fairway",
    "First cut",
    "Rough",
    "Semi-rough",
    "Bunker",
    "Fringe",
    "Green",
    "Bare lie / Terra dura",
    "Pine straw",
]
SHORT_LIE_END = [
    "Fairway",
    "First cut",
    "Rough",
    "Semi-rough",
    "Bunker",
    "Fringe",
    "Green",
    "Fuori limite area target",
]
SHORT_DIR = ["Esattamente in linea", "A destra della buca", "A sinistra della buca"]

PUTT_IMPACT = ["Centro", "Punta", "Tacco", "Flappa"]
PUTT_TRAJ = ["Dritta", "Pull", "Push"]

MENTAL_OPTIONS = [
    "Molto negativa",
    "Negativa",
    "Neutra",
    "Positiva",
    "Molto positiva",
]

CLUBS_LONG = [
    "DR", "3W", "5W", "7W", "3H", "3i", "4i", "5i", "6i", "7i",
    "8i", "9i", "PW", "AW", "GW", "SW", "LW",
]
CLUBS_SHORT = ["LW", "SW", "GW", "AW", "PW", "9i", "8i", "7i"]

PERIOD_LABELS = [
    "Sessione corrente",
    "Ultimi 7 giorni",
    "Ultimo mese",
    "Ultimi 6 mesi",
    "Ultimo anno",
    "Lifelong",
]

NUMERIC_AVG_FIELDS = [
    ("Proximity_Lateral_m", "Err. laterale (m)"),
    ("Proximity_Depth_m", "Err. profondità (m)"),
    ("Start_Dist_m", "Distanza inizio (m)"),
    ("End_Dist_m", "Distanza fine (m)"),
    ("Hole_Dist_Start_m", "Dist. buca inizio (m)"),
    ("Hole_Dist_End_m", "Dist. buca fine (m)"),
    ("Rating", "Voto medio"),
    ("Strokes_Gained", "SG medio"),
]


# =============================================================================
# Stili UI (look desktop/Java: card nette, ombre, tipografia pulita)
# =============================================================================
def inject_styles() -> None:
    st.markdown(
        f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=IBM+Plex+Sans:wght@500;600;700&display=swap');
    #MainMenu {{visibility: hidden; height: 0;}}
    footer {{visibility: hidden; height: 0;}}
    header [data-testid="stHeader"] {{background: transparent;}}
    html, body, [class*="css"] {{
        font-family: 'Source Sans 3', 'Segoe UI', sans-serif;
        color: {TEXT};
    }}
    .stApp {{
        background:
            radial-gradient(ellipse 90% 40% at 100% 0%, {ORANGE_SOFT} 0%, transparent 55%),
            radial-gradient(ellipse 70% 35% at 0% 0%, {OCHRE_LIGHT}33 0%, transparent 50%),
            linear-gradient(180deg, {OFF_WHITE} 0%, {WHITE} 45%, #FAF6EF 100%);
    }}
    .block-container {{
        padding-top: 0.6rem;
        padding-bottom: 5rem;
        max-width: 920px;
    }}
    h1, h2, h3 {{
        color: {BLACK};
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }}
    [data-testid="stTabs"] button[role="tab"] {{
        font-size: 1rem !important;
        font-weight: 700 !important;
        border-radius: 10px 10px 0 0 !important;
        color: {MUTED} !important;
        border: 1px solid transparent !important;
        padding: 10px 18px !important;
    }}
    [data-testid="stTabs"] button[aria-selected="true"] {{
        color: {BLACK} !important;
        border-color: {CARD_BORDER} !important;
        border-bottom-color: {WHITE} !important;
        background: {WHITE} !important;
        box-shadow: 0 -2px 12px rgba(20,20,20,0.06) !important;
    }}
    div[data-baseweb="select"] > div,
    .stTextInput input,
    .stNumberInput input {{
        border-radius: 10px !important;
        border: 1.5px solid #D4C4B0 !important;
        background: {WHITE} !important;
        min-height: 3.1rem !important;
        font-size: 1.05rem !important;
        box-shadow: inset 0 1px 2px rgba(20,20,20,0.04) !important;
    }}
    .stTextInput input:focus,
    .stNumberInput input:focus {{
        border-color: {ORANGE} !important;
        box-shadow: 0 0 0 3px rgba(240, 138, 36, 0.22) !important;
    }}
    [data-testid="stMetric"] {{
        background: {CARD_BG};
        border: 1.5px solid {CARD_BORDER};
        border-radius: 12px;
        padding: 10px 12px;
        box-shadow: 0 4px 14px rgba(20,20,20,0.07);
    }}
    [data-testid="stMetricLabel"] {{
        color: {MUTED} !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {BLACK} !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 700 !important;
    }}
    .stButton > button,
    .stDownloadButton > button {{
        border-radius: 12px !important;
        font-weight: 700 !important;
        transition: transform 0.12s ease, box-shadow 0.12s ease !important;
    }}
    div[data-testid="stHorizontalBlock"] > div .stButton > button {{
        min-height: 4.1rem !important;
        font-size: 1.08rem !important;
        border-radius: 14px !important;
        border: 1.5px solid #D8D0C4 !important;
        background: linear-gradient(180deg, {WHITE} 0%, #FAF8F4 100%) !important;
        color: {BLACK} !important;
        box-shadow: 0 2px 0 #E8E0D4, 0 6px 16px rgba(20,20,20,0.06) !important;
    }}
    div[data-testid="stHorizontalBlock"] > div .stButton > button:hover {{
        border-color: {ORANGE} !important;
        color: {ORANGE_DEEP} !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 0 #E8E0D4, 0 10px 22px rgba(240,138,36,0.22) !important;
    }}
    .stButton > button[kind="primary"],
    .stDownloadButton > button {{
        min-height: 3.5rem !important;
        border: 0 !important;
        color: {WHITE} !important;
        background: linear-gradient(180deg, {ORANGE} 0%, {ORANGE_DEEP} 100%) !important;
        box-shadow: 0 3px 0 {OCHRE_DARK}, 0 10px 24px rgba(240,138,36,0.35) !important;
    }}
    .stButton > button[kind="primary"]:active {{
        transform: translateY(1px) !important;
        box-shadow: 0 1px 0 {OCHRE_DARK}, 0 6px 14px rgba(240,138,36,0.28) !important;
    }}
    .zrs-preset-row .stButton > button {{
        min-height: 3.35rem !important;
        font-size: 1rem !important;
        background: {ORANGE_LIGHT} !important;
        border-color: #F0C89A !important;
        color: {BLACK_SOFT} !important;
    }}
    .stRadio > div {{
        background: {CARD_BG};
        border: 1.5px solid {CARD_BORDER};
        border-radius: 12px;
        padding: 10px 12px;
        box-shadow: 0 4px 12px rgba(20,20,20,0.05);
    }}
    .zrs-footer {{
        text-align: center;
        color: {MUTED};
        font-size: 0.84rem;
        margin-top: 2.2rem;
        padding: 1rem 0.5rem;
        border-top: 1.5px solid {CARD_BORDER};
        line-height: 1.55;
    }}
    .zrs-footer strong {{
        color: {BLACK};
    }}
    .zrs-logo-caption {{
        font-style: italic;
        color: {MUTED};
        font-weight: 600;
        font-size: 0.92rem;
        margin: 0;
    }}
    .zrs-hero {{
        background: linear-gradient(135deg, {WHITE} 0%, {ORANGE_LIGHT} 100%);
        border: 1.5px solid {CARD_BORDER};
        border-left: 5px solid {ORANGE};
        border-radius: 14px;
        padding: 16px 18px;
        margin: 8px 0 18px 0;
        box-shadow: 0 8px 24px rgba(20,20,20,0.08);
    }}
    .zrs-hero-title {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1.12rem;
        font-weight: 700;
        color: {BLACK};
        margin-bottom: 4px;
    }}
    .zrs-hero-sub {{
        color: {MUTED};
        font-size: 0.92rem;
        margin: 0;
    }}
    .zrs-chip {{
        display: inline-block;
        background: {BLACK};
        color: {WHITE};
        border-radius: 6px;
        padding: 5px 11px;
        margin-right: 6px;
        margin-top: 8px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }}
    .zrs-chip-alt {{
        background: {OCHRE};
        color: {BLACK};
    }}
    .zrs-panel {{
        background: {CARD_BG};
        border: 1.5px solid {CARD_BORDER};
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 14px;
        box-shadow: 0 6px 20px rgba(20,20,20,0.06);
    }}
    .zrs-panel-title {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1rem;
        color: {BLACK};
        font-weight: 700;
        margin-bottom: 2px;
    }}
    .zrs-panel-sub {{
        color: {MUTED};
        font-size: 0.88rem;
        margin: 0;
    }}
    .zrs-step-badge {{
        display: inline-block;
        background: {ORANGE};
        color: {WHITE};
        font-weight: 700;
        font-size: 0.75rem;
        padding: 4px 10px;
        border-radius: 6px;
        margin-bottom: 8px;
    }}
</style>
""",
        unsafe_allow_html=True,
    )


def brand_header(title: str | None = None) -> None:
    c1, c2 = st.columns([1, 3])
    with c1:
        try:
            st.image("logo.png", use_container_width=True)
        except Exception:
            st.markdown(
                f"<div style='font-size:1.5rem;font-weight:800;color:{OCHRE};'>{APP_NAME}</div>",
                unsafe_allow_html=True,
            )
    with c2:
        if title:
            st.markdown(f"### {title}")
        st.markdown(
            f"<p class='zrs-logo-caption'>{APP_TAGLINE}</p>",
            unsafe_allow_html=True,
        )


def brand_footer() -> None:
    st.markdown(
        (
            "<div class='zrs-footer'>"
            f"<strong>© {datetime.date.today().year} Andrea Zanardelli</strong><br>"
            "Co-designed by Andrea Zanardelli and Edoardo Venturoli"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, chips: list[str] | None = None) -> None:
    chips_html = ""
    if chips:
        for i, c in enumerate(chips):
            cls = "zrs-chip zrs-chip-alt" if i % 2 else "zrs-chip"
            chips_html += f"<span class='{cls}'>{c}</span>"
    st.markdown(
        (
            "<div class='zrs-hero'>"
            f"<div class='zrs-hero-title'>{title}</div>"
            f"<p class='zrs-hero-sub'>{subtitle}</p>"
            f"{chips_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_panel(title: str, subtitle: str) -> None:
    st.markdown(
        (
            "<div class='zrs-panel'>"
            f"<div class='zrs-panel-title'>{title}</div>"
            f"<p class='zrs-panel-sub'>{subtitle}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_command_header(page: str) -> None:
    st.markdown(
        (
            "<div class='zrs-panel'>"
            f"<div class='zrs-panel-title'>{APP_NAME} — Command Center</div>"
            f"<p class='zrs-panel-sub'>Sezione attiva: <b>{page}</b> · "
            "UI ottimizzata per inserimento rapido sul campo.</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def step_badge(current: int, total: int, label: str) -> None:
    st.markdown(
        f"<span class='zrs-step-badge'>Step {current}/{total} · {label}</span>",
        unsafe_allow_html=True,
    )


# =============================================================================
# Distanze — fix preset (callback + session_state, lettura solo al salva)
# =============================================================================
def _init_distance(key: str, default: float) -> None:
    if key not in st.session_state:
        st.session_state[key] = float(default)


def _set_distance(key: str, value: float) -> None:
    st.session_state[key] = float(value)


def read_distance(key: str, default: float = 0.0) -> float:
    return float(st.session_state.get(key, default))


def distance_input(
    label: str,
    key: str,
    min_value: float,
    max_value: float,
    step: float,
    presets: list[float] | None = None,
    default: float | None = None,
) -> None:
    """Mostra campo distanza + preset; leggere il valore con read_distance() al conferma/salva."""
    init = float(default if default is not None else min_value)
    _init_distance(key, init)
    st.markdown(f"**{label}**")
    if presets:
        st.caption("Tap rapido — aggiorna subito il valore")
        st.markdown('<div class="zrs-preset-row">', unsafe_allow_html=True)
        cols = st.columns(min(len(presets), 6))
        for i, p in enumerate(presets[:6]):
            cols[i].button(
                f"{p:g} m",
                key=f"{key}_preset_{i}",
                use_container_width=True,
                on_click=_set_distance,
                args=(key, float(p)),
            )
        st.markdown("</div>", unsafe_allow_html=True)
    st.number_input(
        "metri",
        min_value=float(min_value),
        max_value=float(max_value),
        step=float(step),
        key=key,
        label_visibility="collapsed",
    )


def big_choice_grid(
    options: list[str],
    key_prefix: str,
    on_pick: Any,
    cols_n: int = 3,
) -> None:
    cols = st.columns(cols_n)
    for i, opt in enumerate(options):
        if cols[i % cols_n].button(
            opt,
            key=f"{key_prefix}_{i}",
            use_container_width=True,
        ):
            on_pick(opt)


# =============================================================================
# Strokes gained
# =============================================================================
def _interp(x: float, xs: list[float], ys: list[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def expected_putts(distance_m: float) -> float:
    if distance_m <= 0:
        return 0.0
    xs = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
    ys = [1.02, 1.06, 1.10, 1.15, 1.23, 1.30, 1.38, 1.45, 1.58, 1.72, 1.85, 2.05, 2.25, 2.42, 2.55]
    return float(_interp(distance_m, xs, ys))


def expected_short_hole(dist_m: float, lie: str) -> float:
    if dist_m <= 0:
        return 0.0
    lie_adj = {
        "Fairway": 0.0,
        "First cut": 0.10,
        "Semi-rough": 0.14,
        "Rough": 0.20,
        "Bunker": 0.55,
        "Fringe": 0.05,
        "Green": 0.0,
        "Bare lie / Terra dura": 0.12,
        "Pine straw": 0.18,
        "Fuori limite area target": 0.30,
    }
    base = 2.08 + (dist_m / 45.0) * 0.95
    return float(base + lie_adj.get(lie, 0.0))


def expected_long_hole(dist_m: float, from_tee: bool) -> float:
    if dist_m <= 0:
        return 0.0
    if from_tee:
        xs = [120, 160, 200, 240, 280, 320, 380, 440]
        ys = [3.05, 3.25, 3.45, 3.62, 3.78, 3.92, 4.08, 4.22]
        return float(_interp(dist_m, xs, ys))
    xs = [30, 60, 90, 120, 150, 180, 210]
    ys = [2.35, 2.72, 3.02, 3.28, 3.48, 3.65, 3.78]
    return float(_interp(dist_m, xs, ys))


def compute_sg_putt(start_m: float, end_m: float) -> float:
    return float(expected_putts(start_m) - expected_putts(end_m) - 1.0)


def compute_sg_short(start_m: float, end_m: float, lie_s: str, lie_e: str) -> float:
    def exp_at(d: float, lie: str) -> float:
        if lie == "Green":
            return expected_putts(d)
        return expected_short_hole(d, lie)

    return float(exp_at(start_m, lie_s) - exp_at(end_m, lie_e) - 1.0)


def compute_sg_long(start_before_m: float, start_after_m: float, from_tee: bool, lie_after: str) -> float:
    exp_before = expected_long_hole(start_before_m, from_tee)
    use_fairway = lie_after.lower() == "fairway"
    exp_after = (
        expected_long_hole(start_after_m, from_tee=False)
        if use_fairway
        else expected_short_hole(start_after_m, lie_after)
    )
    return float(exp_before - exp_after - 1.0)


# =============================================================================
# Dati
# =============================================================================
@st.cache_data(ttl=10)
def load_data() -> pd.DataFrame:
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl=0)
        if df is None or df.empty:
            return pd.DataFrame(columns=DATA_COLUMNS)
        for c in DATA_COLUMNS:
            if c not in df.columns:
                df[c] = np.nan
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        for num in [
            "Proximity_Lateral_m", "Proximity_Depth_m", "Start_Dist_m", "End_Dist_m",
            "Hole_Dist_Start_m", "Hole_Dist_End_m", "Rating", "Strokes_Gained",
        ]:
            df[num] = pd.to_numeric(df[num], errors="coerce")
        return df[DATA_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=DATA_COLUMNS)


def align_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in DATA_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    return out[DATA_COLUMNS]


def save_shot(row: dict[str, Any]) -> None:
    conn = st.connection("gsheets", type=GSheetsConnection)
    existing = load_data()
    new = pd.DataFrame([row])
    merged = align_dataframe(pd.concat([existing, new], ignore_index=True))
    conn.update(data=merged)
    st.cache_data.clear()


def session_shots(df: pd.DataFrame, user: str, session_name: str) -> pd.DataFrame:
    if df.empty:
        return df
    d = df[(df["User"] == user) & (df["SessionName"] == session_name)].copy()
    return d.sort_values(by=["Date", "Time"], ascending=[True, True])


# =============================================================================
# Diario di Gioco — PDF tabellare per bastone
# =============================================================================
def _mode_or_dash(series: pd.Series) -> str:
    s = series.dropna().astype(str)
    s = s[(s != "") & (s != "nan")]
    if s.empty:
        return "—"
    return str(s.mode().iloc[0])


def club_summary_for_diario(df: pd.DataFrame) -> tuple[list[str], list[list[str]]]:
    """Righe = metriche, colonne = bastoni (medie)."""
    if df.empty:
        return [], []
    clubs = (
        df.groupby("Club", dropna=False)
        .size()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    header = ["Metrica"] + clubs
    rows: list[list[str]] = []

    row_counts = ["N° colpi"]
    for cl in clubs:
        row_counts.append(str(int((df["Club"] == cl).sum())))
    rows.append(row_counts)

    for field, label in NUMERIC_AVG_FIELDS:
        row = [label]
        for cl in clubs:
            sub = df[df["Club"] == cl]
            vals = pd.to_numeric(sub[field], errors="coerce").dropna()
            if vals.empty:
                row.append("—")
            elif field == "Strokes_Gained":
                row.append(f"{vals.mean():+.3f}")
            elif field == "Rating":
                row.append(f"{vals.mean():.2f}")
            else:
                row.append(f"{vals.mean():.2f}")
        rows.append(row)

    for field, label in [
        ("Impact", "Impatto più frequente"),
        ("Curvature", "Curvatura più frequente"),
        ("Direction_LR", "Direzione più frequente"),
        ("Mental_Reaction", "Reazione mentale più frequente"),
    ]:
        row = [label]
        for cl in clubs:
            sub = df[df["Club"] == cl]
            row.append(_mode_or_dash(sub[field]) if field in sub.columns else "—")
        rows.append(row)

    return header, rows


def _diario_meta(df_session: pd.DataFrame, user: str, session_name: str) -> dict[str, str]:
    now = datetime.datetime.now()
    times = df_session["Time"].dropna().astype(str).tolist() if not df_session.empty else []
    t_range = f"{times[0]} – {times[-1]}" if len(times) >= 2 else (times[0] if times else now.strftime("%H:%M"))
    cats = df_session["Category"].dropna().unique().tolist() if not df_session.empty else []
    cat_txt = ", ".join(CATEGORIES.get(c, c) for c in cats) if cats else "—"
    return {
        "atleta": user,
        "sessione": session_name,
        "data": datetime.date.today().strftime("%d/%m/%Y"),
        "ora": t_range,
        "settori": cat_txt,
        "colpi": str(len(df_session)),
        "generato": now.strftime("%d/%m/%Y %H:%M"),
    }


def _logo_data_uri() -> str:
    path = Path("logo.png")
    if not path.is_file():
        return ""
    try:
        raw = path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""


def build_diario_html(
    df_session: pd.DataFrame,
    user: str,
    session_name: str,
) -> str:
    """Diario stampabile: zero dipendenze extra, funziona su qualsiasi deploy Streamlit."""
    meta = _diario_meta(df_session, user, session_name)
    header, rows = club_summary_for_diario(df_session)
    logo_uri = _logo_data_uri()
    logo_block = (
        f'<img src="{logo_uri}" alt="Logo" class="logo" />'
        if logo_uri
        else f'<div class="logo-fallback">{html.escape(APP_NAME)}</div>'
    )

    table_html = ""
    if header:
        thead = "".join(f"<th>{html.escape(c)}</th>" for c in header)
        body_rows = []
        for r in rows:
            cells = "".join(f"<td>{html.escape(str(c))}</td>" for c in r)
            body_rows.append(f"<tr>{cells}</tr>")
        table_html = (
            "<table class='stats'>"
            f"<thead><tr>{thead}</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody>"
            "</table>"
        )
    else:
        table_html = "<p class='muted'>Nessun colpo in questa sessione.</p>"

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8"/>
<title>Diario di Gioco — {html.escape(user)}</title>
<style>
  @page {{ margin: 1.5cm; }}
  body {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    color: {BLACK};
    max-width: 900px;
    margin: 0 auto;
    padding: 24px;
    background: {WHITE};
  }}
  .header {{ display: flex; gap: 20px; align-items: flex-start; margin-bottom: 20px; }}
  .logo {{ max-height: 72px; max-width: 120px; }}
  .logo-fallback {{ font-weight: 800; color: {OCHRE}; font-size: 1.1rem; }}
  h1 {{ margin: 0 0 4px 0; font-size: 1.5rem; }}
  .app {{ color: {MUTED}; margin: 0; font-size: 0.95rem; }}
  .meta {{ width: 100%; border-collapse: collapse; margin: 16px 0 24px; font-size: 0.9rem; }}
  .meta td {{ padding: 6px 10px; border-bottom: 1px solid {CARD_BORDER}; }}
  .meta td:first-child {{ font-weight: 700; width: 180px; color: {BLACK}; }}
  h2 {{ font-size: 1.1rem; border-left: 4px solid {ORANGE}; padding-left: 10px; }}
  table.stats {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    margin-top: 8px;
  }}
  table.stats th {{
    background: {ORANGE};
    color: {WHITE};
    padding: 8px 6px;
    text-align: center;
  }}
  table.stats td {{
    padding: 7px 6px;
    border: 1px solid {CARD_BORDER};
    text-align: center;
  }}
  table.stats td:first-child {{ text-align: left; font-weight: 600; }}
  table.stats tbody tr:nth-child(even) {{ background: {OFF_WHITE}; }}
  .note, .muted {{ color: {MUTED}; font-size: 0.85rem; }}
  .footer {{
    margin-top: 32px;
    padding-top: 12px;
    border-top: 1px solid {CARD_BORDER};
    font-size: 0.8rem;
    color: {MUTED};
    text-align: center;
  }}
  @media print {{
    body {{ padding: 0; }}
    .no-print {{ display: none; }}
  }}
</style>
</head>
<body>
<div class="header">
  {logo_block}
  <div>
    <h1>Diario di Gioco</h1>
    <p class="app">{html.escape(APP_NAME)}</p>
  </div>
</div>
<table class="meta">
  <tr><td>Atleta</td><td>{html.escape(meta['atleta'])}</td></tr>
  <tr><td>Sessione</td><td>{html.escape(meta['sessione'])}</td></tr>
  <tr><td>Data</td><td>{html.escape(meta['data'])}</td></tr>
  <tr><td>Ora / intervallo</td><td>{html.escape(meta['ora'])}</td></tr>
  <tr><td>Settori</td><td>{html.escape(meta['settori'])}</td></tr>
  <tr><td>Colpi totali</td><td>{html.escape(meta['colpi'])}</td></tr>
  <tr><td>Generato il</td><td>{html.escape(meta['generato'])}</td></tr>
</table>
<h2>Statistiche medie per bastone</h2>
<p class="note">Medie sui colpi della sessione corrente — una colonna per ogni bastone utilizzato.</p>
{table_html}
<p class="no-print note">Per ottenere un PDF: apri questo file nel browser e usa Stampa → Salva come PDF.</p>
<div class="footer">© Andrea Zanardelli · Co-designed by Andrea Zanardelli and Edoardo Venturoli</div>
</body>
</html>"""


def build_diario_pdf(
    df_session: pd.DataFrame,
    user: str,
    session_name: str,
) -> bytes | None:
    if not HAS_REPORTLAB:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.4 * cm,
        rightMargin=1.4 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ZTitle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=rl_colors.HexColor(BLACK),
        spaceAfter=6,
    )
    sub_style = ParagraphStyle(
        "ZSub",
        parent=styles["Normal"],
        fontSize=10,
        textColor=rl_colors.HexColor(MUTED),
        spaceAfter=4,
    )
    story: list[Any] = []

    try:
        logo = RLImage("logo.png", width=3.2 * cm, height=3.2 * cm)
        logo.hAlign = "LEFT"
        story.append(logo)
    except Exception:
        pass

    story.append(Paragraph("Diario di Gioco", title_style))
    story.append(Paragraph(f"<b>{APP_NAME}</b>", sub_style))

    meta_dict = _diario_meta(df_session, user, session_name)
    meta = [
        ["Atleta", meta_dict["atleta"]],
        ["Sessione", meta_dict["sessione"]],
        ["Data", meta_dict["data"]],
        ["Ora / intervallo", meta_dict["ora"]],
        ["Settori", meta_dict["settori"]],
        ["Colpi totali sessione", meta_dict["colpi"]],
        ["Generato il", meta_dict["generato"]],
    ]
    meta_table = Table(meta, colWidths=[4.5 * cm, 12 * cm])
    meta_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), rl_colors.HexColor(BLACK)),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 0.5 * cm))

    header, rows = club_summary_for_diario(df_session)
    if not header:
        story.append(Paragraph("Nessun colpo registrato in questa sessione.", sub_style))
    else:
        story.append(Paragraph("Statistiche medie per bastone", styles["Heading2"]))
        story.append(Spacer(1, 0.2 * cm))
        table_data = [header] + rows
        n_clubs = max(len(header) - 1, 1)
        col_w = [4.2 * cm] + [max(2.0 * cm, 16.0 * cm / n_clubs)] * n_clubs
        tbl = Table(table_data, colWidths=col_w[: len(header)])
        tbl.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor(ORANGE)),
                ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("ALIGN", (0, 1), (0, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor(CARD_BORDER)),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor(OFF_WHITE)]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ])
        )
        story.append(tbl)
        story.append(Spacer(1, 0.4 * cm))
        story.append(
            Paragraph(
                "Medie calcolate sui colpi della sessione corrente, una colonna per ogni bastone utilizzato.",
                sub_style,
            )
        )

    story.append(Spacer(1, 0.6 * cm))
    story.append(
        Paragraph(
            "© Andrea Zanardelli · Co-designed by Andrea Zanardelli and Edoardo Venturoli",
            sub_style,
        )
    )
    doc.build(story)
    buf.seek(0)
    return buf.read()


def diario_panel(user: str, session_name: str) -> None:
    render_hero(
        "Diario di Gioco",
        "Scarica il report in PDF con un click. Gli atleti usano solo il browser — il PDF è generato dal server.",
        ["PDF", "Medie per bastone", "Sessione live"],
    )
    df_all = load_data()
    df_sess = session_shots(df_all, user, session_name)

    m1, m2, m3 = st.columns(3)
    m1.metric("Colpi in sessione", len(df_sess))
    clubs_n = df_sess["Club"].nunique() if not df_sess.empty else 0
    m2.metric("Bastoni usati", clubs_n)
    sg = pd.to_numeric(df_sess["Strokes_Gained"], errors="coerce").dropna()
    m3.metric("SG medio sessione", f"{sg.mean():+.3f}" if len(sg) else "—")

    if df_sess.empty:
        st.info("Registra almeno un colpo in questa sessione per generare il Diario di Gioco.")
        brand_footer()
        return

    st.markdown("#### Anteprima dati sessione")
    preview_cols = ["Time", "Category", "Club", "Rating", "Strokes_Gained", "Start_Dist_m", "End_Dist_m"]
    st.dataframe(
        df_sess[[c for c in preview_cols if c in df_sess.columns]],
        use_container_width=True,
        hide_index=True,
    )

    header, rows = club_summary_for_diario(df_sess)
    if header:
        st.markdown("#### Medie per bastone (anteprima)")
        st.dataframe(pd.DataFrame(rows, columns=header), use_container_width=True, hide_index=True)

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_name)[:40]
    file_stub = f"Diario_Gioco_{user}_{safe_name}_{datetime.date.today():%Y%m%d}"

    if not HAS_REPORTLAB:
        st.error(
            "Il modulo **reportlab** non è installato sul server che esegue l'app. "
            "Chi gestisce il deploy deve eseguire una sola volta: "
            "`pip install -r requirements.txt` (oppure rideploy su Streamlit Cloud con il requirements aggiornato)."
        )
        st.download_button(
            label="Scarica Diario (HTML — alternativa)",
            data=build_diario_html(df_sess, user, session_name).encode("utf-8"),
            file_name=f"{file_stub}.html",
            mime="text/html",
            use_container_width=True,
        )
        brand_footer()
        return

    pdf_bytes = build_diario_pdf(df_sess, user, session_name)
    if not pdf_bytes:
        st.error("Errore nella generazione del PDF. Riprova tra qualche secondo.")
        brand_footer()
        return

    st.download_button(
        label="Scarica Diario di Gioco (PDF)",
        data=pdf_bytes,
        file_name=f"{file_stub}.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )
    with st.expander("Alternativa: scarica HTML (stampa come PDF dal browser)"):
        st.download_button(
            label="Scarica versione HTML",
            data=build_diario_html(df_sess, user, session_name).encode("utf-8"),
            file_name=f"{file_stub}.html",
            mime="text/html",
            use_container_width=True,
        )
    brand_footer()


# =============================================================================
# Splash & login (splash mantiene SUPERNOVA)
# =============================================================================
def run_splash_sequence() -> None:
    holder = st.empty()
    slides = [
        (3.0, "logo", ""),
        (3.0, "text", "The first…"),
        (3.0, "text", "the easiest…"),
        (3.0, "text", "the original RANGE DATA SUITE"),
    ]
    for dur, kind, msg in slides:
        with holder.container():
            st.markdown("<br><br>", unsafe_allow_html=True)
            cc1, cc2, cc3 = st.columns([1, 3, 1])
            with cc2:
                if kind == "logo":
                    try:
                        st.image("logo.png", use_container_width=True)
                    except Exception:
                        st.markdown(
                            f"<h1 style='text-align:center;color:{OCHRE};'>SUPERNOVA</h1>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        f"<h2 style='text-align:center;color:{OCHRE_DARK};margin-top:4rem;'>{msg}</h2>",
                        unsafe_allow_html=True,
                    )
        time.sleep(dur)
    holder.empty()


def login_screen() -> None:
    brand_header("Accesso")
    st.caption("Inserisci le credenziali per salvare i tuoi colpi sul foglio collegato.")
    u = st.text_input("Username / ID atleta", key="login_user").strip()
    p = st.text_input("Password", type="password", key="login_pass")
    privacy = st.checkbox(
        "Ho letto e accetto l'informativa privacy e il trattamento dei dati.",
        key="privacy_ok",
    )
    if st.button("Entra nella suite", type="primary", use_container_width=True):
        if not privacy:
            st.error("È necessario accettare la privacy policy.")
            return
        if not u:
            st.error("Inserisci uno username.")
            return
        pwd_ok = p == PASSWORD_DEFAULT
        try:
            env_p = st.secrets.get("APP_PASSWORD")
            if env_p:
                pwd_ok = pwd_ok or (p == str(env_p))
        except Exception:
            pass
        if pwd_ok:
            st.session_state["logged_in"] = True
            st.session_state["user"] = u.upper()
            st.session_state["post_auth_logo_pending"] = True
            st.rerun()
        else:
            st.error("Credenziali non valide.")
    brand_footer()
    st.stop()


def run_post_auth_logo() -> None:
    holder = st.empty()
    with holder.container():
        st.markdown("<br><br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 3, 1])
        with c2:
            try:
                st.image("logo.png", use_container_width=True)
            except Exception:
                st.markdown(
                    f"<h1 style='text-align:center;color:{OCHRE};'>{APP_NAME}</h1>",
                    unsafe_allow_html=True,
                )
    time.sleep(2.0)
    holder.empty()


# =============================================================================
# Helpers UI wizard
# =============================================================================
def reset_wizard() -> None:
    for k in list(st.session_state.keys()):
        if k.startswith("wz_"):
            del st.session_state[k]
    st.session_state["wz_cat"] = None
    st.session_state["wz_step"] = 0
    st.session_state["wz_payload"] = {}


def lat_sign(direction: str, lateral_abs: float) -> float:
    if direction.startswith("A destra"):
        return float(abs(lateral_abs))
    if direction.startswith("A sinistra"):
        return -float(abs(lateral_abs))
    return 0.0


def depth_sign(depth_m: float, label: str) -> float:
    if label == "Corto del bersaglio":
        return -abs(depth_m)
    if label == "Lungo del bersaglio":
        return abs(depth_m)
    return 0.0


def filter_period(df: pd.DataFrame, session_name: str, period: str) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    today = datetime.date.today()
    if period == "Sessione corrente":
        return d[d["SessionName"] == session_name]
    if period == "Ultimi 7 giorni":
        return d[d["Date"] >= today - datetime.timedelta(days=7)]
    if period == "Ultimo mese":
        return d[d["Date"] >= today - datetime.timedelta(days=30)]
    if period == "Ultimi 6 mesi":
        return d[d["Date"] >= today - datetime.timedelta(days=182)]
    if period == "Ultimo anno":
        return d[d["Date"] >= today - datetime.timedelta(days=365)]
    return d


def plot_pie(df: pd.DataFrame, column: str, title: str, legend_help: str) -> None:
    if df.empty or column not in df.columns:
        st.info("Nessun dato per questo grafico.")
        return
    if column == "Rating":
        s = pd.to_numeric(df[column], errors="coerce").dropna().astype(int).astype(str)
    else:
        s = df[column].astype(str)
    s = s.replace("nan", "(vuoto)").replace("", "(vuoto)")
    vc = s.value_counts()
    if vc.empty:
        st.info("Nessuna categoria disponibile.")
        return
    fig = px.pie(
        values=vc.values,
        names=vc.index,
        title=title,
        hole=0.35,
        color_discrete_sequence=[ORANGE, OCHRE, OCHRE_LIGHT, ORANGE_DEEP, BLACK_SOFT, MUTED],
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        legend_title_text="Legenda",
        font=dict(color=TEXT),
        title=dict(font=dict(size=18, color=BLACK)),
        margin=dict(t=48, b=24, l=24, r=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(legend_help)


def plot_dispersion(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        return
    d = df.copy()
    d["x_lateral_m"] = pd.to_numeric(d["Proximity_Lateral_m"], errors="coerce")
    d["y_depth_m"] = pd.to_numeric(d["Proximity_Depth_m"], errors="coerce")
    d = d.dropna(subset=["x_lateral_m", "y_depth_m"])
    if d.empty:
        st.info("Aggiungi errore laterale e profondità per vedere la dispersione dall'alto.")
        return
    fig = px.scatter(
        d,
        x="x_lateral_m",
        y="y_depth_m",
        color="Club",
        hover_data=["Impact", "Rating", "Date"],
        title=title,
        labels={
            "x_lateral_m": "Errore laterale (m): sinistra ← 0 → destra",
            "y_depth_m": "Errore in profondità (m): indietro ← 0 → avanti",
        },
        color_discrete_sequence=px.colors.sequential.Oranges_r,
    )
    fig.add_vline(x=0, line_dash="dash", line_color=OCHRE)
    fig.add_hline(y=0, line_dash="dash", line_color=OCHRE)
    fig.update_layout(legend_title_text="Legenda", font=dict(color=TEXT))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Ogni punto è un colpo visto dall'alto: incrocio delle linee = bersaglio."
    )


def putting_make_table(df_putt: pd.DataFrame) -> None:
    if df_putt.empty:
        st.info("Nessun putt nel periodo.")
        return
    d = df_putt.copy()
    d["sd"] = pd.to_numeric(d["Start_Dist_m"], errors="coerce")
    d["ed"] = pd.to_numeric(d["End_Dist_m"], errors="coerce")
    d = d.dropna(subset=["sd"])
    rows = []
    for hi in range(15, 0, -2):
        lo = max(hi - 2, 0)
        sub = d[(d["sd"] > lo) & (d["sd"] <= hi)]
        n = len(sub)
        made = int((sub["ed"].fillna(999) <= 0).sum())
        pct = (made / n * 100.0) if n else 0.0
        rows.append({"Fascia di partenza": f"{lo}–{hi} m", "Putt": n, "Realizzati": made, "% Made": pct})
    out = pd.DataFrame(rows)
    st.markdown("#### Tabella realizzazione putt per distanza di partenza")
    st.dataframe(out.style.format({"% Made": "{:.1f}%"}), use_container_width=True, hide_index=True)


def sg_summary_table(df: pd.DataFrame, cat_key: str) -> None:
    sub = df[df["Category"] == cat_key]
    if sub.empty:
        st.info("Nessuno strokes gained: dati assenti per questo settore.")
        return
    sg = pd.to_numeric(sub["Strokes_Gained"], errors="coerce").dropna()
    if sg.empty:
        st.info("Colonna strokes gained vuota per questo periodo.")
        return
    st.markdown("#### Riepilogo Strokes Gained (modello practice)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Media SG", f"{sg.mean():+.3f}")
    c2.metric("Totale SG", f"{sg.sum():+.3f}")
    c3.metric("Colpi", f"{len(sg)}")
    c4.metric("Migliore", f"{sg.max():+.3f}")
    hist = px.histogram(sg, nbins=20, title="Distribuzione SG", color_discrete_sequence=[ORANGE])
    hist.update_layout(showlegend=False)
    st.plotly_chart(hist, use_container_width=True)


def satisfaction_breakdown(df: pd.DataFrame, cat_key: str) -> None:
    sub = df[df["Category"] == cat_key]
    if sub.empty:
        return
    plot_pie(sub, "Rating", "Distribuzione voto colpo (1–5)", "Legenda: voto auto-valutato.")
    plot_pie(sub, "Mental_Reaction", "Reazione mentale", "Legenda: reazioni dichiarate dopo il colpo.")


def trend_panel(df_sector: pd.DataFrame, sector_label: str) -> None:
    if df_sector.empty:
        return
    d = df_sector.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"])
    if d.empty:
        return
    d["Rating"] = pd.to_numeric(d["Rating"], errors="coerce")
    d["Strokes_Gained"] = pd.to_numeric(d["Strokes_Gained"], errors="coerce")
    grp = (
        d.groupby("Date", as_index=False)
        .agg(rating_mean=("Rating", "mean"), sg_mean=("Strokes_Gained", "mean"), shots=("Category", "count"))
        .sort_values("Date")
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=grp["Date"], y=grp["rating_mean"], mode="lines+markers", name="Voto medio", line=dict(color=OCHRE, width=3))
    )
    fig.add_trace(
        go.Scatter(
            x=grp["Date"], y=grp["sg_mean"], mode="lines+markers", name="SG medio",
            line=dict(color=BLACK, width=2), yaxis="y2",
        )
    )
    fig.update_layout(
        title=f"Andamento performance - {sector_label}",
        yaxis=dict(title="Voto medio (1-5)"),
        yaxis2=dict(title="SG medio", overlaying="y", side="right"),
    )
    st.plotly_chart(fig, use_container_width=True)


def club_breakdown_table(df_sector: pd.DataFrame) -> None:
    d = df_sector.copy()
    if d.empty:
        return
    d["Rating"] = pd.to_numeric(d["Rating"], errors="coerce")
    d["Strokes_Gained"] = pd.to_numeric(d["Strokes_Gained"], errors="coerce")
    g = (
        d.groupby("Club", as_index=False)
        .agg(Colpi=("Club", "count"), Voto_medio=("Rating", "mean"), SG_medio=("Strokes_Gained", "mean"))
        .sort_values(["Colpi", "Voto_medio"], ascending=[False, False])
    )
    st.markdown("#### Ranking bastoni")
    st.dataframe(g.style.format({"Voto_medio": "{:.2f}", "SG_medio": "{:+.3f}"}), use_container_width=True, hide_index=True)


def sg_distance_table(df_sector: pd.DataFrame) -> None:
    d = df_sector.copy()
    d["Start_Dist_m"] = pd.to_numeric(d["Start_Dist_m"], errors="coerce")
    d["Strokes_Gained"] = pd.to_numeric(d["Strokes_Gained"], errors="coerce")
    d = d.dropna(subset=["Start_Dist_m", "Strokes_Gained"])
    if d.empty:
        return
    bins = [0, 2, 5, 10, 20, 35, 50, 80, 130, 200, 600]
    labels = ["0-2", "2-5", "5-10", "10-20", "20-35", "35-50", "50-80", "80-130", "130-200", "200+"]
    d["Distance_Bucket"] = pd.cut(d["Start_Dist_m"], bins=bins, labels=labels, include_lowest=True, right=False)
    g = d.groupby("Distance_Bucket", as_index=False).agg(Colpi=("Strokes_Gained", "count"), SG_medio=("Strokes_Gained", "mean")).dropna()
    if g.empty:
        return
    fig = px.bar(g, x="Distance_Bucket", y="SG_medio", color="SG_medio", color_continuous_scale="RdYlGn", title="SG per fascia distanza")
    fig.add_hline(y=0, line_dash="dash", line_color=OCHRE_DARK)
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


def directional_bias_panel(df_sector: pd.DataFrame) -> None:
    d = df_sector.copy()
    d["x"] = pd.to_numeric(d["Proximity_Lateral_m"], errors="coerce")
    d = d.dropna(subset=["x"])
    if d.empty:
        return
    left = int((d["x"] < 0).sum())
    right = int((d["x"] > 0).sum())
    center = int((d["x"] == 0).sum())
    total = len(d)
    bias = pd.DataFrame({
        "Direzione": ["Sinistra", "In linea", "Destra"],
        "Colpi": [left, center, right],
        "Percentuale": [left / total * 100, center / total * 100, right / total * 100],
    })
    fig = px.bar(
        bias, x="Direzione", y="Percentuale",
        text=bias["Percentuale"].map(lambda v: f"{v:.1f}%"),
        color="Direzione",
        color_discrete_map={"Sinistra": "#d45858", "In linea": SUCCESS_GREEN, "Destra": ACCENT_BLUE},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Wizard inserimento — meno tap, distanze lette al conferma/salva
# =============================================================================
def _pick_club(shot: dict, clubs: list[str], cols_n: int, key_p: str, next_step: int) -> None:
    cols = st.columns(cols_n)
    for i, cl in enumerate(clubs):
        if cols[i % cols_n].button(cl, key=f"{key_p}{i}", use_container_width=True):
            shot["Club"] = cl
            st.session_state["wz_step"] = next_step
            st.rerun()


def _pick_option(shot: dict, field: str, opt: str, next_step: int, extra: dict | None = None) -> None:
    shot[field] = opt
    if extra:
        shot.update(extra)
    st.session_state["wz_step"] = next_step
    st.rerun()


def wizard_range(session_name: str, user: str) -> None:
    st.session_state.setdefault("wz_step", 0)
    step = st.session_state["wz_step"]
    shot: dict[str, Any] = st.session_state.setdefault("wz_payload", {})
    total = 9

    if step == 0:
        step_badge(1, total, "Bastone")
        _pick_club(shot, CLUBS_LONG, 4, "zr_cl_", 1)
    elif step == 1:
        step_badge(2, total, "Impatto")
        for opt in LONG_IMPACT:
            if st.button(opt, key=f"zr_im_{opt}", use_container_width=True):
                _pick_option(shot, "Impact", opt, 2)
    elif step == 2:
        step_badge(3, total, "Curvatura")
        for opt in LONG_CURVE:
            if st.button(opt, key=f"zr_cv_{opt}", use_container_width=True):
                _pick_option(shot, "Curvature", opt, 3, {"Trajectory": ""})
    elif step == 3:
        step_badge(4, total, "Direzione")
        for opt in LONG_DIR:
            if st.button(opt, key=f"zr_dir_{opt}", use_container_width=True):
                _pick_option(shot, "Direction_LR", opt, 4)
    elif step == 4:
        step_badge(5, total, "Errore laterale")
        distance_input("Metri a destra/sinistra dal punto mirato", "wz_range_lat", 0.0, 120.0, 0.5, [0, 2, 5, 10, 20])
        if st.button("Conferma laterale", type="primary", use_container_width=True):
            lat = read_distance("wz_range_lat", 0.0)
            shot["Proximity_Lateral_m"] = lat_sign(str(shot.get("Direction_LR", "")), lat)
            st.session_state["wz_step"] = 5
            st.rerun()
    elif step == 5:
        step_badge(6, total, "Errore profondità")
        distance_input("Quanti metri corto/lungo?", "wz_range_depth", 0.0, 120.0, 0.5, [0, 2, 5, 10, 20])
        sense = st.radio("Senso", ["In linea col bersaglio", "Corto del bersaglio", "Lungo del bersaglio"], horizontal=True)
        if st.button("Conferma profondità", type="primary", use_container_width=True):
            shot["Proximity_Depth_m"] = depth_sign(read_distance("wz_range_depth", 0.0), sense)
            st.session_state["wz_step"] = 6
            st.rerun()
    elif step == 6:
        step_badge(7, total, "Voto 1–5")
        cols = st.columns(5)
        for v in range(1, 6):
            if cols[v - 1].button(str(v), key=f"zr_rt_{v}", use_container_width=True):
                shot["Rating"] = v
                st.session_state["wz_step"] = 7
                st.rerun()
    elif step == 7:
        step_badge(8, total, "Reazione mentale")
        cols = st.columns(2)
        for i, opt in enumerate(MENTAL_OPTIONS):
            if cols[i % 2].button(opt, key=f"zr_mn_{opt}", use_container_width=True):
                shot["Mental_Reaction"] = opt
                st.session_state["wz_step"] = 8
                st.rerun()
    elif step == 8:
        step_badge(9, total, "Strokes Gained")
        shot["Lie_Long"] = st.radio("Lie di partenza", ["Tee", "Fairway"], horizontal=True)
        distance_input(
            "Distanza dalla buca prima del colpo (m)",
            "wz_range_hole_start", 0.0, 550.0, 1.0,
            [40, 80, 120, 160, 200, 250],
            default=120.0,
        )
        distance_input(
            "Distanza dalla buca dopo il colpo (m)",
            "wz_range_hole_end", 0.0, 550.0, 1.0,
            [0, 10, 30, 60, 100, 150],
            default=30.0,
        )
        lie_after = st.selectbox(
            "Lie dopo il colpo",
            ["Fairway", "First cut", "Semi-rough", "Rough", "Bunker", "Fringe", "Green"],
        )
        if st.button("Salva colpo RANGE", type="primary", use_container_width=True):
            hole_start = read_distance("wz_range_hole_start", 120.0)
            hole_end = read_distance("wz_range_hole_end", 30.0)
            from_tee = shot.get("Lie_Long") == "Tee"
            sg = compute_sg_long(hole_start, hole_end, from_tee, lie_after)
            row = {
                "User": user,
                "Date": datetime.date.today(),
                "SessionName": session_name,
                "Time": datetime.datetime.now().strftime("%H:%M"),
                "Category": "RANGE",
                "Club": shot.get("Club", ""),
                "Impact": shot.get("Impact", ""),
                "Curvature": shot.get("Curvature", ""),
                "Trajectory": "",
                "Lie_Start": shot.get("Lie_Long", ""),
                "Lie_End": lie_after,
                "Direction_LR": shot.get("Direction_LR", ""),
                "Proximity_Lateral_m": shot.get("Proximity_Lateral_m", np.nan),
                "Proximity_Depth_m": shot.get("Proximity_Depth_m", np.nan),
                "Start_Dist_m": hole_start,
                "End_Dist_m": hole_end,
                "Hole_Dist_Start_m": hole_start,
                "Hole_Dist_End_m": hole_end,
                "Lie_Long": shot.get("Lie_Long", ""),
                "Rating": shot.get("Rating", np.nan),
                "Mental_Reaction": shot.get("Mental_Reaction", ""),
                "Strokes_Gained": sg,
            }
            save_shot(row)
            st.success("Colpo RANGE salvato.")
            reset_wizard()
            st.rerun()

    if st.button("Annulla inserimento", key="cancel_r"):
        reset_wizard()
        st.rerun()


def wizard_short(session_name: str, user: str) -> None:
    st.session_state.setdefault("wz_step", 0)
    step = st.session_state["wz_step"]
    shot: dict[str, Any] = st.session_state.setdefault("wz_payload", {})
    total = 11

    if step == 0:
        step_badge(1, total, "Bastone")
        _pick_club(shot, CLUBS_SHORT, 4, "zs_cl_", 1)
    elif step == 1:
        step_badge(2, total, "Distanza iniziale")
        distance_input("Distanza iniziale dalla buca (m)", "wz_short_start", 0.0, 50.0, 0.5, [5, 10, 15, 20, 30, 40], default=20.0)
        if st.button("Conferma distanza iniziale", type="primary", use_container_width=True):
            shot["Start_Dist_m"] = read_distance("wz_short_start", 20.0)
            st.session_state["wz_step"] = 2
            st.rerun()
    elif step == 2:
        step_badge(3, total, "Lie iniziale")
        cols = st.columns(2)
        for i, opt in enumerate(SHORT_LIE_START):
            if cols[i % 2].button(opt, key=f"zs_ls_{i}", use_container_width=True):
                shot["Lie_Start"] = opt
                st.session_state["wz_step"] = 3
                st.rerun()
    elif step == 3:
        step_badge(4, total, "Distanza finale")
        distance_input("Distanza finale dalla buca (m)", "wz_short_end", 0.0, 80.0, 0.5, [0, 1, 2, 4, 8, 15], default=4.0)
        if st.button("Conferma distanza finale", type="primary", use_container_width=True):
            shot["End_Dist_m"] = read_distance("wz_short_end", 4.0)
            st.session_state["wz_step"] = 4
            st.rerun()
    elif step == 4:
        step_badge(5, total, "Lie finale")
        cols = st.columns(2)
        for i, opt in enumerate(SHORT_LIE_END):
            if cols[i % 2].button(opt, key=f"zs_le_{i}", use_container_width=True):
                shot["Lie_End"] = opt
                st.session_state["wz_step"] = 5
                st.rerun()
    elif step == 5:
        step_badge(6, total, "Impatto")
        for opt in SHORT_IMPACT:
            if st.button(opt, key=f"zs_im_{opt}", use_container_width=True):
                _pick_option(shot, "Impact", opt, 6, {"Curvature": ""})
    elif step == 6:
        step_badge(7, total, "Direzione")
        for opt in SHORT_DIR:
            if st.button(opt, key=f"zs_dir_{opt}", use_container_width=True):
                _pick_option(shot, "Direction_LR", opt, 7)
    elif step == 7:
        step_badge(8, total, "Errore laterale")
        distance_input("Metri a destra/sinistra dalla buca", "wz_short_lat", 0.0, 80.0, 0.5, [0, 1, 2, 4, 8])
        if st.button("Conferma laterale", type="primary", use_container_width=True):
            shot["Proximity_Lateral_m"] = lat_sign(str(shot.get("Direction_LR", "")), read_distance("wz_short_lat", 0.0))
            st.session_state["wz_step"] = 8
            st.rerun()
    elif step == 8:
        step_badge(9, total, "Profondità")
        distance_input("Metri corto/lungo", "wz_short_depth", 0.0, 80.0, 0.5, [0, 1, 2, 4, 8])
        sense = st.radio("Senso", ["In linea", "Corto", "Lungo"], horizontal=True)
        conv = {"In linea": "In linea col bersaglio", "Corto": "Corto del bersaglio", "Lungo": "Lungo del bersaglio"}
        if st.button("Conferma profondità", type="primary", use_container_width=True):
            shot["Proximity_Depth_m"] = depth_sign(read_distance("wz_short_depth", 0.0), conv[sense])
            st.session_state["wz_step"] = 9
            st.rerun()
    elif step == 9:
        step_badge(10, total, "Voto 1–5")
        cols = st.columns(5)
        for v in range(1, 6):
            if cols[v - 1].button(str(v), key=f"zs_rt_{v}", use_container_width=True):
                shot["Rating"] = v
                st.session_state["wz_step"] = 10
                st.rerun()
    elif step == 10:
        step_badge(11, total, "Reazione mentale")
        cols = st.columns(2)
        for i, opt in enumerate(MENTAL_OPTIONS):
            if cols[i % 2].button(opt, key=f"zs_mn_{opt}", use_container_width=True):
                shot["Mental_Reaction"] = opt
        if st.button("Salva gioco corto", type="primary", use_container_width=True):
            if "Mental_Reaction" not in shot:
                st.error("Seleziona prima la reazione mentale.")
                return
            start_m = float(shot.get("Start_Dist_m", read_distance("wz_short_start", 0)))
            end_m = float(shot.get("End_Dist_m", read_distance("wz_short_end", 0)))
            sg = compute_sg_short(start_m, end_m, str(shot.get("Lie_Start", "")), str(shot.get("Lie_End", "")))
            row = {
                "User": user,
                "Date": datetime.date.today(),
                "SessionName": session_name,
                "Time": datetime.datetime.now().strftime("%H:%M"),
                "Category": "SHORT",
                "Club": shot.get("Club", ""),
                "Impact": shot.get("Impact", ""),
                "Curvature": "",
                "Trajectory": "",
                "Lie_Start": shot.get("Lie_Start", ""),
                "Lie_End": shot.get("Lie_End", ""),
                "Direction_LR": shot.get("Direction_LR", ""),
                "Proximity_Lateral_m": shot.get("Proximity_Lateral_m", np.nan),
                "Proximity_Depth_m": shot.get("Proximity_Depth_m", np.nan),
                "Start_Dist_m": start_m,
                "End_Dist_m": end_m,
                "Hole_Dist_Start_m": start_m,
                "Hole_Dist_End_m": end_m,
                "Lie_Long": "",
                "Rating": shot.get("Rating", np.nan),
                "Mental_Reaction": shot.get("Mental_Reaction", ""),
                "Strokes_Gained": sg,
            }
            save_shot(row)
            st.success("Gioco corto salvato.")
            reset_wizard()
            st.rerun()

    if st.button("Annulla inserimento", key="cancel_s"):
        reset_wizard()
        st.rerun()


def wizard_putt(session_name: str, user: str) -> None:
    st.session_state.setdefault("wz_step", 0)
    step = st.session_state["wz_step"]
    shot: dict[str, Any] = st.session_state.setdefault("wz_payload", {})
    total = 6

    if step == 0:
        step_badge(1, total, "Distanza iniziale")
        distance_input("Distanza iniziale (m)", "wz_putt_start", 0.0, 60.0, 0.1, [0.5, 1, 2, 3, 4, 8], default=2.0)
        if st.button("Avanti", type="primary", use_container_width=True):
            shot["Start_Dist_m"] = read_distance("wz_putt_start", 2.0)
            st.session_state["wz_step"] = 1
            st.rerun()
    elif step == 1:
        step_badge(2, total, "Distanza finale")
        distance_input("Distanza finale — 0 se in buca", "wz_putt_end", 0.0, 30.0, 0.05, [0, 0.3, 0.6, 1.0, 2.0], default=0.0)
        if st.button("Conferma distanze", type="primary", use_container_width=True):
            shot["Start_Dist_m"] = read_distance("wz_putt_start", shot.get("Start_Dist_m", 2.0))
            shot["End_Dist_m"] = read_distance("wz_putt_end", 0.0)
            st.session_state["wz_step"] = 2
            st.rerun()
    elif step == 2:
        step_badge(3, total, "Impatto")
        cols = st.columns(2)
        for i, opt in enumerate(PUTT_IMPACT):
            if cols[i % 2].button(opt, key=f"zp_im_{opt}", use_container_width=True):
                shot["Impact"] = opt
                st.session_state["wz_step"] = 3
                st.rerun()
    elif step == 3:
        step_badge(4, total, "Traiettoria")
        for opt in PUTT_TRAJ:
            if st.button(opt, key=f"zp_tr_{opt}", use_container_width=True):
                shot["Trajectory"] = opt
                shot["Curvature"] = opt
                st.session_state["wz_step"] = 4
                st.rerun()
    elif step == 4:
        step_badge(5, total, "Voto")
        cols = st.columns(5)
        for v in range(1, 6):
            if cols[v - 1].button(str(v), key=f"zp_rt_{v}", use_container_width=True):
                shot["Rating"] = v
                st.session_state["wz_step"] = 5
                st.rerun()
    elif step == 5:
        step_badge(6, total, "Mentale e salva")
        cols = st.columns(2)
        for i, opt in enumerate(MENTAL_OPTIONS):
            if cols[i % 2].button(opt, key=f"zp_mn_{opt}", use_container_width=True):
                shot["Mental_Reaction"] = opt
        if st.button("Salva putt", type="primary", use_container_width=True):
            if "Mental_Reaction" not in shot:
                st.error("Seleziona la reazione mentale.")
                return
            start_m = float(shot.get("Start_Dist_m", read_distance("wz_putt_start", 0)))
            end_m = float(shot.get("End_Dist_m", read_distance("wz_putt_end", 0)))
            sg = compute_sg_putt(start_m, end_m)
            row = {
                "User": user,
                "Date": datetime.date.today(),
                "SessionName": session_name,
                "Time": datetime.datetime.now().strftime("%H:%M"),
                "Category": "PUTT",
                "Club": "Putter",
                "Impact": shot.get("Impact", ""),
                "Curvature": shot.get("Curvature", ""),
                "Trajectory": shot.get("Trajectory", ""),
                "Lie_Start": "Green",
                "Lie_End": "Green",
                "Direction_LR": "",
                "Proximity_Lateral_m": np.nan,
                "Proximity_Depth_m": np.nan,
                "Start_Dist_m": start_m,
                "End_Dist_m": end_m,
                "Hole_Dist_Start_m": start_m,
                "Hole_Dist_End_m": end_m,
                "Lie_Long": "",
                "Rating": shot.get("Rating", np.nan),
                "Mental_Reaction": shot.get("Mental_Reaction", ""),
                "Strokes_Gained": sg,
            }
            save_shot(row)
            st.success("Putt salvato.")
            reset_wizard()
            st.rerun()

    if st.button("Annulla inserimento", key="cancel_p"):
        reset_wizard()
        st.rerun()


# =============================================================================
# Review
# =============================================================================
def review_panel(user: str, session_name: str) -> None:
    df_all = load_data()
    df_u = df_all[df_all["User"] == user]
    render_hero(
        "Review performance",
        "Dashboard con grafici, SG e tabelle per periodo e settore.",
        ["Pie charts", "Dispersione", "Strokes Gained", "Trend"],
    )
    period = st.selectbox("Periodo", PERIOD_LABELS, key="rev_period")
    df_f = filter_period(df_u, session_name, period)
    sector = st.radio(
        "Settore",
        ["RANGE", "SHORT", "PUTT"],
        format_func=lambda x: CATEGORIES[x],
        horizontal=True,
        key="rev_sector",
    )
    dsec = df_f[df_f["Category"] == sector]
    st.caption(f"Utente **{user}** · **{period}** · **{CATEGORIES[sector]}** · n = **{len(dsec)}**")

    if dsec.empty:
        st.info("Nessun colpo in questo filtro.")
        brand_footer()
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Colpi", len(dsec))
    rmean = pd.to_numeric(dsec["Rating"], errors="coerce").mean()
    m2.metric("Voto medio", f"{rmean:.2f}" if pd.notna(rmean) else "—")
    sg_series = pd.to_numeric(dsec["Strokes_Gained"], errors="coerce").dropna()
    m3.metric("SG medio", f"{sg_series.mean():+.3f}" if len(sg_series) else "—")

    shots_cols = [
        "Date", "Time", "SessionName", "Category", "Club", "Impact", "Curvature",
        "Direction_LR", "Proximity_Lateral_m", "Proximity_Depth_m", "Rating", "Strokes_Gained",
    ]
    st.dataframe(
        dsec[[c for c in shots_cols if c in dsec.columns]].sort_values(["Date", "Time"], ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    sg_summary_table(df_f, sector)
    trend_panel(dsec, CATEGORIES[sector])
    club_breakdown_table(dsec)
    sg_distance_table(dsec)
    if sector in ("RANGE", "SHORT"):
        directional_bias_panel(dsec)

    if sector == "RANGE":
        plot_pie(dsec, "Impact", "Impatto", "Legenda impatti.")
        plot_pie(dsec, "Curvature", "Curvatura", "Legenda curvatura.")
        plot_dispersion(dsec, "Dispersione RANGE")
        satisfaction_breakdown(df_f, "RANGE")
    elif sector == "SHORT":
        plot_pie(dsec, "Lie_Start", "Lie iniziale", "")
        plot_pie(dsec, "Impact", "Impatto", "")
        plot_dispersion(dsec, "Dispersione gioco corto")
        satisfaction_breakdown(df_f, "SHORT")
    else:
        plot_pie(dsec, "Impact", "Impatto putter", "")
        putting_make_table(dsec)
        satisfaction_breakdown(df_f, "PUTT")

    brand_footer()


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    inject_styles()

    if "splash_done" not in st.session_state:
        run_splash_sequence()
        st.session_state["splash_done"] = True
        st.rerun()

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "post_auth_logo_pending" not in st.session_state:
        st.session_state["post_auth_logo_pending"] = False

    if not st.session_state["logged_in"]:
        login_screen()
        return

    user = str(st.session_state["user"])
    if st.session_state.get("post_auth_logo_pending", False):
        run_post_auth_logo()
        st.session_state["post_auth_logo_pending"] = False
        st.rerun()

    brand_header("Profilo")
    st.write(f"**Atleta:** {user}")
    session_name = st.text_input(
        "Nome sessione / note",
        value=st.session_state.get("session_name_main", "Sessione Allenamento"),
        key="session_name_main",
    )
    page = st.radio(
        "Scegli sezione",
        ["Inserimento dati", "Review", "Diario di Gioco"],
        horizontal=True,
        key="main_page_home",
    )
    _, c_logout = st.columns([3, 1])
    with c_logout:
        if st.button("Logout", use_container_width=True):
            st.session_state["logged_in"] = False
            st.session_state.pop("user", None)
            st.rerun()

    render_command_header(page)

    if page == "Inserimento dati":
        render_hero(
            "Inserimento rapido",
            "Pulsanti grandi, preset distanza affidabili, meno conferme ridondanti.",
            ["Range", "Short game", "Putting"],
        )
        st.session_state.setdefault("wz_cat", None)
        if st.session_state["wz_cat"] is None:
            st.markdown("#### Scegli il settore")
            c1, c2, c3 = st.columns(3)
            if c1.button("RANGE\n(gioco lungo)", use_container_width=True):
                reset_wizard()
                st.session_state["wz_cat"] = "RANGE"
                st.rerun()
            if c2.button("Gioco corto\n(<50 m)", use_container_width=True):
                reset_wizard()
                st.session_state["wz_cat"] = "SHORT"
                st.rerun()
            if c3.button("Putting", use_container_width=True):
                reset_wizard()
                st.session_state["wz_cat"] = "PUTT"
                st.rerun()
            brand_footer()
        else:
            st.caption(f"Sessione: **{session_name}**")
            if st.button("← Cambia settore"):
                reset_wizard()
                st.session_state["wz_cat"] = None
                st.rerun()
            cat = st.session_state["wz_cat"]
            if cat == "RANGE":
                wizard_range(session_name, user)
            elif cat == "SHORT":
                wizard_short(session_name, user)
            else:
                wizard_putt(session_name, user)
            brand_footer()

    elif page == "Diario di Gioco":
        brand_header("Diario di Gioco")
        diario_panel(user, session_name)

    else:
        brand_header()
        review_panel(user, session_name)


if __name__ == "__main__":
    main()


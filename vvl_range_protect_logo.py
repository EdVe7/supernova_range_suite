"""
Supernova Range Suite — tracking allenamento (range, gioco corto, putting).
UI mobile-first, tema bianco/oro, persistenza Google Sheets. Nessun export PDF.
"""

from __future__ import annotations

import datetime
import time
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

# =============================================================================
# Config pagina
# =============================================================================
st.set_page_config(
    page_title="Supernova Range Suite",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
    <style>
    #MainMenu {{visibility: hidden;}} footer {{visibility: hidden;}} header {{visibility: hidden;}}
    .stDeployButton {{display:none;}}
    [data-testid="stToolbar"] {{visibility: hidden !important;}}
    .stApp {{background: linear-gradient(180deg, #FFFFFF 0%, #FFFBEF 75%, #F8EFCF 100%);}}
    </style>
    """, unsafe_allow_html=True)

GOLD = "#C9A227"
GOLD_LIGHT = "#E8D48A"
GOLD_DARK = "#8B6914"
WHITE = "#FFFFFF"
OFF_WHITE = "#FAFAF8"
TEXT = "#2B2B2B"
MUTED = "#6B6560"

PASSWORD_DEFAULT = "supernova.analytics"

CATEGORIES = {
    "RANGE": "Gioco lungo / Range",
    "SHORT": "Gioco corto (<50 m)",
    "PUTT": "Putting",
}

# Schema colonne foglio (ordine stabile per concat/update)
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
SHORT_LIE_START = ["Fairway", "Rough", "Bunker"]
SHORT_LIE_END = ["Fairway", "Rough", "Bunker", "Green"]
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
    "DR",
    "3W",
    "5W",
    "7W",
    "3H",
    "3i",
    "4i",
    "5i",
    "6i",
    "7i",
    "8i",
    "9i",
    "PW",
    "AW",
    "GW",
    "SW",
    "LW",
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


def inject_styles() -> None:
    st.markdown(
        f"""
<style>
    #MainMenu {{visibility: hidden; height: 0;}}
    footer {{visibility: hidden; height: 0;}}
    header [data-testid="stHeader"] {{background: {WHITE};}}
    .stApp {{ background: radial-gradient(circle at 15% -5%, #f6ecd0 0%, {OFF_WHITE} 30%, {WHITE} 70%); color: {TEXT}; }}
    .block-container {{ padding-top: 0.75rem; padding-bottom: 4rem; max-width: 760px; }}
    h1, h2, h3 {{ color: {GOLD_DARK}; font-weight: 700; letter-spacing: 0.02em; }}
    [data-testid="stTabs"] button[role="tab"] {{
        font-size: 1rem !important;
        font-weight: 700 !important;
        border-radius: 12px 12px 0 0 !important;
        color: {MUTED} !important;
    }}
    [data-testid="stTabs"] button[aria-selected="true"] {{
        color: {GOLD_DARK} !important;
        background: linear-gradient(180deg, #fffdf7, #f6f1df) !important;
    }}
    div[data-testid="stHorizontalBlock"] button {{
        min-height: 3.2rem !important;
        font-size: 1.05rem !important;
        border-radius: 14px !important;
        border: 2px solid {GOLD} !important;
        background: {WHITE} !important;
        color: {TEXT} !important;
        font-weight: 600 !important;
    }}
    div[data-testid="stHorizontalBlock"] button:hover {{
        background: linear-gradient(180deg, {GOLD_LIGHT}, {GOLD}) !important;
        color: #111 !important;
    }}
    .sn-big-btn > button {{
        width: 100%;
        min-height: 3.5rem;
        font-size: 1.1rem;
        border-radius: 16px;
        background: {WHITE};
        border: 2px solid {GOLD};
        font-weight: 700;
    }}
    .sn-footer {{
        text-align: center;
        color: {MUTED};
        font-size: 0.85rem;
        margin-top: 2rem;
        padding: 0.75rem;
        border-top: 1px solid {GOLD_LIGHT};
    }}
    .sn-logo-caption {{
        font-style: italic;
        color: {GOLD_DARK};
        font-weight: 600;
        font-size: 0.95rem;
        margin: 0;
    }}
    [data-testid="stSidebar"] {{
        background: {OFF_WHITE};
    }}
    .sn-hero {{
        background: linear-gradient(120deg, #fffef9, #f8f2df);
        border: 1px solid #e8dcb4;
        border-radius: 18px;
        padding: 14px 16px;
        margin: 8px 0 16px 0;
        box-shadow: 0 3px 14px rgba(139, 105, 20, 0.07);
    }}
    .sn-hero-title {{
        font-size: 1.05rem;
        font-weight: 800;
        color: {GOLD_DARK};
        margin-bottom: 4px;
    }}
    .sn-hero-sub {{
        color: {MUTED};
        font-size: 0.92rem;
        margin: 0;
    }}
    .sn-chip {{
        display: inline-block;
        background: #fff;
        border: 1px solid #e7d9ab;
        border-radius: 999px;
        padding: 4px 10px;
        margin-right: 6px;
        margin-top: 6px;
        color: #6d5717;
        font-size: 0.8rem;
        font-weight: 700;
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
                f"<div style='font-size:1.6rem;font-weight:800;color:{GOLD};'>SUPERNOVA</div>",
                unsafe_allow_html=True,
            )
    with c2:
        if title:
            st.markdown(f"### {title}")
        st.markdown(
            "<p class='sn-logo-caption'>Range Data Suite · Data over talent</p>",
            unsafe_allow_html=True,
        )


def brand_footer() -> None:
    st.markdown(
        "<div class='sn-footer'>Powered by Supernova Sport Science</div>",
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, chips: list[str] | None = None) -> None:
    chips_html = ""
    if chips:
        chips_html = "".join([f"<span class='sn-chip'>{c}</span>" for c in chips])
    st.markdown(
        (
            "<div class='sn-hero'>"
            f"<div class='sn-hero-title'>{title}</div>"
            f"<p class='sn-hero-sub'>{subtitle}</p>"
            f"{chips_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


# =============================================================================
# Strokes gained (modello semplificato da practice — coerente tra settori)
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
    """Colpi attesi PGA-style (approssimazione) da distanza in metri."""
    if distance_m <= 0:
        return 0.0
    xs = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
    ys = [1.02, 1.06, 1.10, 1.15, 1.23, 1.30, 1.38, 1.45, 1.58, 1.72, 1.85, 2.05, 2.25, 2.42, 2.55]
    return float(_interp(distance_m, xs, ys))


def expected_short_hole(dist_m: float, lie: str) -> float:
    """Colpi attesi fino alla buca dal gioco corto (non green)."""
    if dist_m <= 0:
        return 0.0
    lie_adj = {"Fairway": 0.0, "Rough": 0.18, "Bunker": 0.55, "Green": 0.0}
    base = 2.08 + (dist_m / 45.0) * 0.95
    return float(base + lie_adj.get(lie, 0.0))


def expected_long_hole(dist_m: float, from_tee: bool) -> float:
    """Approccio / tee: colpi attesi verso la buca prima/dopo il colpo."""
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
    exp_before = expected_putts(start_m)
    exp_after = expected_putts(end_m)
    return float(exp_before - exp_after - 1.0)


def compute_sg_short(start_m: float, end_m: float, lie_s: str, lie_e: str) -> float:
    def exp_at(d: float, lie: str) -> float:
        if lie == "Green":
            return expected_putts(d)
        return expected_short_hole(d, lie)

    exp_before = exp_at(start_m, lie_s)
    exp_after = exp_at(end_m, lie_e)
    return float(exp_before - exp_after - 1.0)


def compute_sg_long(start_before_m: float, start_after_m: float, from_tee: bool, lie_after: str) -> float:
    exp_before = expected_long_hole(start_before_m, from_tee)
    use_fairway = lie_after.lower() == "fairway"
    exp_after = expected_long_hole(start_after_m, from_tee=False) if use_fairway else expected_short_hole(start_after_m, lie_after)
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
            "Proximity_Lateral_m",
            "Proximity_Depth_m",
            "Start_Dist_m",
            "End_Dist_m",
            "Hole_Dist_Start_m",
            "Hole_Dist_End_m",
            "Rating",
            "Strokes_Gained",
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


# =============================================================================
# Splash & login
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
                            f"<h1 style='text-align:center;color:{GOLD};'>SUPERNOVA</h1>",
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        f"<p style='text-align:center;color:{MUTED};'>Range Data Suite</p>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<h2 style='text-align:center;color:{GOLD_DARK};margin-top:4rem;'>{msg}</h2>",
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
        env_p = None
        try:
            env_p = st.secrets.get("APP_PASSWORD")
        except Exception:
            env_p = None
        if env_p:
            pwd_ok = pwd_ok or (p == str(env_p))
        if pwd_ok:
            st.session_state["logged_in"] = True
            st.session_state["user"] = u.upper()
            st.rerun()
        else:
            st.error("Credenziali non valide.")
    brand_footer()
    st.stop()


# =============================================================================
# Helpers UI wizard
# =============================================================================
def reset_wizard() -> None:
    for k in list(st.session_state.keys()):
        if k.startswith("wz_"):
            del st.session_state[k]
    st.session_state["wz_cat"] = None
    st.session_state["wz_step"] = 0


def lat_sign(direction: str, lateral_abs: float) -> float:
    if direction.startswith("A destra"):
        return float(abs(lateral_abs))
    if direction.startswith("A sinistra"):
        return -float(abs(lateral_abs))
    return 0.0


def depth_sign(depth_m: float, label: str) -> float:
    """Profondità: positivo = lungo, negativo = corto (optional convention)."""
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
        color_discrete_sequence=px.colors.sequential.YlOrBr,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        legend_title_text="Legenda",
        font=dict(color=TEXT),
        title=dict(font=dict(size=18, color=GOLD_DARK)),
        margin=dict(t=48, b=24, l=24, r=24),
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
        st.info("Aggiungi errore laterale e profondità per vedere la dispersione dall’alto.")
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
    )
    fig.add_vline(x=0, line_dash="dash", line_color=GOLD)
    fig.add_hline(y=0, line_dash="dash", line_color=GOLD)
    fig.update_layout(
        legend_title_text="Legenda",
        font=dict(color=TEXT),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Ogni punto è un colpo visto dall’alto: incrocio delle linee = bersaglio. "
        "L’asse orizzontale è l’errore a sinistra/destra, quello verticale corto/lungo."
    )


def putting_make_table(df_putt: pd.DataFrame) -> None:
    """Bucket da 15 m in giù ogni 2 m."""
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
    st.caption(
        "Percentuale di putt chiusi in buca al primo tentativo (distanza finale = 0 m), "
        "raggruppati per ampiezza di 2 metri fino a 15 m."
    )
    st.dataframe(
        out.style.format({"% Made": "{:.1f}%"}),
        use_container_width=True,
        hide_index=True,
    )


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
    st.caption(
        "Valori positivi indicano un colpo migliore della media di riferimento usata dal modello "
        "(approssimazione didattica, non ufficiale PGA)."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Media SG", f"{sg.mean():+.3f}")
    c2.metric("Totale SG", f"{sg.sum():+.3f}")
    c3.metric("Colpi", f"{len(sg)}")
    c4.metric("Migliore", f"{sg.max():+.3f}")
    hist = px.histogram(
        sg,
        nbins=20,
        title="Distribuzione SG colpo per colpo",
        labels={"value": "Strokes gained per colpo", "count": "Numero di colpi"},
        color_discrete_sequence=[GOLD],
    )
    hist.update_layout(showlegend=False)
    st.plotly_chart(hist, use_container_width=True)


def satisfaction_breakdown(df: pd.DataFrame, cat_key: str) -> None:
    sub = df[df["Category"] == cat_key]
    if sub.empty:
        return
    plot_pie(
        sub,
        "Rating",
        "Distribuzione voto colpo (1–5)",
        "Legenda: percentuale di colpi per ogni voto di qualità auto-valutata.",
    )
    plot_pie(
        sub,
        "Mental_Reaction",
        "Reazione mentale",
        "Legenda: mix delle reazioni emotive/cognitive dichiarate dopo il colpo.",
    )


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
        .agg(
            rating_mean=("Rating", "mean"),
            sg_mean=("Strokes_Gained", "mean"),
            shots=("Category", "count"),
        )
        .sort_values("Date")
    )
    st.markdown("#### Trend giornaliero")
    st.caption(
        "Linea oro = voto medio; linea scura = strokes gained medio per giorno. "
        "Serve a capire se la qualità sale o scende nel tempo."
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grp["Date"],
            y=grp["rating_mean"],
            mode="lines+markers",
            name="Voto medio",
            line=dict(color=GOLD, width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grp["Date"],
            y=grp["sg_mean"],
            mode="lines+markers",
            name="SG medio",
            line=dict(color="#5c4a12", width=2),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title=f"Andamento performance - {sector_label}",
        xaxis_title="Data",
        yaxis=dict(title="Voto medio (1-5)"),
        yaxis2=dict(title="SG medio", overlaying="y", side="right"),
        legend_title_text="Legenda",
        margin=dict(t=48, b=24, l=24, r=24),
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
        .agg(
            Colpi=("Club", "count"),
            Voto_medio=("Rating", "mean"),
            SG_medio=("Strokes_Gained", "mean"),
        )
        .sort_values(["Colpi", "Voto_medio"], ascending=[False, False])
    )
    if g.empty:
        return
    st.markdown("#### Ranking bastoni (nel filtro scelto)")
    st.caption(
        "Tabella sintetica per bastone: volume, voto medio e strokes gained medio."
    )
    st.dataframe(
        g.style.format({"Voto_medio": "{:.2f}", "SG_medio": "{:+.3f}"}),
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# Wizard inserimento colpo
# =============================================================================
def wizard_range(session_name: str, user: str) -> None:
    st.session_state.setdefault("wz_step", 0)
    step = st.session_state["wz_step"]
    shot: dict[str, Any] = st.session_state.setdefault("wz_payload", {})

    if step == 0:
        st.markdown("#### Bastone")
        cols = st.columns(3)
        for i, cl in enumerate(CLUBS_LONG):
            if cols[i % 3].button(cl, key=f"cl{i}"):
                shot["Club"] = cl
                st.session_state["wz_step"] = 1
                st.rerun()
    elif step == 1:
        st.markdown("#### Impatto")
        for opt in LONG_IMPACT:
            if st.button(opt, key=f"im{opt}", use_container_width=True):
                shot["Impact"] = opt
                st.session_state["wz_step"] = 2
                st.rerun()
    elif step == 2:
        st.markdown("#### Curvatura palla")
        for opt in LONG_CURVE:
            if st.button(opt, key=f"cv{opt}", use_container_width=True):
                shot["Curvature"] = opt
                shot["Trajectory"] = ""
                st.session_state["wz_step"] = 3
                st.rerun()
    elif step == 3:
        st.markdown("#### Posizione rispetto al bersaglio (linea)")
        for opt in LONG_DIR:
            if st.button(opt, key=f"dir{opt}", use_container_width=True):
                shot["Direction_LR"] = opt
                st.session_state["wz_step"] = 4
                st.rerun()
    elif step == 4:
        st.markdown("#### Errore laterale (metri assoluti)")
        lat = st.number_input("Metri a destra/sinistra dal punto mirato", min_value=0.0, step=0.5)
        if st.button("Conferma errore laterale", use_container_width=True):
            shot["Proximity_Lateral_m"] = lat_sign(shot["Direction_LR"], lat)
            st.session_state["wz_step"] = 5
            st.rerun()
    elif step == 5:
        st.markdown("#### Errore in profondità (per mappa dall’alto)")
        depth_amt = st.number_input("Quanti metri corto/lungo?", min_value=0.0, step=0.5)
        sense = st.radio("Senso", ["In linea col bersaglio", "Corto del bersaglio", "Lungo del bersaglio"])
        if st.button("Conferma profondità", use_container_width=True):
            shot["Proximity_Depth_m"] = depth_sign(depth_amt, sense)
            st.session_state["wz_step"] = 6
            st.rerun()
    elif step == 6:
        st.markdown("#### Voto colpo (1–5)")
        cols = st.columns(5)
        for v in range(1, 6):
            if cols[v - 1].button(str(v)):
                shot["Rating"] = v
                st.session_state["wz_step"] = 7
                st.rerun()
    elif step == 7:
        st.markdown("#### Reazione mentale")
        for opt in MENTAL_OPTIONS:
            if st.button(opt, key=f"mn{opt}", use_container_width=True):
                shot["Mental_Reaction"] = opt
                st.session_state["wz_step"] = 8
                st.rerun()
    elif step == 8:
        st.markdown("#### Dati per Strokes Gained — gioco lungo")
        shot["Lie_Long"] = st.radio("Lie di partenza", ["Tee", "Fairway"])
        shot["Hole_Dist_Start_m"] = st.number_input(
            "Distanza dalla buca prima del colpo (metri)", min_value=0.0, step=5.0
        )
        shot["Hole_Dist_End_m"] = st.number_input(
            "Distanza dalla buca dopo il colpo (metri)", min_value=0.0, step=5.0
        )
        lie_after = st.selectbox(
            "Lie dopo il colpo (per il modello)",
            ["Fairway", "Rough", "Bunker"],
        )
        if st.button("Calcola e salva colpo", type="primary", use_container_width=True):
            from_tee = shot["Lie_Long"] == "Tee"
            sg = compute_sg_long(
                float(shot["Hole_Dist_Start_m"]),
                float(shot["Hole_Dist_End_m"]),
                from_tee,
                lie_after,
            )
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
                "Start_Dist_m": shot.get("Hole_Dist_Start_m", np.nan),
                "End_Dist_m": shot.get("Hole_Dist_End_m", np.nan),
                "Hole_Dist_Start_m": shot.get("Hole_Dist_Start_m", np.nan),
                "Hole_Dist_End_m": shot.get("Hole_Dist_End_m", np.nan),
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

    if step == 0:
        st.markdown("#### Bastone")
        cols = st.columns(4)
        for i, cl in enumerate(CLUBS_SHORT):
            if cols[i % 4].button(cl, key=f"scl{i}"):
                shot["Club"] = cl
                st.session_state["wz_step"] = 1
                st.rerun()
    elif step == 1:
        shot["Start_Dist_m"] = st.number_input(
            "Distanza iniziale dalla buca (metri)", min_value=0.0, max_value=50.0, step=1.0
        )
        if st.button("Conferma distanza", use_container_width=True):
            st.session_state["wz_step"] = 2
            st.rerun()
    elif step == 2:
        st.markdown("#### Lie iniziale")
        for opt in SHORT_LIE_START:
            if st.button(opt, key=f"ls{opt}", use_container_width=True):
                shot["Lie_Start"] = opt
                st.session_state["wz_step"] = 3
                st.rerun()
    elif step == 3:
        shot["End_Dist_m"] = st.number_input("Distanza finale dalla buca (metri)", min_value=0.0, step=0.5)
        if st.button("Conferma distanza finale", use_container_width=True):
            st.session_state["wz_step"] = 4
            st.rerun()
    elif step == 4:
        st.markdown("#### Lie finale")
        for opt in SHORT_LIE_END:
            if st.button(opt, key=f"le{opt}", use_container_width=True):
                shot["Lie_End"] = opt
                st.session_state["wz_step"] = 5
                st.rerun()
    elif step == 5:
        st.markdown("#### Impatto")
        for opt in SHORT_IMPACT:
            if st.button(opt, key=f"sim{opt}", use_container_width=True):
                shot["Impact"] = opt
                shot["Curvature"] = ""
                st.session_state["wz_step"] = 6
                st.rerun()
    elif step == 6:
        st.markdown("#### Direzione rispetto alla buca")
        for opt in SHORT_DIR:
            if st.button(opt, key=f"sd{opt}", use_container_width=True):
                shot["Direction_LR"] = opt
                st.session_state["wz_step"] = 7
                st.rerun()
    elif step == 7:
        lat = st.number_input("Metri a destra/sinistra dalla buca", min_value=0.0, step=0.5)
        if st.button("Conferma errore laterale", use_container_width=True):
            shot["Proximity_Lateral_m"] = lat_sign(shot["Direction_LR"], lat)
            st.session_state["wz_step"] = 8
            st.rerun()
    elif step == 8:
        depth_amt = st.number_input("Metri corto/lungo rispetto alla buca", min_value=0.0, step=0.5)
        sense = st.radio("Senso", ["In linea", "Corto", "Lungo"])
        conv = {"In linea": "In linea col bersaglio", "Corto": "Corto del bersaglio", "Lungo": "Lungo del bersaglio"}
        if st.button("Conferma profondità", use_container_width=True):
            shot["Proximity_Depth_m"] = depth_sign(depth_amt, conv[sense])
            st.session_state["wz_step"] = 9
            st.rerun()
    elif step == 9:
        st.markdown("#### Voto (1–5)")
        cols = st.columns(5)
        for v in range(1, 6):
            if cols[v - 1].button(str(v), key=f"sv{v}"):
                shot["Rating"] = v
                st.session_state["wz_step"] = 10
                st.rerun()
    elif step == 10:
        st.markdown("#### Reazione mentale")
        for opt in MENTAL_OPTIONS:
            if st.button(opt, key=f"smn{opt}", use_container_width=True):
                shot["Mental_Reaction"] = opt
                st.session_state["wz_step"] = 11
                st.rerun()
    elif step == 11:
        st.markdown("#### Strokes gained (usa distanze e lie già inseriti)")
        if st.button("Calcola e salva colpo", type="primary", use_container_width=True):
            sg = compute_sg_short(
                float(shot["Start_Dist_m"]),
                float(shot["End_Dist_m"]),
                str(shot["Lie_Start"]),
                str(shot["Lie_End"]),
            )
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
                "Start_Dist_m": shot.get("Start_Dist_m", np.nan),
                "End_Dist_m": shot.get("End_Dist_m", np.nan),
                "Hole_Dist_Start_m": shot.get("Start_Dist_m", np.nan),
                "Hole_Dist_End_m": shot.get("End_Dist_m", np.nan),
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

    if step == 0:
        shot["Start_Dist_m"] = st.number_input(
            "Distanza iniziale dalla buca (metri)", min_value=0.0, step=0.25
        )
        if st.button("Avanti", use_container_width=True):
            st.session_state["wz_step"] = 1
            st.rerun()
    elif step == 1:
        shot["End_Dist_m"] = st.number_input(
            "Distanza finale (0 se in buca)", min_value=0.0, step=0.1
        )
        if st.button("Conferma distanze", use_container_width=True):
            st.session_state["wz_step"] = 2
            st.rerun()
    elif step == 2:
        st.markdown("#### Impatto sulla faccia")
        for opt in PUTT_IMPACT:
            if st.button(opt, key=f"pi{opt}", use_container_width=True):
                shot["Impact"] = opt
                st.session_state["wz_step"] = 3
                st.rerun()
    elif step == 3:
        st.markdown("#### Traiettoria")
        for opt in PUTT_TRAJ:
            if st.button(opt, key=f"pt{opt}", use_container_width=True):
                shot["Trajectory"] = opt
                shot["Curvature"] = opt
                st.session_state["wz_step"] = 4
                st.rerun()
    elif step == 4:
        st.markdown("#### Voto (1–5)")
        cols = st.columns(5)
        for v in range(1, 6):
            if cols[v - 1].button(str(v), key=f"pv{v}"):
                shot["Rating"] = v
                st.session_state["wz_step"] = 5
                st.rerun()
    elif step == 5:
        st.markdown("#### Reazione mentale")
        for opt in MENTAL_OPTIONS:
            if st.button(opt, key=f"pmn{opt}", use_container_width=True):
                shot["Mental_Reaction"] = opt
                st.session_state["wz_step"] = 6
                st.rerun()
    elif step == 6:
        st.markdown("#### Salva putt (strokes gained dal primo putt)")
        if st.button("Calcola SG e salva", type="primary", use_container_width=True):
            sg = compute_sg_putt(float(shot["Start_Dist_m"]), float(shot["End_Dist_m"]))
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
                "Start_Dist_m": shot.get("Start_Dist_m", np.nan),
                "End_Dist_m": shot.get("End_Dist_m", np.nan),
                "Hole_Dist_Start_m": shot.get("Start_Dist_m", np.nan),
                "Hole_Dist_End_m": shot.get("End_Dist_m", np.nan),
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
        "Seleziona settore e periodo per aprire una dashboard completa con grafici, SG e tabelle.",
        ["Pie charts", "Dispersione", "Strokes Gained", "Trend", "Putting make%"],
    )
    st.markdown("### Review — statistiche")
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
    st.caption(
        f"Utente **{user}** · periodo **{period}** · settore **{CATEGORIES[sector]}** · "
        f"n = **{len(dsec)}** colpi."
    )

    if dsec.empty:
        st.info("Nessun colpo in questo filtro.")
        brand_footer()
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Colpi registrati", len(dsec))
    rmean = pd.to_numeric(dsec["Rating"], errors="coerce").mean()
    m2.metric("Voto medio", f"{rmean:.2f}" if pd.notna(rmean) else "—")
    sg_series = pd.to_numeric(dsec["Strokes_Gained"], errors="coerce").dropna()
    m3.metric("SG medio", f"{sg_series.mean():+.3f}" if len(sg_series) else "—")

    sg_summary_table(df_f, sector)
    trend_panel(dsec, CATEGORIES[sector])
    club_breakdown_table(dsec)

    if sector == "RANGE":
        plot_pie(
            dsec,
            "Impact",
            "Tipologia di impatto — percentuali",
            "Legenda: ripartizione percentuale degli impatti dichiarati.",
        )
        plot_pie(
            dsec,
            "Curvature",
            "Curvatura — percentuali",
            "Legenda: forma di volo predominante nel campione.",
        )
        plot_pie(
            dsec,
            "Direction_LR",
            "Tendenza direzionale vs bersaglio",
            "Legenda: orientamento medio rispetto alla linea di punteria.",
        )
        plot_dispersion(dsec, "Dispersione dall’alto — RANGE")
        satisfaction_breakdown(df_f, "RANGE")

    elif sector == "SHORT":
        plot_pie(dsec, "Lie_Start", "Lie iniziale", "Legenda: da dove parte la palla più spesso.")
        plot_pie(dsec, "Lie_End", "Lie finale", "Legenda: dove finisce la palla dopo il colpo.")
        plot_pie(dsec, "Impact", "Impatto", "Legenda: qualità di contatto dichiarata.")
        plot_pie(dsec, "Direction_LR", "Linea vs buca", "Legenda: tendenza destra/sinistra.")
        plot_dispersion(dsec, "Dispersione dall’alto — gioco corto")
        satisfaction_breakdown(df_f, "SHORT")

    else:
        plot_pie(dsec, "Impact", "Impatto sulla faccia", "Legenda: zona di contatto sul putter.")
        plot_pie(dsec, "Trajectory", "Traiettoria di rotazione", "Legenda: pull/dritta/push.")
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

    if not st.session_state["logged_in"]:
        login_screen()
        return

    user = str(st.session_state["user"])

    with st.sidebar:
        brand_header("Profilo")
        st.write(f"**Atleta:** {user}")
        st.markdown("### Sezione")
        page = st.selectbox(
            "Apri sezione",
            ["Inserimento dati", "Review"],
            index=0,
            key="main_page_sidebar",
            label_visibility="collapsed",
        )
        session_name = st.text_input("Nome sessione / note", value="Sessione Allenamento")
        st.divider()
        if st.button("Logout / cambia utente", use_container_width=True):
            st.session_state["logged_in"] = False
            st.session_state.pop("user", None)
            st.rerun()

    st.markdown("### Navigazione")
    page_top = st.radio(
        "Vai a",
        ["Inserimento dati", "Review"],
        horizontal=True,
        index=0 if page == "Inserimento dati" else 1,
        key="main_page_top",
        label_visibility="collapsed",
    )
    page = page_top

    if page == "Inserimento dati":
        brand_header("Inserimento rapido")
        render_hero(
            "Sessione di raccolta dati",
            "Input veloce a step singoli con pulsanti grandi, pensato per utilizzo smartphone sul campo pratica.",
            ["Range", "Short game", "Putting"],
        )
        st.session_state.setdefault("wz_cat", None)
        if st.session_state["wz_cat"] is None:
            st.markdown("#### Scegli il settore")
            c1, c2, c3 = st.columns(3)
            if c1.button("Range\n(gioco lungo)", use_container_width=True):
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
            if st.button("Torna alla scelta settore"):
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

    else:
        brand_header()
        review_panel(user, session_name)


if __name__ == "__main__":
    main()

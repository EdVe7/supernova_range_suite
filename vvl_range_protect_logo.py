import datetime
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_gsheets import GSheetsConnection

# ==============================================================================
# 1. CONFIGURAZIONE E STILI
# ==============================================================================
st.set_page_config(page_title="Supernova Sport Science", page_icon="⛳", layout="wide")

COLORS = {
    
    "Gold": "#DAA520",
    "White": "#FFFFFF",
    "Grey": "#F3F4F6",
}

st.markdown(
    f"""
<style>
    #MainMenu {{visibility: hidden; display: none;}}
    header {{visibility: hidden; display: none;}}
    footer {{visibility: hidden; display: none;}}
    [data-testid="stToolbar"] {{visibility: hidden; display: none;}}
    [data-testid="stDecoration"] {{visibility: hidden; display: none;}}
    .block-container {{ padding-top: 1rem; }}
    .stApp {{ background-color: {COLORS['White']}; color: #1f2937; }}
    h1, h2, h3 {{ font-family: 'Helvetica', sans-serif; color: {COLORS['BrandTeal']}; }}
    .stButton>button {{ background-color: {COLORS['BrandTeal']}; color: white; border-radius: 10px; font-weight: bold; width: 100%; border: none; padding: 16px 12px; font-size: 1.05rem; min-height: 52px; }}
    .stButton>button:hover {{ background-color: {COLORS['DarkTeal']}; }}
    [data-testid="stFormSubmitButton"] button {{ min-height: 56px; font-size: 1.2rem; border-radius: 10px; }}
    .metric-box {{ background: {COLORS['Grey']}; border-left: 4px solid {COLORS['Gold']}; border-radius: 4px; padding: 15px; text-align: center; }}
    .metric-title {{ font-size: 0.85rem; color: #6b7280; text-transform: uppercase; font-weight: bold; }}
    .metric-value {{ font-size: 1.8rem; color: {COLORS['BrandTeal']}; font-weight: 800; }}
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. COSTANTI DATI
# ==============================================================================
COLUMNS = [
    "User",
    "Date",
    "SessionName",
    "Time",
    "Category",
    "Club",
    "Start_Dist",
    "Lie",
    "Impact",
    "Curvature",
    "Height",
    "Direction",
    "Proximity",
    "Rating",
    "SG_Dist_Pin_Pre_m",
    "SG_Lie_Pre",
    "SG_Dist_Pin_Post_m",
    "SG_Lie_Post",
    "Strokes_Gained",
]

CATEGORIES = ["LONG GAME / RANGE", "SHORT GAME", "PUTTING"]
CLUBS = ["DR", "3W", "5W", "7W", "3H", "3i", "4i", "5i", "6i", "7i", "8i", "9i", "PW", "AW", "GW", "SW", "LW"]
SHORT_GAME_CLUBS = ["LW", "SW", "GW", "AW", "PW", "9i", "8i"]

SG_LIE_OPTIONS = ["Tee", "Fairway", "Rough", "Bunker", "Fringe", "Green"]


def _m_to_yd(m):
    return float(m) * 1.09361


def _expected_putts_m(dist_m):
    if dist_m is None or (isinstance(dist_m, float) and np.isnan(dist_m)):
        return np.nan
    d = max(0.0, float(dist_m))
    if d <= 0.05:
        return 0.0
    xs = np.array([0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0, 15.0], dtype=float)
    ys = np.array([1.04, 1.08, 1.13, 1.23, 1.34, 1.42, 1.50, 1.56, 1.61, 1.65, 1.72, 1.85], dtype=float)
    d = min(d, 15.0)
    return float(np.interp(d, xs, ys))


def _expected_non_green_approach_m(dist_m, lie):
    if dist_m is None or (isinstance(dist_m, float) and np.isnan(dist_m)):
        return np.nan
    yd = _m_to_yd(max(1.0, float(dist_m)))
    yd = min(yd, 330.0)
    xs = np.array([30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 225, 250, 275, 300], dtype=float)
    ys = np.array([2.06, 2.14, 2.21, 2.29, 2.36, 2.41, 2.45, 2.50, 2.65, 2.80, 2.95, 3.10, 3.25, 3.40, 3.55, 3.70], dtype=float)
    base = float(np.interp(yd, xs, ys))
    if lie == "Rough":
        base += 0.22
    elif lie == "Bunker":
        base += 0.55
    elif lie == "Tee":
        base += 0.0
    elif lie == "Fringe":
        base += 0.08
    return base


def expected_strokes_from_state(dist_m, lie):
    if lie == "Green":
        return _expected_putts_m(dist_m)
    if lie == "Fringe":
        return _expected_putts_m(dist_m) + 0.12
    return _expected_non_green_approach_m(dist_m, lie)


def compute_strokes_gained(pre_m, lie_pre, post_m, lie_post):
    if pre_m is None or post_m is None or lie_pre is None or lie_post is None:
        return np.nan
    try:
        pre_m = float(pre_m)
        post_m = float(post_m)
    except (TypeError, ValueError):
        return np.nan
    e_pre = expected_strokes_from_state(pre_m, lie_pre)
    if post_m <= 0.03 and lie_post == "Green":
        e_post = 0.0
    else:
        e_post = expected_strokes_from_state(post_m, lie_post)
    if np.isnan(e_pre) or np.isnan(e_post):
        return np.nan
    return float(e_pre - 1.0 - e_post)


# ==============================================================================
# 3. SPLASH SCREEN & LOGIN
# ==============================================================================
if "splash_done" not in st.session_state:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            try:
                st.image("logo.png", use_container_width=True)
                st.markdown(
                    f"<p style='text-align:center; font-style:italic; color:{COLORS['DarkTeal']}; font-weight:bold;'>Data over talent</p>",
                    unsafe_allow_html=True,
                )
            except Exception:
                st.markdown(
                    f"<h1 style='text-align:center; font-size: 5rem; color:{COLORS['BrandTeal']};'>SUPERNOVA</h1><p style='text-align:center;'>SPORT SCIENCE SOLUTIONS</p><p style='text-align:center; font-style:italic; color:{COLORS['DarkTeal']}; font-weight:bold;'>Data over talent</p>",
                    unsafe_allow_html=True,
                )
    time.sleep(2.0)
    placeholder.empty()
    st.session_state["splash_done"] = True

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image("logo.png", width=200)
            st.markdown(
                f"<p style='text-align:center; font-style:italic; color:{COLORS['DarkTeal']}; font-weight:bold;'>Data over talent</p>",
                unsafe_allow_html=True,
            )
        except Exception:
            pass
        st.markdown("### Accesso Piattaforma Pro")
        with st.expander("Termini di Servizio e Privacy (lettura consigliata)", expanded=False):
            st.markdown(
                """
Utilizzando questa applicazione confermi di aver compreso che:

1. I dati inseriti (inclusi identificativo atleta e metriche di allenamento) vengono trattati per finalita operative e di analisi della performance sportiva.
2. Sei responsabile della veridicita dei dati inseriti e dell'uso conforme della piattaforma.
3. Il calcolo **Strokes Gained** qui mostrato usa **benchmark semplificati** (modello educativo / allenamento) e **non** sostituisce dataset ufficiali Tour completi o software commerciali certificati.
4. Per richieste su privacy, conservazione o cancellazione dati, contatta il titolare del trattamento / staff Supernova.

Cliccando su "Accetto" sotto dichiari di accettare questi termini e di proseguire.
"""
            )
        tos_ok = st.checkbox("Ho letto e accetto Termini di Servizio e Informativa Privacy", value=False, key="tos_accept")
        user_input = st.text_input("ID Atleta (Nome)").upper().strip()
        pass_input = st.text_input("Master Password", type="password")
        if st.button("AUTENTICAZIONE"):
            if not tos_ok:
                st.error("Devi accettare i Termini di Servizio e la Privacy per accedere.")
            elif pass_input == "supernova.analytics" and user_input != "":
                st.session_state["logged_in"] = True
                st.session_state["user"] = user_input
                st.session_state["tos_accepted"] = True
                st.rerun()
            else:
                st.error("Credenziali respinte.")
    st.stop()


# ==============================================================================
# 4. DATA ENGINE (Google Sheets) - INVARIATO
# ==============================================================================
@st.cache_data(ttl=5)
def load_data():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl=0)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
        df["Proximity"] = pd.to_numeric(df["Proximity"], errors="coerce")
        df["Start_Dist"] = pd.to_numeric(df["Start_Dist"], errors="coerce")
        for c in ("SG_Dist_Pin_Pre_m", "SG_Dist_Pin_Post_m", "Strokes_Gained"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = np.nan
        return df
    except Exception:
        return pd.DataFrame(columns=COLUMNS)


def save_shot(shot_data):
    conn = st.connection("gsheets", type=GSheetsConnection)
    df_existing = load_data()
    df_new = pd.DataFrame([shot_data])
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
    conn.update(data=df_final)
    st.cache_data.clear()


def calc_lateral(row):
    if row.get("Direction") == "Dx":
        return row.get("Proximity", 0.0)
    if row.get("Direction") == "Sx":
        return -row.get("Proximity", 0.0)
    return 0.0


def safe_mean(series, default=0.0):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else default


def safe_std(series, default=0.0):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.std()) if len(s) > 1 else default


def normalize_0_100(value, min_v, max_v):
    if max_v <= min_v:
        return 0.0
    v = max(min(value, max_v), min_v)
    return ((v - min_v) / (max_v - min_v)) * 100.0


def compute_kpis(df_area):
    if df_area.empty:
        return {
            "shots": 0,
            "avg_rating": 0.0,
            "elite_rate": 0.0,
            "avg_proximity": 0.0,
            "prox_std": 0.0,
            "lat_std": 0.0,
            "execution_index": 0.0,
            "consistency_index": 0.0,
            "sample_quality": "LOW",
        }

    shots = len(df_area)
    avg_rating = safe_mean(df_area["Rating"])
    elite_rate = (len(df_area[df_area["Rating"] == 3]) / shots) * 100.0
    avg_prox = safe_mean(df_area["Proximity"])
    prox_std = safe_std(df_area["Proximity"])
    lat_std = safe_std(df_area.apply(calc_lateral, axis=1))

    exec_rating_component = normalize_0_100(avg_rating, 1.0, 3.0)
    execution_index = (0.55 * exec_rating_component) + (0.45 * elite_rate)

    prox_component = 100.0 - normalize_0_100(prox_std, 0.0, 12.0)
    lat_component = 100.0 - normalize_0_100(lat_std, 0.0, 15.0)
    consistency = max(0.0, (0.6 * prox_component) + (0.4 * lat_component))

    sample_quality = "LOW" if shots < 25 else "MEDIUM" if shots < 80 else "HIGH"
    return {
        "shots": shots,
        "avg_rating": avg_rating,
        "elite_rate": elite_rate,
        "avg_proximity": avg_prox,
        "prox_std": prox_std,
        "lat_std": lat_std,
        "execution_index": execution_index,
        "consistency_index": consistency,
        "sample_quality": sample_quality,
    }


def _last_defaults():
    return st.session_state.get("last_shot") or {}


def _idx_in(lst, val, default_idx=0):
    try:
        return lst.index(val) if val in lst else default_idx
    except ValueError:
        return default_idx


# ==============================================================================
# 5. UI PRINCIPALE
# ==============================================================================
st.sidebar.markdown(f"### Atleta: {st.session_state['user']}")
session_name = st.sidebar.text_input("Sessione / Note", "Test Valutazione")

tab_input, tab_review = st.tabs(["REGISTRO COLPI", "ANALYTICS REVIEW"])

with tab_input:
    st.markdown("### Inserimento rapido colpo")
    st.caption("Workflow campo pratica: seleziona area, compila pochi campi, salva subito.")

    ld = _last_defaults()
    r1, r2, r3 = st.columns(3)
    with r1:
        if st.button("LONG GAME / RANGE", use_container_width=True, key="quick_long"):
            st.session_state["quick_cat"] = "LONG GAME / RANGE"
            st.rerun()
    with r2:
        if st.button("SHORT GAME", use_container_width=True, key="quick_short"):
            st.session_state["quick_cat"] = "SHORT GAME"
            st.rerun()
    with r3:
        if st.button("PUTTING", use_container_width=True, key="quick_putt"):
            st.session_state["quick_cat"] = "PUTTING"
            st.rerun()

    if st.button("RIPETI ULTIMO COLPO (stessi campi, nuovo orario)", use_container_width=True, key="repeat_last"):
        prev = st.session_state.get("last_shot")
        if not prev:
            st.warning("Nessun colpo precedente da ripetere.")
        else:
            shot_repeat = {
                **prev,
                "User": st.session_state["user"],
                "Date": datetime.date.today(),
                "SessionName": session_name,
                "Time": datetime.datetime.now().strftime("%H:%M"),
            }
            save_shot(shot_repeat)
            st.session_state["last_shot"] = shot_repeat
            st.success("Ultimo colpo duplicato e salvato.")
            st.rerun()

    qc = st.session_state.get("quick_cat")
    if qc in CATEGORIES:
        del st.session_state["quick_cat"]
        ix_cat = _idx_in(CATEGORIES, qc, 0)
    else:
        ix_cat = _idx_in(CATEGORIES, ld.get("Category"), 0)
    cat_scelta = st.radio("Area Tecnica", CATEGORIES, horizontal=True, index=ix_cat, key="area_radio")

    with st.form("form_dati", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        start_dist = 0.0
        lie = "-"
        height = "-"
        direction = "Dritta"

        if cat_scelta == "LONG GAME / RANGE":
            with col1:
                club = st.selectbox("Bastone", CLUBS, index=_idx_in(CLUBS, ld.get("Club"), 0))
                impact = st.selectbox(
                    "Impatto",
                    ["Solido", "Punta", "Tacco", "Shank", "Flappa", "Top"],
                    index=_idx_in(["Solido", "Punta", "Tacco", "Shank", "Flappa", "Top"], ld.get("Impact"), 0),
                )
            with col2:
                curvature = st.selectbox(
                    "Curvatura (Effetto)",
                    ["Dritta", "Push", "Pull", "Slice", "Hook"],
                    index=_idx_in(["Dritta", "Push", "Pull", "Slice", "Hook"], ld.get("Curvature"), 0),
                )
                height = st.selectbox(
                    "Altezza",
                    ["Giusta", "Alta", "Bassa", "Rasoterra", "Flappa"],
                    index=_idx_in(["Giusta", "Alta", "Bassa", "Rasoterra", "Flappa"], ld.get("Height"), 0),
                )
            with col3:
                direction = st.selectbox(
                    "Direzione vs Target",
                    ["Dritta", "Dx", "Sx"],
                    index=_idx_in(["Dritta", "Dx", "Sx"], ld.get("Direction"), 0),
                )
                proximity = st.number_input(
                    "Proximity Target (metri)",
                    min_value=0.0,
                    step=1.0,
                    value=float(ld.get("Proximity", 0.0) or 0.0),
                )
            voto = st.slider("Voto Esecuzione (1-3)", 1, 3, int(ld.get("Rating", 2) or 2))

        elif cat_scelta == "SHORT GAME":
            with col1:
                club = st.selectbox("Bastone", SHORT_GAME_CLUBS, index=_idx_in(SHORT_GAME_CLUBS, ld.get("Club"), 0))
                start_dist = st.number_input(
                    "Distanza di Partenza (metri)",
                    min_value=1.0,
                    step=1.0,
                    value=float(ld.get("Start_Dist", 10.0) or 10.0),
                )
            with col2:
                lie = st.selectbox(
                    "Lie",
                    ["Fairway", "Rough", "Bunker", "Sponda"],
                    index=_idx_in(["Fairway", "Rough", "Bunker", "Sponda"], ld.get("Lie"), 0),
                )
                impact = st.selectbox(
                    "Impatto",
                    ["Solido", "Punta", "Tacco", "Shank", "Flappa", "Top"],
                    index=_idx_in(["Solido", "Punta", "Tacco", "Shank", "Flappa", "Top"], ld.get("Impact"), 0),
                )
            with col3:
                curvature = st.selectbox(
                    "Curvatura Volo",
                    ["Dritta", "Push", "Pull", "Slice", "Hook"],
                    index=_idx_in(["Dritta", "Push", "Pull", "Slice", "Hook"], ld.get("Curvature"), 0),
                )
                height = st.selectbox(
                    "Altezza",
                    ["Giusta", "Alta", "Bassa", "Rasoterra", "Flappa"],
                    index=_idx_in(["Giusta", "Alta", "Bassa", "Rasoterra", "Flappa"], ld.get("Height"), 0),
                )
            bot1, bot2 = st.columns(2)
            direction = bot1.selectbox(
                "Direzione vs Target",
                ["Dritta", "Dx", "Sx"],
                index=_idx_in(["Dritta", "Dx", "Sx"], ld.get("Direction"), 0),
            )
            proximity = bot2.number_input(
                "Proximity Finale (metri)",
                min_value=0.0,
                step=0.5,
                value=float(ld.get("Proximity", 2.0) or 2.0),
            )
            voto = st.slider("Voto Esecuzione (1-3)", 1, 3, int(ld.get("Rating", 2) or 2))

        else:
            club = "Putter"
            with col1:
                start_dist = st.number_input(
                    "Distanza dal Buco (metri)",
                    min_value=0.5,
                    step=0.5,
                    value=float(ld.get("Start_Dist", 3.0) or 3.0),
                )
            with col2:
                impact = st.selectbox(
                    "Impatto sulla Faccia",
                    ["Centro", "Punta", "Tacco"],
                    index=_idx_in(["Centro", "Punta", "Tacco"], ld.get("Impact"), 0),
                )
            with col3:
                curvature = st.selectbox(
                    "Traiettoria / Linea",
                    ["Dritta", "Push", "Pull"],
                    index=_idx_in(["Dritta", "Push", "Pull"], ld.get("Curvature"), 0),
                )
            proximity = st.number_input(
                "Proximity (distanza residua metri)",
                min_value=0.0,
                step=0.1,
                value=float(ld.get("Proximity", 0.5) or 0.5),
            )
            voto = st.slider("Voto (3=Perfetto, 2=Data, 1=Errore)", 1, 3, int(ld.get("Rating", 2) or 2))

        st.markdown("##### Strokes Gained (dati per calcolo — benchmark semplificato Tour-style)")
        st.caption(
            "Inserisci distanza al bersaglio/buco **prima** e **dopo** il colpo, e la **lie** coerente. "
            "Per putting: pre = primo putt, post = residuo (0 se buca). "
            "Sul Google Sheet aggiungi una tantum le colonne: SG_Dist_Pin_Pre_m, SG_Lie_Pre, SG_Dist_Pin_Post_m, SG_Lie_Post, Strokes_Gained."
        )
        sync_sg = st.checkbox(
            "Compila SG automaticamente dai campi colpo (dove applicabile)",
            value=True,
            key="sg_sync",
        )

        def _map_lie_short_to_sg(lie_val):
            m = {"Fairway": "Fairway", "Rough": "Rough", "Bunker": "Bunker", "Sponda": "Rough"}
            return m.get(lie_val, "Fairway")

        if cat_scelta == "LONG GAME / RANGE":
            default_pre = float(ld.get("SG_Dist_Pin_Pre_m", 180) or 180)
            default_post = float(ld.get("SG_Dist_Pin_Post_m", 50) or 50)
            default_lpre = ld.get("SG_Lie_Pre", "Tee") or "Tee"
            default_lpost = ld.get("SG_Lie_Post", "Fairway") or "Fairway"
        elif cat_scelta == "SHORT GAME":
            default_pre = float(ld.get("SG_Dist_Pin_Pre_m", ld.get("Start_Dist", 20)) or 20)
            default_post = float(ld.get("SG_Dist_Pin_Post_m", ld.get("Proximity", 3)) or 3)
            default_lpre = ld.get("SG_Lie_Pre") or _map_lie_short_to_sg(lie if lie != "-" else "Fairway")
            default_lpost = ld.get("SG_Lie_Post", "Green") or "Green"
        else:
            default_pre = float(ld.get("SG_Dist_Pin_Pre_m", ld.get("Start_Dist", 3)) or 3)
            default_post = float(ld.get("SG_Dist_Pin_Post_m", ld.get("Proximity", 0)) or 0)
            default_lpre = "Green"
            default_lpost = "Green"

        sg_c1, sg_c2 = st.columns(2)
        with sg_c1:
            sg_pre_m = st.number_input(
                "Distanza al buco PRIMA del colpo (m)",
                min_value=0.0,
                step=1.0,
                value=default_pre,
                key="sg_pre",
            )
            sg_lie_pre = st.selectbox(
                "Lie PRIMA del colpo",
                SG_LIE_OPTIONS,
                index=_idx_in(SG_LIE_OPTIONS, default_lpre, 1),
                key="sg_lie_pre",
            )
        with sg_c2:
            sg_post_m = st.number_input(
                "Distanza al buco DOPO il colpo (m, 0 = in buca)",
                min_value=0.0,
                step=0.5,
                value=default_post,
                key="sg_post",
            )
            sg_lie_post = st.selectbox(
                "Lie DOPO il colpo",
                SG_LIE_OPTIONS,
                index=_idx_in(SG_LIE_OPTIONS, default_lpost, 5),
                key="sg_lie_post",
            )

        if sync_sg:
            if cat_scelta == "SHORT GAME":
                sg_pre_m = float(start_dist)
                sg_post_m = float(proximity)
                sg_lie_pre = _map_lie_short_to_sg(lie)
                sg_lie_post = st.session_state.get("sg_lie_post", "Green")
            elif cat_scelta == "PUTTING":
                sg_pre_m = float(start_dist)
                sg_post_m = float(proximity)
                sg_lie_pre = "Green"
                sg_lie_post = "Green"

        sg_preview = compute_strokes_gained(sg_pre_m, sg_lie_pre, sg_post_m, sg_lie_post)
        if not np.isnan(sg_preview):
            st.info(f"Strokes Gained stimato (colpo): **{sg_preview:+.3f}**")
        else:
            st.warning("Completa i campi SG per vedere la stima.")

        submitted = st.form_submit_button("SALVA COLPO", use_container_width=True)
        if submitted:
            sg_val = compute_strokes_gained(sg_pre_m, sg_lie_pre, sg_post_m, sg_lie_post)
            shot = {
                "User": st.session_state["user"],
                "Date": datetime.date.today(),
                "SessionName": session_name,
                "Time": datetime.datetime.now().strftime("%H:%M"),
                "Category": cat_scelta,
                "Club": club,
                "Start_Dist": start_dist,
                "Lie": lie,
                "Impact": impact,
                "Curvature": curvature,
                "Height": height,
                "Direction": direction,
                "Proximity": proximity,
                "Rating": voto,
                "SG_Dist_Pin_Pre_m": sg_pre_m,
                "SG_Lie_Pre": sg_lie_pre,
                "SG_Dist_Pin_Post_m": sg_post_m,
                "SG_Lie_Post": sg_lie_post,
                "Strokes_Gained": sg_val,
            }
            save_shot(shot)
            st.session_state["last_shot"] = shot
            st.success("Colpo registrato correttamente.")


with tab_review:
    st.markdown("### Performance Review")
    df_all = load_data()
    df_user = df_all[df_all["User"] == st.session_state["user"]].copy()

    if df_user.empty:
        st.info("Nessun dato disponibile per l'atleta corrente.")
    else:
        colf1, colf2, colf3 = st.columns([1.2, 1.2, 1.6])
        with colf1:
            periodo = st.selectbox("Filtro Temporale", ["Sessione Attuale", "Ultimi 7 Giorni", "Ultimi 30 Giorni", "Tutti i Dati"])
        with colf2:
            area_sel = st.selectbox("Area", ["TUTTE"] + CATEGORIES)
        with colf3:
            club_filter = st.multiselect("Filtro Bastoni (opzionale)", sorted(df_user["Club"].dropna().astype(str).unique().tolist()))

        oggi = datetime.date.today()
        if periodo == "Sessione Attuale":
            df_f = df_user[df_user["SessionName"] == session_name]
        elif periodo == "Ultimi 7 Giorni":
            df_f = df_user[df_user["Date"] >= (oggi - datetime.timedelta(days=7))]
        elif periodo == "Ultimi 30 Giorni":
            df_f = df_user[df_user["Date"] >= (oggi - datetime.timedelta(days=30))]
        else:
            df_f = df_user

        if area_sel != "TUTTE":
            df_f = df_f[df_f["Category"] == area_sel]
        if club_filter:
            df_f = df_f[df_f["Club"].isin(club_filter)]

        if df_f.empty:
            st.warning("Nessun dato nel filtro selezionato.")
        else:
            df_f = df_f.copy()
            df_f["DateTime"] = pd.to_datetime(
                df_f["Date"].astype(str) + " " + df_f["Time"].astype(str),
                errors="coerce",
            )
            df_f["Lateral_Error"] = df_f.apply(calc_lateral, axis=1)
            df_f["ShotSeq"] = np.arange(1, len(df_f) + 1)

            kpi = compute_kpis(df_f)
            sg_series = pd.to_numeric(df_f.get("Strokes_Gained", pd.Series(dtype=float)), errors="coerce").dropna()
            sg_mean = float(sg_series.mean()) if len(sg_series) else None

            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Shot Totali", f"{kpi['shots']}")
            k2.metric("Rating Medio", f"{kpi['avg_rating']:.2f}")
            k3.metric("Elite Rate (3/3)", f"{kpi['elite_rate']:.1f}%")
            k4.metric("Execution Index", f"{kpi['execution_index']:.1f}/100")
            k5.metric("Consistency Index", f"{kpi['consistency_index']:.1f}/100")
            k6.metric("SG medio (stimato)", f"{sg_mean:+.3f}" if sg_mean is not None else "N/D")

            st.caption(f"Qualita campione: {kpi['sample_quality']} | Proximity media: {kpi['avg_proximity']:.2f} m")
            st.divider()

            g1, g2 = st.columns(2)
            with g1:
                fig_imp = px.pie(
                    df_f,
                    names="Impact",
                    title="Distribuzione Impatti",
                    hole=0.35,
                    color_discrete_sequence=px.colors.sequential.Teal,
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            with g2:
                fig_curv = px.pie(
                    df_f,
                    names="Curvature",
                    title="Distribuzione Traiettorie",
                    hole=0.35,
                    color_discrete_sequence=px.colors.sequential.Teal,
                )
                st.plotly_chart(fig_curv, use_container_width=True)

            g3, g4 = st.columns(2)
            with g3:
                rating_mix = (
                    df_f["Rating"]
                    .value_counts(dropna=True)
                    .rename_axis("Rating")
                    .reset_index(name="Count")
                    .sort_values("Rating")
                )
                if not rating_mix.empty:
                    fig_rating = px.bar(
                        rating_mix,
                        x="Rating",
                        y="Count",
                        title="Distribuzione Rating (1-2-3)",
                        text="Count",
                        color="Rating",
                        color_discrete_sequence=px.colors.sequential.Teal,
                    )
                    st.plotly_chart(fig_rating, use_container_width=True)
            with g4:
                dir_mix = (
                    df_f["Direction"]
                    .fillna("N/D")
                    .value_counts()
                    .rename_axis("Direction")
                    .reset_index(name="Count")
                )
                if not dir_mix.empty:
                    fig_dir = px.bar(
                        dir_mix,
                        x="Direction",
                        y="Count",
                        title="Direzione vs Target",
                        text="Count",
                        color="Direction",
                        color_discrete_sequence=px.colors.sequential.Teal,
                    )
                    st.plotly_chart(fig_dir, use_container_width=True)

            if area_sel in ["TUTTE", "LONG GAME / RANGE", "SHORT GAME"]:
                fig_scatter = px.scatter(
                    df_f,
                    x="Lateral_Error",
                    y="Club",
                    color="Club",
                    title="Mappa Dispersione Laterale",
                    labels={"Lateral_Error": "Sx <-- 0 --> Dx"},
                    hover_data=["Date", "Category", "Proximity", "Rating", "SessionName"],
                )
                fig_scatter.add_vline(x=0, line_dash="dash", line_color=COLORS["Gold"])
                st.plotly_chart(fig_scatter, use_container_width=True)

            dft = df_f.copy()
            dft["Date"] = pd.to_datetime(dft["Date"], errors="coerce")
            trend = dft.dropna(subset=["Date"]).groupby("Date", as_index=False).agg(
                Rating=("Rating", "mean"),
                Proximity=("Proximity", "mean"),
            ).sort_values("Date")
            if not trend.empty:
                fig_trend = px.line(
                    trend,
                    x="Date",
                    y=["Rating", "Proximity"],
                    markers=True,
                    title="Trend Rating e Proximity",
                    color_discrete_sequence=[COLORS["BrandTeal"], COLORS["Gold"]],
                )
                st.plotly_chart(fig_trend, use_container_width=True)

            rolling_src = df_f.sort_values("DateTime").copy()
            rolling_src["Rating_Rolling_20"] = rolling_src["Rating"].rolling(20, min_periods=5).mean()
            rolling_src["Proximity_Rolling_20"] = rolling_src["Proximity"].rolling(20, min_periods=5).mean()
            rolling_src = rolling_src.dropna(subset=["Rating_Rolling_20", "Proximity_Rolling_20"])
            if not rolling_src.empty:
                fig_roll = px.line(
                    rolling_src,
                    x="ShotSeq",
                    y=["Rating_Rolling_20", "Proximity_Rolling_20"],
                    title="Trend Rolling (ultimi 20 colpi)",
                    color_discrete_sequence=[COLORS["BrandTeal"], COLORS["Gold"]],
                )
                fig_roll.update_layout(xaxis_title="Sequenza colpi", yaxis_title="Media mobile")
                st.plotly_chart(fig_roll, use_container_width=True)

            dclub = df_f.copy()
            dclub["Rating"] = pd.to_numeric(dclub["Rating"], errors="coerce")
            dclub["Proximity"] = pd.to_numeric(dclub["Proximity"], errors="coerce")
            club_perf = dclub.groupby("Club", as_index=False).agg(Rating=("Rating", "mean"), Proximity=("Proximity", "mean"), Volume=("Club", "count"))
            club_perf = club_perf[club_perf["Volume"] >= 3]
            if not club_perf.empty:
                fig_club = px.scatter(
                    club_perf.sort_values("Rating", ascending=False),
                    x="Proximity",
                    y="Rating",
                    size="Volume",
                    color="Club",
                    title="Club Efficiency Map",
                    hover_data=["Volume"],
                    color_discrete_sequence=px.colors.sequential.Teal,
                )
                fig_club.update_layout(xaxis_title="Proximity media (m)", yaxis_title="Rating medio")
                st.plotly_chart(fig_club, use_container_width=True)

            heat_src = df_f.copy()
            heat_src["Lie"] = heat_src["Lie"].fillna("-")
            heat_src["Impact"] = heat_src["Impact"].fillna("-")
            heat = pd.pivot_table(
                heat_src,
                index="Lie",
                columns="Impact",
                values="Rating",
                aggfunc="mean",
            )
            if not heat.empty:
                fig_heat = px.imshow(
                    heat,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="Teal",
                    title="Heatmap Qualita Esecuzione (Lie x Impatto)",
                )
                st.plotly_chart(fig_heat, use_container_width=True)

            df_sg = df_f.copy()
            df_sg["Strokes_Gained"] = pd.to_numeric(df_sg["Strokes_Gained"], errors="coerce")
            df_sg_ok = df_sg.dropna(subset=["Strokes_Gained"])
            if not df_sg_ok.empty:
                st.markdown("#### Strokes Gained (review)")
                sg_sorted = df_sg_ok.sort_values("DateTime")
                fig_sg = px.bar(
                    sg_sorted,
                    x="ShotSeq",
                    y="Strokes_Gained",
                    color="Category",
                    title="SG per colpo (sequenza filtro attivo)",
                    color_discrete_sequence=px.colors.sequential.Teal,
                )
                fig_sg.add_hline(y=0, line_dash="dash", line_color=COLORS["Gold"])
                st.plotly_chart(fig_sg, use_container_width=True)
                sg_by_cat = (
                    df_sg_ok.groupby("Category", as_index=False)["Strokes_Gained"]
                    .mean()
                    .rename(columns={"Strokes_Gained": "SG_Medio"})
                )
                if not sg_by_cat.empty:
                    fig_sg_cat = px.bar(
                        sg_by_cat,
                        x="Category",
                        y="SG_Medio",
                        text="SG_Medio",
                        title="SG medio per area (filtro attivo)",
                        color="Category",
                        color_discrete_sequence=px.colors.sequential.Teal,
                    )
                    fig_sg_cat.update_traces(texttemplate="%{y:+.3f}", textposition="outside")
                    st.plotly_chart(fig_sg_cat, use_container_width=True)

            st.markdown("#### Tabelle Pro")
            t1, t2 = st.columns(2)
            with t1:
                club_table = (
                    df_f.groupby("Club", as_index=False)
                    .agg(
                        Shots=("Club", "count"),
                        Rating_Medio=("Rating", "mean"),
                        Proximity_Media=("Proximity", "mean"),
                        Proximity_STD=("Proximity", "std"),
                    )
                    .fillna(0.0)
                )
                if not club_table.empty:
                    club_table["Performance_Score"] = (
                        (club_table["Rating_Medio"] / 3.0) * 70
                        + (1 - club_table["Proximity_Media"] / (club_table["Proximity_Media"].max() + 1e-9)) * 30
                    )
                    club_table = club_table.sort_values("Performance_Score", ascending=False).round(2)
                    st.dataframe(club_table, use_container_width=True, height=280)
            with t2:
                session_table = (
                    df_f.groupby("SessionName", as_index=False)
                    .agg(
                        Shots=("SessionName", "count"),
                        Rating_Medio=("Rating", "mean"),
                        Elite_Rate=("Rating", lambda s: (s == 3).mean() * 100),
                        Proximity_Media=("Proximity", "mean"),
                    )
                    .sort_values("Shots", ascending=False)
                    .round(2)
                )
                if not session_table.empty:
                    st.dataframe(session_table, use_container_width=True, height=280)

            if not df_sg_ok.empty:
                st.markdown("#### Tabella SG (dettaglio)")
                st.dataframe(
                    df_sg_ok[
                        [
                            "Date",
                            "Time",
                            "SessionName",
                            "Category",
                            "Club",
                            "SG_Dist_Pin_Pre_m",
                            "SG_Lie_Pre",
                            "SG_Dist_Pin_Post_m",
                            "SG_Lie_Post",
                            "Strokes_Gained",
                        ]
                    ]
                    .sort_values(["Date", "Time"], ascending=[False, False]),
                    use_container_width=True,
                    height=260,
                )

            st.markdown("#### Registro colpi (filtro attivo)")
            base_cols = [
                "Date",
                "Time",
                "SessionName",
                "Category",
                "Club",
                "Start_Dist",
                "Lie",
                "Impact",
                "Curvature",
                "Height",
                "Direction",
                "Proximity",
                "Rating",
            ]
            extra_sg = ["SG_Dist_Pin_Pre_m", "SG_Lie_Pre", "SG_Dist_Pin_Post_m", "SG_Lie_Post", "Strokes_Gained"]
            cols_view = [c for c in base_cols + extra_sg if c in df_f.columns]
            st.dataframe(df_f[cols_view].sort_values(["Date", "Time"], ascending=[False, False]), use_container_width=True, height=320)

if st.sidebar.button("LOGOUT / CAMBIA UTENTE"):
    st.session_state["logged_in"] = False
    st.rerun()


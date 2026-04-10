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
    "BrandTeal": "#3AB4B8",
    "DarkTeal": "#2A8285",
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
    .stButton>button {{ background-color: {COLORS['BrandTeal']}; color: white; border-radius: 8px; font-weight: bold; width: 100%; border: none; padding: 12px; }}
    .stButton>button:hover {{ background-color: {COLORS['DarkTeal']}; }}
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
]

CATEGORIES = ["LONG GAME / RANGE", "SHORT GAME", "PUTTING"]
CLUBS = ["DR", "3W", "5W", "7W", "3H", "3i", "4i", "5i", "6i", "7i", "8i", "9i", "PW", "AW", "GW", "SW", "LW"]
SHORT_GAME_CLUBS = ["LW", "SW", "GW", "AW", "PW", "9i", "8i"]


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
        user_input = st.text_input("ID Atleta (Nome)").upper().strip()
        pass_input = st.text_input("Master Password", type="password")
        if st.button("AUTENTICAZIONE"):
            if pass_input == "supernova.analytics" and user_input != "":
                st.session_state["logged_in"] = True
                st.session_state["user"] = user_input
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


# ==============================================================================
# 5. UI PRINCIPALE
# ==============================================================================
st.sidebar.markdown(f"### Atleta: {st.session_state['user']}")
session_name = st.sidebar.text_input("Sessione / Note", "Test Valutazione")

tab_input, tab_review = st.tabs(["REGISTRO COLPI", "ANALYTICS REVIEW"])

with tab_input:
    st.markdown("### Inserimento rapido colpo")
    st.caption("Workflow campo pratica: seleziona area, compila pochi campi, salva subito.")

    cat_scelta = st.radio("Area Tecnica", CATEGORIES, horizontal=True)
    with st.form("form_dati", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        start_dist = 0.0
        lie = "-"
        height = "-"
        direction = "Dritta"

        if cat_scelta == "LONG GAME / RANGE":
            with col1:
                club = st.selectbox("Bastone", CLUBS)
                impact = st.selectbox("Impatto", ["Solido", "Punta", "Tacco", "Shank", "Flappa", "Top"])
            with col2:
                curvature = st.selectbox("Curvatura (Effetto)", ["Dritta", "Push", "Pull", "Slice", "Hook"])
                height = st.selectbox("Altezza", ["Giusta", "Alta", "Bassa", "Rasoterra", "Flappa"])
            with col3:
                direction = st.selectbox("Direzione vs Target", ["Dritta", "Dx", "Sx"])
                proximity = st.number_input("Proximity Target (metri)", min_value=0.0, step=1.0)
            voto = st.slider("Voto Esecuzione (1-3)", 1, 3, 2)

        elif cat_scelta == "SHORT GAME":
            with col1:
                club = st.selectbox("Bastone", SHORT_GAME_CLUBS)
                start_dist = st.number_input("Distanza di Partenza (metri)", min_value=1.0, step=1.0)
            with col2:
                lie = st.selectbox("Lie", ["Fairway", "Rough", "Bunker", "Sponda"])
                impact = st.selectbox("Impatto", ["Solido", "Punta", "Tacco", "Shank", "Flappa", "Top"])
            with col3:
                curvature = st.selectbox("Curvatura Volo", ["Dritta", "Push", "Pull", "Slice", "Hook"])
                height = st.selectbox("Altezza", ["Giusta", "Alta", "Bassa", "Rasoterra", "Flappa"])
            bot1, bot2 = st.columns(2)
            direction = bot1.selectbox("Direzione vs Target", ["Dritta", "Dx", "Sx"])
            proximity = bot2.number_input("Proximity Finale (metri)", min_value=0.0, step=0.5)
            voto = st.slider("Voto Esecuzione (1-3)", 1, 3, 2)

        else:
            club = "Putter"
            with col1:
                start_dist = st.number_input("Distanza dal Buco (metri)", min_value=0.5, step=0.5)
            with col2:
                impact = st.selectbox("Impatto sulla Faccia", ["Centro", "Punta", "Tacco"])
            with col3:
                curvature = st.selectbox("Traiettoria / Linea", ["Dritta", "Push", "Pull"])
            proximity = st.number_input("Proximity (distanza residua metri)", min_value=0.0, step=0.1)
            voto = st.slider("Voto (3=Perfetto, 2=Data, 1=Errore)", 1, 3, 2)

        submitted = st.form_submit_button("SALVA COLPO")
        if submitted:
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
            }
            save_shot(shot)
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
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Shot Totali", f"{kpi['shots']}")
            k2.metric("Rating Medio", f"{kpi['avg_rating']:.2f}")
            k3.metric("Elite Rate (3/3)", f"{kpi['elite_rate']:.1f}%")
            k4.metric("Execution Index", f"{kpi['execution_index']:.1f}/100")
            k5.metric("Consistency Index", f"{kpi['consistency_index']:.1f}/100")

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

            st.markdown("#### Registro colpi (filtro attivo)")
            cols_view = ["Date", "Time", "SessionName", "Category", "Club", "Start_Dist", "Lie", "Impact", "Curvature", "Height", "Direction", "Proximity", "Rating"]
            st.dataframe(df_f[cols_view].sort_values(["Date", "Time"], ascending=[False, False]), use_container_width=True, height=320)

if st.sidebar.button("LOGOUT / CAMBIA UTENTE"):
    st.session_state["logged_in"] = False
    st.rerun()

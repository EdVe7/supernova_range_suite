import streamlit as st
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import datetime
import time
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile
from streamlit_gsheets import GSheetsConnection

# ==============================================================================
# 1. CONFIGURAZIONE E STILI
# ==============================================================================
st.set_page_config(page_title="Supernova Sport Science", page_icon="⛳", layout="wide")

COLORS = {
    'BrandTeal': '#3AB4B8',  
    'DarkTeal': '#2A8285',
    'Gold': '#DAA520', 
    'White': '#FFFFFF',
    'Grey': '#F3F4F6'
}

st.markdown(f"""
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
""", unsafe_allow_html=True)

# ==============================================================================
# 2. COSTANTI DATI
# ==============================================================================
COLUMNS = ['User', 'Date', 'SessionName', 'Time', 'Category', 'Club', 'Start_Dist', 'Lie', 'Impact', 'Curvature', 'Height', 'Direction', 'Proximity', 'Rating']
CATEGORIES = ["LONG GAME / RANGE", "SHORT GAME", "PUTTING"]
CLUBS = ["DR", "3W", "5W", "7W", "3H", "3i", "4i", "5i", "6i", "7i", "8i", "9i", "PW", "AW", "GW", "SW", "LW"]

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
                st.markdown(f"<p style='text-align:center; font-style:italic; color:{COLORS['DarkTeal']}; font-weight:bold;'>Data over talent</p>", unsafe_allow_html=True)
            except:
                st.markdown(f"<h1 style='text-align:center; font-size: 5rem; color:{COLORS['BrandTeal']};'>SUPERNOVA</h1><p style='text-align:center;'>SPORT SCIENCE SOLUTIONS</p><p style='text-align:center; font-style:italic; color:{COLORS['DarkTeal']}; font-weight:bold;'>Data over talent</p>", unsafe_allow_html=True)
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
            st.markdown(f"<p style='text-align:center; font-style:italic; color:{COLORS['DarkTeal']}; font-weight:bold;'>Data over talent</p>", unsafe_allow_html=True)
        except: pass
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
# 4. DATA ENGINE (Google Sheets)
# ==============================================================================
@st.cache_data(ttl=5)
def load_data():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl=0)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
        df['Proximity'] = pd.to_numeric(df['Proximity'], errors='coerce')
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

# ==============================================================================
# 5. GENERATORE GRAFICI SEABORN PER PDF
# ==============================================================================
def create_seaborn_pie(df, column, title):
    plt.figure(figsize=(6, 4))
    data = df[column].value_counts()
    colors = sns.color_palette("GnBu_d", len(data))
    plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 10})
    plt.title(title, fontsize=12, fontweight='bold', color=COLORS['DarkTeal'])
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(tmp.name, bbox_inches='tight', dpi=150)
    plt.close()
    return tmp.name

def create_seaborn_scatter(df, title):
    plt.figure(figsize=(8, 4))
    df_plot = df.copy()
    def calc_lat(row):
        if row['Direction'] == 'Dx': return row['Proximity']
        elif row['Direction'] == 'Sx': return -row['Proximity']
        return 0.0
    df_plot['Lateral_Error'] = df_plot.apply(calc_lat, axis=1)
    
    sns.scatterplot(data=df_plot, x='Lateral_Error', y='Club', hue='Club', s=100, palette="viridis", alpha=0.7)
    plt.axvline(0, color=COLORS['Gold'], linestyle='--', linewidth=2)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel("Errore Laterale (Metri: Sx < 0 > Dx)")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(tmp.name, bbox_inches='tight', dpi=150)
    plt.close()
    return tmp.name

# ==============================================================================
# 6. GENERATORE REPORT PDF (SUPERNOVA PRO)
# ==============================================================================
class PDFReport(FPDF):
    def header(self):
        try:
            self.image("logo.png", 10, 8, 33) # Logo in alto a sinistra
        except: pass
        self.set_font('Arial', 'B', 14)
        self.set_text_color(218, 165, 32)
        self.cell(0, 15, 'SUPERNOVA SPORT SCIENCE SOLUTIONS', 0, 1, 'R')
        self.set_draw_color(218, 165, 32)
        self.line(10, 32, 200, 32)
        self.ln(15) # Spazio dal logo/header al testo

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(58, 180, 184) # Verde Acqua / Teal
        self.cell(0, 10, 'Data over talent', 0, 0, 'C')

def calc_perc(df, col, val):
    if len(df) == 0: return 0.0
    return (len(df[df[col] == val]) / len(df)) * 100

def generate_pro_pdf(df, user, period_name):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 6, f"ANALISI PERFORMANCE ELITE - ATLETA: {user}", ln=True)
    pdf.cell(0, 6, f"PERIODO ANALISI: {period_name} | GENERATO IL: {datetime.date.today()}", ln=True)
    pdf.ln(5)

    for cat in CATEGORIES:
        pdf.set_font('Arial', 'B', 12)
        pdf.set_text_color(255, 255, 255)
        pdf.set_fill_color(58, 180, 184)
        pdf.cell(0, 8, f" REPARTO: {cat} ", 0, 1, 'L', fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

        df_cat = df[df['Category'] == cat]
        if df_cat.empty:
            pdf.set_font('Arial', 'I', 10)
            pdf.set_text_color(218, 165, 32) # Colore Oro per segnalare l'assenza di dati
            pdf.cell(0, 8, f"Nessuno score registrato per questa area in questo periodo.", ln=True)
            pdf.ln(8)
            continue
        
        # Statistiche testuali
        pdf.set_font('Arial', '', 9)
        pdf.cell(0, 5, f"Volume: {len(df_cat)} colpi | Voto Medio: {df_cat['Rating'].mean():.2f}/3.0", ln=True)
        pdf.cell(0, 5, f"Efficienza (Voto 3): {calc_perc(df_cat, 'Rating', 3):.1f}%", ln=True)
        pdf.ln(5)

        # Inserimento Grafici Seaborn nel PDF
        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(218, 165, 32)
        pdf.cell(0, 6, "DISTRIBUZIONE TECNICA E TENDENZE:", ln=True)
        
        # Grafico Impatti
        img_imp = create_seaborn_pie(df_cat, 'Impact', f"Qualita Impatto - {cat}")
        pdf.image(img_imp, x=15, y=pdf.get_y(), w=85)
        
        # Grafico Traiettorie
        img_curv = create_seaborn_pie(df_cat, 'Curvature', f"Tendenza Effetto - {cat}")
        pdf.image(img_curv, x=110, y=pdf.get_y(), w=85)
        pdf.set_y(pdf.get_y() + 65)

        # Spiegazione Grafici staccata e colorata
        pdf.ln(12)
        pdf.set_font('Arial', 'I', 9)
        pdf.set_fill_color(58, 180, 184) # Sfondo Verde Acqua
        pdf.set_text_color(255, 255, 255) # Testo Bianco
        pdf.multi_cell(0, 6, " Interpretazione:", 0, 'L', fill=True)
        pdf.set_text_color(42, 130, 133) # Testo Dark Teal
        pdf.multi_cell(0, 5, "I grafici a torta mostrano la consistenza del punto di impatto e la curvatura predominante. Uno slice/push costante indica un errore di cammino/faccia sistematico; impatti decentrati suggeriscono instabilita del raggio dell'arco di swing.")
        pdf.ln(8)

        if cat in ["LONG GAME / RANGE", "SHORT GAME"]:
            img_scat = create_seaborn_scatter(df_cat, f"Mappa Dispersione Laterale - {cat}")
            pdf.image(img_scat, x=30, y=pdf.get_y(), w=150)
            pdf.set_y(pdf.get_y() + 85)
            
            pdf.ln(10)
            pdf.set_font('Arial', 'I', 9)
            pdf.set_text_color(218, 165, 32) # Testo Oro
            pdf.multi_cell(0, 5, "Mappa Dispersione: Rappresenta la distanza laterale dal target. L'obiettivo e raggruppare i punti sulla linea centrale. Una dispersione ampia indica scarsa ripetibilita dinamica.")
            pdf.ln(5)

        if pdf.get_y() > 220: pdf.add_page() # Salto pagina se spazio esaurito

    # Script Finale per l'Atleta
    pdf.ln(10)
    pdf.set_draw_color(58, 180, 184)
    pdf.set_fill_color(243, 244, 246)
    pdf.set_font('Arial', 'B', 11)
    pdf.set_text_color(58, 180, 184)
    pdf.cell(0, 10, " MESSAGGIO DAL SUPERNOVA PERFORMANCE LAB ", 1, 1, 'C', fill=True)
    pdf.set_font('Arial', 'I', 10)
    pdf.set_text_color(42, 130, 133)
    script_text = (
        "Il talento fornisce la base, ma i dati costruiscono la vittoria. Analizzando i tuoi volumi, "
        "focalizzati sul trasformare i tuoi colpi 'Accettabili' in 'Esecuzioni Perfette' attraverso il controllo dell'impatto. "
        "La tua tendenza laterale e la chiave per il prossimo step tecnico. Torna in campo, la costanza e l'unica via."
    )
    pdf.multi_cell(0, 6, script_text, 0, 'C')
    
    return pdf.output(dest='S').encode('latin-1')

# ==============================================================================
# 7. INTERFACCIA APP PRINCIPALE
# ==============================================================================
st.sidebar.markdown(f"### Atleta: {st.session_state['user']}")
session_name = st.sidebar.text_input("Sessione / Note", "Test Valutazione")

tab_in, tab_an = st.tabs([" REGISTRO TELEMETRIA", " ANALYTICS & REPORT"])

with tab_in:
    cat_scelta = st.radio("Seleziona Area Tecnica", CATEGORIES, horizontal=True)
    with st.form("form_dati", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        start_dist = 0.0; lie = "-"; height = "-"; direction = "-"

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
            voto = st.slider("Voto Esecuzione (1=Scarso, 2=Accettabile, 3=Perfetto)", 1, 3, 2)
            
        elif cat_scelta == "SHORT GAME":
            with col1:
                club = st.selectbox("Bastone", ["LW", "SW", "GW", "AW", "PW", "9i", "8i"])
                start_dist = st.number_input("Distanza di Partenza (metri)", min_value=1.0, step=1.0)
            with col2:
                lie = st.selectbox("Lie", ["Fairway", "Rough", "Bunker", "Sponda"])
                impact = st.selectbox("Impatto", ["Solido", "Punta", "Tacco", "Shank", "Flappa", "Top"])
            with col3:
                curvature = st.selectbox("Curvatura Volo", ["Dritta", "Push", "Pull", "Slice", "Hook"])
                height = st.selectbox("Altezza", ["Giusta", "Alta", "Bassa", "Rasoterra", "Flappa"])
            c_bot1, c_bot2 = st.columns(2)
            direction = c_bot1.selectbox("Direzione vs Target", ["Dritta", "Dx", "Sx"])
            proximity = c_bot2.number_input("Proximity Finale (metri)", min_value=0.0, step=0.5)
            voto = st.slider("Voto Esecuzione (1=Scarso, 2=Accettabile, 3=Perfetto)", 1, 3, 2)
            
        else: # PUTTING
            club = "Putter"
            with col1: start_dist = st.number_input("Distanza dal Buco (metri)", min_value=0.5, step=0.5)
            with col2: impact = st.selectbox("Impatto sulla Faccia", ["Centro", "Punta", "Tacco"])
            with col3: curvature = st.selectbox("Traiettoria / Linea", ["Dritta", "Push", "Pull"])
            proximity = st.number_input("Proximity (Distanza residua metri)", min_value=0.0, step=0.1)
            voto = st.slider("Voto (3=Buca/Perfetto, 2=Data, 1=Errore)", 1, 3, 2)

        if st.form_submit_button("SALVA COLPO NEL DATABASE"):
            shot = {
                'User': st.session_state['user'], 'Date': datetime.date.today(),
                'SessionName': session_name, 'Time': datetime.datetime.now().strftime("%H:%M"),
                'Category': cat_scelta, 'Club': club, 'Start_Dist': start_dist, 'Lie': lie,
                'Impact': impact, 'Curvature': curvature, 'Height': height,
                'Direction': direction, 'Proximity': proximity, 'Rating': voto
            }
            save_shot(shot)
            st.success("Metriche acquisite con successo.")

with tab_an:
    df_all = load_data()
    df_user = df_all[df_all['User'] == st.session_state['user']]
    periodo = st.selectbox("Filtro Temporale", ["Sessione Attuale", "Ultimi 7 Giorni", "Ultimi 30 Giorni", "Tutti i Dati"])
    oggi = datetime.date.today()
    if periodo == "Sessione Attuale": df_f = df_user[df_user['SessionName'] == session_name]
    elif periodo == "Ultimi 7 Giorni": df_f = df_user[df_user['Date'] >= (oggi - datetime.timedelta(days=7))]
    elif periodo == "Ultimi 30 Giorni": df_f = df_user[df_user['Date'] >= (oggi - datetime.timedelta(days=30))]
    else: df_f = df_user

    if not df_f.empty:
        pdf_bytes = generate_pro_pdf(df_f, st.session_state['user'], periodo)
        st.download_button(" SCARICA REPORT SUPERNOVA COMPLETO (PDF)", data=pdf_bytes, file_name=f"Supernova_Report_{st.session_state['user']}.pdf", mime="application/pdf")
        st.divider()

        cat_grafici = st.radio("Dettaglio Grafici per Area", CATEGORIES, horizontal=True)
        df_p = df_f[df_f['Category'] == cat_grafici].copy()

        if not df_p.empty:
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f"<div class='metric-box'><div class='metric-title'>Voto Medio</div><div class='metric-value'>{df_p['Rating'].mean():.2f}</div></div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div class='metric-box'><div class='metric-title'>Colpi Registrati</div><div class='metric-value'>{len(df_p)}</div></div>", unsafe_allow_html=True)
            with c3: st.markdown(f"<div class='metric-box'><div class='metric-title'>Proximity Media</div><div class='metric-value'>{df_p['Proximity'].mean():.1f} m</div></div>", unsafe_allow_html=True)

            col_g1, col_g2 = st.columns(2)
            with col_g1: st.plotly_chart(px.pie(df_p, names='Impact', title="Analisi Impatti", hole=0.3, color_discrete_sequence=px.colors.sequential.Teal), use_container_width=True)
            with col_g2: st.plotly_chart(px.pie(df_p, names='Curvature', title="Analisi Traiettorie", hole=0.3, color_discrete_sequence=px.colors.sequential.Teal), use_container_width=True)
            
            if cat_grafici in ["LONG GAME / RANGE", "SHORT GAME"]:
                def calc_lateral(row):
                    if row['Direction'] == 'Dx': return row['Proximity']
                    elif row['Direction'] == 'Sx': return -row['Proximity']
                    return 0.0
                df_p['Lateral_Error'] = df_p.apply(calc_lateral, axis=1)
                fig_scatter = px.scatter(df_p, x='Lateral_Error', y='Club', color='Club', title="Dispersione Laterale", labels={'Lateral_Error': 'Sx <-- 0 --> Dx'})
                fig_scatter.add_vline(x=0, line_dash="dash", line_color=COLORS['Gold'])
                st.plotly_chart(fig_scatter, use_container_width=True)

# ==============================================================================
# 8. ELITE ANALYTICS UPGRADE (APPEND-ONLY, NON-INVASIVO)
# ==============================================================================

ELITE_PROX_TARGETS = {
    "LONG GAME / RANGE": 18.0,
    "SHORT GAME": 4.0,
    "PUTTING": 0.9
}

def _safe_mean(series, default=0.0):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else default

def _safe_std(series, default=0.0):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.std()) if len(s) > 1 else default

def _calc_lateral_error_series(df):
    def calc_lateral(row):
        if row.get('Direction') == 'Dx':
            return row.get('Proximity', 0.0)
        elif row.get('Direction') == 'Sx':
            return -row.get('Proximity', 0.0)
        return 0.0
    return df.apply(calc_lateral, axis=1)

def _normalize_0_100(value, min_v, max_v):
    if max_v <= min_v:
        return 0.0
    v = max(min(value, max_v), min_v)
    return ((v - min_v) / (max_v - min_v)) * 100.0

def compute_elite_kpis(df_area, area_name):
    if df_area.empty:
        return {
            "shots": 0, "avg_rating": 0.0, "elite_rate": 0.0, "acceptable_plus": 0.0,
            "avg_proximity": 0.0, "proximity_std": 0.0, "lateral_std": 0.0,
            "consistency_index": 0.0, "execution_index": 0.0, "pressure_index": 0.0,
            "sample_quality": "LOW"
        }

    shots = len(df_area)
    avg_rating = _safe_mean(df_area["Rating"])
    elite_rate = (len(df_area[df_area["Rating"] == 3]) / shots) * 100.0
    acceptable_plus = (len(df_area[df_area["Rating"] >= 2]) / shots) * 100.0
    avg_prox = _safe_mean(df_area["Proximity"])
    prox_std = _safe_std(df_area["Proximity"])

    lat = _calc_lateral_error_series(df_area)
    lat_std = _safe_std(lat)

    # Indice consistenza: penalizza variabilita su proximity e laterale
    prox_component = 100.0 - _normalize_0_100(prox_std, 0.0, 12.0)
    lat_component = 100.0 - _normalize_0_100(lat_std, 0.0, 15.0)
    consistency = max(0.0, (0.6 * prox_component) + (0.4 * lat_component))

    # Indice esecuzione: voto medio + elite rate
    exec_rating_component = _normalize_0_100(avg_rating, 1.0, 3.0)
    execution_index = (0.55 * exec_rating_component) + (0.45 * elite_rate)

    # Pressure index: performance nei colpi piu "difficili"
    # Heuristica: long -> proximity > target, short -> lie bunker/rough, putting -> start_dist >= 2m
    target = ELITE_PROX_TARGETS.get(area_name, 5.0)
    hard_subset = pd.DataFrame(columns=df_area.columns)
    if area_name == "LONG GAME / RANGE":
        hard_subset = df_area[pd.to_numeric(df_area["Proximity"], errors="coerce") > target]
    elif area_name == "SHORT GAME":
        hard_subset = df_area[df_area["Lie"].isin(["Rough", "Bunker", "Sponda"])]
    elif area_name == "PUTTING":
        hard_subset = df_area[pd.to_numeric(df_area["Start_Dist"], errors="coerce") >= 2.0]

    if len(hard_subset) >= 5:
        pressure_index = _normalize_0_100(_safe_mean(hard_subset["Rating"]), 1.0, 3.0)
    else:
        pressure_index = execution_index * 0.9  # fallback prudente

    if shots < 25:
        sample_quality = "LOW"
    elif shots < 80:
        sample_quality = "MEDIUM"
    else:
        sample_quality = "HIGH"

    return {
        "shots": shots,
        "avg_rating": avg_rating,
        "elite_rate": elite_rate,
        "acceptable_plus": acceptable_plus,
        "avg_proximity": avg_prox,
        "proximity_std": prox_std,
        "lateral_std": lat_std,
        "consistency_index": consistency,
        "execution_index": execution_index,
        "pressure_index": pressure_index,
        "sample_quality": sample_quality
    }

def generate_focus_points(kpi, area_name):
    tips = []
    target = ELITE_PROX_TARGETS.get(area_name, 5.0)

    if kpi["avg_proximity"] > target:
        tips.append(f"Ridurre proximity media: target area {target:.1f}m, attuale {kpi['avg_proximity']:.1f}m.")
    if kpi["proximity_std"] > 5.5:
        tips.append("Aumentare ripetibilita sul controllo distanza (variabilita troppo alta).")
    if kpi["elite_rate"] < 35:
        tips.append("Incrementare volume di drill orientati a conversione in colpi '3/3'.")
    if kpi["consistency_index"] < 60:
        tips.append("Priorita tecnica: stabilita impatto + gestione faccia/linea al momento del contatto.")
    if kpi["pressure_index"] < 55:
        tips.append("Inserire blocchi 'pressure sets' (es. 5x5 colpi con obiettivo minimo score).")
    if not tips:
        tips.append("Trend molto solido: mantenere routine e alzare progressivamente la difficolta bersaglio.")
    return tips[:4]

def create_trend_chart(df_area, title):
    if df_area.empty:
        return None
    dft = df_area.copy()
    dft["Date"] = pd.to_datetime(dft["Date"], errors="coerce")
    dft = dft.dropna(subset=["Date"])
    if dft.empty:
        return None
    grp = dft.groupby("Date", as_index=False).agg({
        "Rating": "mean",
        "Proximity": "mean"
    }).sort_values("Date")
    fig = px.line(
        grp,
        x="Date",
        y=["Rating", "Proximity"],
        markers=True,
        title=title,
        color_discrete_sequence=[COLORS["BrandTeal"], COLORS["Gold"]]
    )
    fig.update_layout(legend_title_text="Metriche")
    return fig

def create_club_performance_chart(df_area, title):
    if df_area.empty:
        return None
    d = df_area.copy()
    d["Rating"] = pd.to_numeric(d["Rating"], errors="coerce")
    d["Proximity"] = pd.to_numeric(d["Proximity"], errors="coerce")
    grp = d.groupby("Club", as_index=False).agg(
        Rating=("Rating", "mean"),
        Proximity=("Proximity", "mean"),
        Volume=("Club", "count")
    )
    grp = grp[grp["Volume"] >= 3].sort_values("Rating", ascending=False)
    if grp.empty:
        return None
    fig = px.scatter(
        grp, x="Proximity", y="Rating", size="Volume", color="Club",
        title=title, hover_data=["Volume"],
        color_discrete_sequence=px.colors.sequential.Teal
    )
    fig.update_layout(xaxis_title="Proximity media (m)", yaxis_title="Rating medio")
    return fig

def generate_elite_pdf(df, user, period_name):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, f"ELITE PERFORMANCE DOSSIER - ATLETA: {user}", ln=True)
    pdf.cell(0, 7, f"PERIODO: {period_name} | GENERATO: {datetime.date.today()}", ln=True)
    pdf.ln(3)

    # Executive Summary
    all_kpi = compute_elite_kpis(df, "GLOBAL")
    pdf.set_fill_color(42, 130, 133)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 8, " EXECUTIVE SUMMARY ", 0, 1, 'L', fill=True)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 9)
    pdf.cell(0, 6, f"Volume totale: {len(df)} colpi", ln=True)
    pdf.cell(0, 6, f"Execution Index: {all_kpi['execution_index']:.1f}/100 | Consistency Index: {all_kpi['consistency_index']:.1f}/100", ln=True)
    pdf.cell(0, 6, f"Elite Rate (3/3): {all_kpi['elite_rate']:.1f}% | Pressure Index: {all_kpi['pressure_index']:.1f}/100", ln=True)
    pdf.ln(4)

    for cat in CATEGORIES:
        df_cat = df[df["Category"] == cat]
        kpi = compute_elite_kpis(df_cat, cat)

        pdf.set_fill_color(58, 180, 184)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 8, f" AREA: {cat} ", 0, 1, 'L', fill=True)

        if df_cat.empty:
            pdf.set_text_color(218, 165, 32)
            pdf.set_font('Arial', 'I', 9)
            pdf.cell(0, 6, "Nessun dato nel periodo selezionato.", ln=True)
            pdf.ln(3)
            continue

        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', '', 9)
        pdf.cell(0, 5, f"Shots: {kpi['shots']} | Avg Rating: {kpi['avg_rating']:.2f} | Elite Rate: {kpi['elite_rate']:.1f}%", ln=True)
        pdf.cell(0, 5, f"Avg Proximity: {kpi['avg_proximity']:.2f}m | Prox Std: {kpi['proximity_std']:.2f} | Sample: {kpi['sample_quality']}", ln=True)
        pdf.cell(0, 5, f"Execution: {kpi['execution_index']:.1f}/100 | Consistency: {kpi['consistency_index']:.1f}/100 | Pressure: {kpi['pressure_index']:.1f}/100", ln=True)
        pdf.ln(2)

        # Grafici come nel report originale
        img_imp = create_seaborn_pie(df_cat, 'Impact', f"Impact Quality - {cat}")
        img_curv = create_seaborn_pie(df_cat, 'Curvature', f"Shot Shape - {cat}")
        y_now = pdf.get_y()
        pdf.image(img_imp, x=15, y=y_now, w=85)
        pdf.image(img_curv, x=110, y=y_now, w=85)
        pdf.set_y(y_now + 62)

        if cat in ["LONG GAME / RANGE", "SHORT GAME"] and len(df_cat) >= 5:
            img_scat = create_seaborn_scatter(df_cat, f"Lateral Dispersion - {cat}")
            pdf.image(img_scat, x=28, y=pdf.get_y(), w=155)
            pdf.set_y(pdf.get_y() + 78)

        pdf.set_text_color(42, 130, 133)
        pdf.set_font('Arial', 'B', 9)
        pdf.cell(0, 5, "Focus tecnico consigliato:", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', '', 9)
        for tip in generate_focus_points(kpi, cat):
            pdf.multi_cell(0, 5, f"- {tip}")
        pdf.ln(2)

        if pdf.get_y() > 240:
            pdf.add_page()

    pdf.ln(2)
    pdf.set_font('Arial', 'I', 9)
    pdf.set_text_color(58, 180, 184)
    pdf.multi_cell(0, 6, "Coach note: utilizzare i tre indici (Execution/Consistency/Pressure) per decidere priorita microciclo e progressione difficolta.")
    return pdf.output(dest='S').encode('latin-1')


# ------------------------------------------------------------------------------
# UI ELITE (nuova sezione dentro tab analytics, non sostituisce nulla)
# ------------------------------------------------------------------------------
with tab_an:
    st.divider()
    st.markdown("### ELITE PERFORMANCE LAB (PRO UPGRADE)")
    st.caption("Sezione avanzata per atleta/coach: KPI professionali, trend e PDF esteso.")

    if 'df_f' in locals() and not df_f.empty:
        area_elite = st.selectbox("Area Elite Analysis", CATEGORIES, key="elite_area")
        df_elite = df_f[df_f["Category"] == area_elite].copy()
        kpi = compute_elite_kpis(df_elite, area_elite)

        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Execution Index", f"{kpi['execution_index']:.1f}/100")
        ec2.metric("Consistency Index", f"{kpi['consistency_index']:.1f}/100")
        ec3.metric("Pressure Index", f"{kpi['pressure_index']:.1f}/100")
        ec4.metric("Elite Rate (3/3)", f"{kpi['elite_rate']:.1f}%")

        tcol1, tcol2 = st.columns(2)
        with tcol1:
            fig_trend = create_trend_chart(df_elite, f"Trend Rating/Proximity - {area_elite}")
            if fig_trend is not None:
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Trend non disponibile (dati insufficienti).")
        with tcol2:
            fig_club = create_club_performance_chart(df_elite, f"Club Efficiency Map - {area_elite}")
            if fig_club is not None:
                st.plotly_chart(fig_club, use_container_width=True)
            else:
                st.info("Club map non disponibile (servono almeno 3 colpi per club).")

        st.markdown("#### Focus Operativo Coach")
        for tip in generate_focus_points(kpi, area_elite):
            st.write(f"- {tip}")

        elite_pdf = generate_elite_pdf(df_f, st.session_state['user'], periodo)
        st.download_button(
            " SCARICA ELITE DOSSIER (PDF PRO COACH)",
            data=elite_pdf,
            file_name=f"Elite_Dossier_{st.session_state['user']}.pdf",
            mime="application/pdf",
            key="elite_pdf_btn"
        )
    else:
        st.info("Nessun dato disponibile nel filtro corrente per attivare l'Elite Lab.")
        
if st.sidebar.button("LOGOUT / CAMBIA UTENTE"):
    st.session_state["logged_in"] = False
    st.rerun()

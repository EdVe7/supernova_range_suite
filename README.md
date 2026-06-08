# Zanardelli Range Suite

## requirements.txt

```
streamlit
pandas
numpy
plotly
git+https://github.com/streamlit/gsheets-connection.git@main
```

La riga `git+...` installa `streamlit_gsheets` (non è su PyPI). **Serve su Streamlit Cloud** per evitare `ModuleNotFoundError`. Il deploy ha git e la installa automaticamente.

## secrets.toml (Streamlit Cloud — invariato)

```toml
[connections.gsheets]
spreadsheet = "URL_O_ID_DEL_TUO_FOGLIO"
# credenziali service account come nell'originale

APP_PASSWORD = "tua_password"   # opzionale
```

## Colonne Google Sheet (ordine fisso — DATA_COLUMNS)

| Colonna | Descrizione |
|---------|-------------|
| User | ID atleta |
| Date | Data (YYYY-MM-DD) |
| SessionName | Nome sessione |
| Time | Ora (HH:MM) |
| Category | RANGE / SHORT / PUTT |
| Club | Bastone |
| Impact | Tipo impatto |
| Curvature | Curvatura |
| Trajectory | Traiettoria (putting) |
| Lie_Start | Lie iniziale |
| Lie_End | Lie finale |
| Direction_LR | Direzione vs bersaglio/buca |
| Proximity_Lateral_m | Errore laterale (m, con segno) |
| Proximity_Depth_m | Errore profondità (m, con segno) |
| Start_Dist_m | Distanza inizio (m) |
| End_Dist_m | Distanza fine (m) |
| Hole_Dist_Start_m | Distanza buca prima colpo |
| Hole_Dist_End_m | Distanza buca dopo colpo |
| Lie_Long | Tee/Fairway (range) |
| Rating | Voto 1–5 |
| Mental_Reaction | Reazione mentale |
| Strokes_Gained | SG calcolato |

## Avvio

```bash
pip install -r requirements.txt
streamlit run app.py
```

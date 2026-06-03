# Zanardelli Range Suite

## Avvio locale (coach / server)

```bash
pip install -r requirements.txt
streamlit run app.py
```

`reportlab` è incluso in `requirements.txt`: serve **solo** sulla macchina che esegue Streamlit, non sui PC degli atleti.

## Streamlit Cloud

1. Carica su GitHub: `app.py`, `requirements.txt`, `logo.png`, `.streamlit/secrets.toml` (credenziali).
2. Collega il repo su [share.streamlit.io](https://share.streamlit.io) — le dipendenze si installano da sole da `requirements.txt`.
3. Gli utenti aprono il link e usano **Scarica Diario di Gioco (PDF)** nel browser.

## File necessari nel repo

| File | Ruolo |
|------|--------|
| `app.py` | Applicazione |
| `requirements.txt` | Dipendenze (incluso `reportlab` per il PDF) |
| `logo.png` | Logo splash e Diario PDF |
| `.streamlit/secrets.toml` | Password app + Google Sheets (non committare in pubblico) |

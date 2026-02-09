import streamlit as st
import requests
import json

# Configurazione Pagina
st.set_page_config(page_title="UFC Predictor AI", page_icon="🥊", layout="centered")

# --- CUSTOM CSS (UFC STYLE: Red, Black, White) ---
st.markdown("""
    <style>
    .main {
        background-color: #111111;
        color: #FFFFFF;
    }
    .stButton>button {
        width: 100%;
        background-color: #D20A0A; /* Rosso UFC */
        color: white;
        border-radius: 5px;
        border: none;
        font-weight: bold;
        padding: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #A00808;
        border: none;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #222222;
        color: white;
        border: 1px solid #444444;
    }
    .prediction-card {
        background-color: #222222;
        padding: 25px;
        border-radius: 10px;
        border-left: 8px solid #D20A0A;
        text-align: center;
        margin-top: 20px;
    }
    .confidence-text {
        color: #D20A0A;
        font-size: 24px;
        font-weight: bold;
    }
    .ai-act-disclaimer {
        font-size: 11px;
        color: #888888;
        text-align: center;
        margin-top: 50px;
        line-height: 1.4;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.image("https://pngimg.com/d/ufc_PNG61.png", width=150)  # Logo UFC
st.title("AI FIGHT PREDICTOR")
st.subheader("Data-driven Match Analysis")

# --- INPUT AREA ---
col1, col2 = st.columns(2)

with col1:
    f1_name = st.text_input("Fighter 1", placeholder="Es: Ilia Topuria")

with col2:
    f2_name = st.text_input("Fighter 2", placeholder="Es: Max Holloway")

# --- PREDICTION LOGIC ---
if st.button("CALCOLA PREDIZIONE"):
    if f1_name and f2_name:
        with st.spinner('Analizzando le statistiche dei fighter...'):
            try:
                # Chiamata alla tua API FastAPI
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json={"f1": f1_name, "f2": f2_name}
                )

                if response.status_code == 200:
                    data = response.json()

                    # --- DISPLAY RESULT ---
                    st.markdown(f"""
                        <div class="prediction-card">
                            <h2 style='margin-bottom: 0;'>VINCITORE PREVISTO</h2>
                            <h1 style='color: #FFFFFF; font-size: 45px; margin-top: 10px;'>{data['prediction'].upper()}</h1>
                            <p style='margin-bottom: 5px;'>CONFIDENCE LIVELLO</p>
                            <span class="confidence-text">{data['confidence']}</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"Errore: {response.json().get('detail', 'Impossibile recuperare i dati')}")

            except Exception as e:
                st.error(f"Connessione all'API fallita. Assicurati che api.py sia in esecuzione. ({e})")
    else:
        st.warning("Inserisci il nome di entrambi i lottatori per continuare.")

# --- AI ACT COMPLIANCE FOOTER ---
st.markdown("""
    <div class="ai-act-disclaimer">
        <b>Informativa AI Act:</b> Questa predizione è generata da un sistema automatizzato basato su un <b>Ensemble Model</b> 
        che combina algoritmi di <b>Random Forest, LightGBM e XGBoost</b>. <br>
        Il modello analizza dati storici (record, età, reach, striking accuracy) estratti in tempo reale dal sito
        <a href="http://ufcstats.com" target="_blank"> ufcstats.com</a>. 
        Le predizioni non costituiscono consigli finanziari o di scommessa e sono fornite a scopo puramente informativo.
    </div>
    """, unsafe_allow_html=True)
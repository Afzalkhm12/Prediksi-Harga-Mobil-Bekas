import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings
import random

# 1. SETUP & CONFIG
warnings.filterwarnings("ignore")
st.set_page_config(page_title="AutoValue Pro", page_icon="🚘", layout="wide")

# CSS Styling Professional
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background-color: white; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); text-align: center;
        border-top: 4px solid #3498db;
    }
    .verdict-box {
        padding: 20px; border-radius: 12px; text-align: center; 
        font-weight: 800; color: white; margin-bottom: 25px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1); font-size: 1.2rem;
    }
    .stButton>button {
        width: 100%; border-radius: 8px; height: 3em; font-weight: bold;
        transition: all 0.3s;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. KAMUS MAPPING (USER FRIENDLY VS MODEL)
# ==============================================================================

# Mapping Wilayah (Kode -> Nama Lengkap)
STATE_MAP = {
    'al': 'Alabama', 'ak': 'Alaska', 'az': 'Arizona', 'ar': 'Arkansas', 'ca': 'California',
    'co': 'Colorado', 'ct': 'Connecticut', 'de': 'Delaware', 'fl': 'Florida', 'ga': 'Georgia',
    'hi': 'Hawaii', 'id': 'Idaho', 'il': 'Illinois', 'in': 'Indiana', 'ia': 'Iowa',
    'ks': 'Kansas', 'ky': 'Kentucky', 'la': 'Louisiana', 'me': 'Maine', 'md': 'Maryland',
    'ma': 'Massachusetts', 'mi': 'Michigan', 'mn': 'Minnesota', 'ms': 'Mississippi', 'mo': 'Missouri',
    'mt': 'Montana', 'ne': 'Nebraska', 'nv': 'Nevada', 'nh': 'New Hampshire', 'nj': 'New Jersey',
    'nm': 'New Mexico', 'ny': 'New York', 'nc': 'North Carolina', 'nd': 'North Dakota', 'oh': 'Ohio',
    'ok': 'Oklahoma', 'or': 'Oregon', 'pa': 'Pennsylvania', 'ri': 'Rhode Island', 'sc': 'South Carolina',
    'sd': 'South Dakota', 'tn': 'Tennessee', 'tx': 'Texas', 'ut': 'Utah', 'vt': 'Vermont',
    'va': 'Virginia', 'wa': 'Washington', 'wv': 'West Virginia', 'wi': 'Wisconsin', 'wy': 'Wyoming'
}

# Mapping Kondisi (Tampilan Indo -> Nilai Model)
CONDITION_MAP_UI = {
    'Baru (New)': 'new',
    'Seperti Baru (Like New)': 'like new',
    'Sangat Bagus (Excellent)': 'excellent',
    'Bagus (Good)': 'good',
    'Layak (Fair)': 'fair',
    'Rusak (Salvage)': 'salvage'
}

# ==============================================================================
# 3. LOAD ASSETS
# ==============================================================================
@st.cache_resource
def load_assets():
    status = {"model": False, "scaler": False}
    try:
        model = joblib.load('model_final.pkl')
        cols = joblib.load('model_columns.pkl')
        ref_data = joblib.load('reference_data.pkl')
        status["model"] = True
        try:
            scaler = joblib.load('scaler.pkl')
            status["scaler"] = True
        except: scaler = None
        return model, cols, ref_data, scaler, status
    except: return None, None, None, None, status

model, model_cols, ref_data, scaler, status = load_assets()

# ==============================================================================
# 4. SIDEBAR INPUT
# ==============================================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3202/3202926.png", width=70)
st.sidebar.title("🚘 Parameter Mobil")

# --- FITUR TOMBOL DEMO (PENTING BUAT PRESENTASI) ---
if st.sidebar.button("🎲 Isi Data Acak (Demo)"):
    st.session_state['demo_run'] = True
else:
    if 'demo_run' not in st.session_state:
        st.session_state['demo_run'] = False

input_data = {}
btn_calc = False

if status["model"]:
    # Gunakan Session State untuk nilai default (biar bisa di-randomize)
    def get_default(key, default_val):
        if st.session_state['demo_run']:
            if isinstance(default_val, int): return random.randint(default_val, default_val + 5)
            return default_val
        return default_val

    # 1. INPUT UTAMA
    manufacturers = ref_data.get('manufacturer', ['ford'])
    input_data['manufacturer'] = st.sidebar.selectbox("Merk", manufacturers, index=0)
    
    c1, c2 = st.sidebar.columns(2)
    with c1: input_data['year'] = st.number_input("Tahun", 1990, 2026, 2018 if not st.session_state['demo_run'] else random.randint(2015, 2024))
    with c2: input_data['odometer'] = st.number_input("KM (Miles)", 0, 500000, 45000 if not st.session_state['demo_run'] else random.randint(10000, 80000), step=1000)
    
    # Kondisi (UI Bahasa Indonesia)
    cond_label = st.sidebar.selectbox("Kondisi", list(CONDITION_MAP_UI.keys()), index=2)
    input_data['condition'] = CONDITION_MAP_UI[cond_label]

    # Silinder (Format User Friendly)
    # Model butuh "6 cylinders", User butuh "6 Silinder"
    cyl_display = ['3 Silinder', '4 Silinder', '5 Silinder', '6 Silinder', '8 Silinder', '10 Silinder', '12 Silinder']
    cyl_values = ['3 cylinders', '4 cylinders', '5 cylinders', '6 cylinders', '8 cylinders', '10 cylinders', '12 cylinders']
    cyl_idx = st.sidebar.selectbox("Silinder", cyl_display, index=3)
    input_data['cylinders'] = cyl_values[cyl_display.index(cyl_idx)] # Konversi balik ke format model
    
    # 2. INPUT TAMBAHAN
    with st.sidebar.expander("⚙️ Opsi Teknis & Wilayah"):
        input_data['fuel'] = st.selectbox("Bahan Bakar", ref_data.get('fuel', ['gas']), index=0)
        input_data['transmission'] = st.selectbox("Transmisi", ref_data.get('transmission', ['automatic']), index=0)
        
        # WILAYAH YANG DIPERBAIKI (Format: Nama Lengkap)
        # Ambil daftar kode state dari data referensi
        available_states = ref_data.get('state', ['ca', 'tx', 'fl'])
        # Buat list nama lengkap untuk dropdown
        state_labels = [f"{STATE_MAP.get(s, s.upper())} ({s.upper()})" for s in available_states]
        
        state_choice = st.selectbox("Lokasi / Wilayah", state_labels, index=0)
        # Ambil balik kode singkatan (misal 'ca') untuk dikirim ke model
        input_data['state'] = available_states[state_labels.index(state_choice)]
        
        # Fitur Lain
        input_data['drive'] = st.selectbox("Penggerak", ['4wd', 'fwd', 'rwd'], index=0)
        input_data['type'] = st.selectbox("Tipe Bodi", ['sedan', 'SUV', 'truck', 'coupe'], index=1)
        input_data['paint_color'] = st.selectbox("Warna", ['white', 'black', 'silver', 'red', 'blue'], index=0)
        input_data['title_status'] = st.selectbox("Status Dokumen", ['clean', 'rebuilt', 'salvage'], index=0)
        input_data['size'] = st.selectbox("Ukuran", ['full-size', 'mid-size', 'compact'], index=0)

    # 3. KOMPARASI HARGA
    st.sidebar.markdown("---")
    st.sidebar.caption("💰 Penawaran Penjual (Opsional)")
    listing_price = st.sidebar.number_input("Harga Iklan ($)", 0, 100000, 0, step=500)
    
    st.sidebar.markdown("###")
    btn_calc = st.sidebar.button("🚀 ANALISIS SEKARANG")

# ==============================================================================
# 5. ENGINE LOGIC
# ==============================================================================
def predict_price(data_dict):
    df = pd.DataFrame([data_dict])
    df['car_age'] = 2026 - df['year']
    
    cond_map = {'salvage': 0, 'unknown': 1, 'fair': 2, 'good': 3, 'excellent': 4, 'like new': 5, 'new': 6}
    df['condition_score'] = df['condition'].map(cond_map)
    
    df = df.drop(columns=['year', 'condition'])
    df_encoded = pd.get_dummies(df)
    df_aligned = df_encoded.reindex(columns=model_cols, fill_value=0)
    
    if scaler:
        try: final_input = scaler.transform(df_aligned)
        except: final_input = df_aligned.values
    else: final_input = df_aligned.values
        
    price = model.predict(final_input)[0]
    return max(price, 0)

# ==============================================================================
# 6. DASHBOARD MAIN UI
# ==============================================================================
st.markdown("<h1 style='text-align: center;'>🤖 AutoValue Pro <span style='color:#3498db; font-size:0.6em;'>Ultimate</span></h1>", unsafe_allow_html=True)

if not btn_calc and not st.session_state['demo_run']:
    st.info("👈 Masukkan spesifikasi mobil di panel kiri. Atau klik 'Isi Data Acak' untuk demo cepat.")
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown("<div class='metric-card'><h3>📊 Market Data</h3><p>Real-time Analytics</p></div>", unsafe_allow_html=True)
    with c2: st.markdown("<div class='metric-card'><h3>🧠 AI Prediction</h3><p>XGBoost Algorithm</p></div>", unsafe_allow_html=True)
    with c3: st.markdown("<div class='metric-card'><h3>💡 Smart Deal</h3><p>Fair Price Detector</p></div>", unsafe_allow_html=True)

if btn_calc or st.session_state['demo_run']:
    # Reset demo state agar tidak auto-run terus
    st.session_state['demo_run'] = False 
    
    with st.spinner("🤖 AI sedang menghitung valuasi pasar..."):
        ai_price = predict_price(input_data)
        low_bound, high_bound = ai_price * 0.92, ai_price * 1.08
        
        # --- DEAL RATING (KILLER FEATURE) ---
        if listing_price > 0:
            diff = listing_price - ai_price
            if listing_price < low_bound:
                bg, txt = "#27ae60", f"🔥 GREAT DEAL! (Hemat ${abs(diff):,.0f})"
            elif low_bound <= listing_price <= high_bound:
                bg, txt = "#f39c12", "✅ FAIR PRICE (Harga Wajar)"
            else:
                bg, txt = "#c0392b", f"⚠️ OVERPRICED (Kemahalan ${diff:,.0f})"
            st.markdown(f"<div class='verdict-box' style='background-color: {bg};'>VONIS AI: {txt}</div>", unsafe_allow_html=True)

        # --- METRICS ---
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(f"<div class='metric-card'><h3>🏷️ Valuasi AI</h3><h2 style='color:#2980b9'>${ai_price:,.0f}</h2></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='metric-card'><h3>🛡️ Rentang Wajar</h3><h3>${low_bound:,.0f} - ${high_bound:,.0f}</h3></div>", unsafe_allow_html=True)
        with c3: 
            gap = f"${listing_price - ai_price:,.0f}" if listing_price > 0 else "-"
            st.markdown(f"<div class='metric-card'><h3>Gap Iklan</h3><h3>{gap}</h3></div>", unsafe_allow_html=True)

        st.markdown("---")
        
        # --- CHARTS ---
        t1, t2 = st.tabs(["📉 Grafik Depresiasi", "💬 Asisten Negosiasi"])
        
        with t1:
            st.subheader(f"Proyeksi Harga {input_data['manufacturer'].title()} (5 Tahun)")
            years, vals = [0, 1, 2, 3, 4], []
            for y in years:
                tmp = input_data.copy()
                tmp['year'] -= y; tmp['odometer'] += (y*15000)
                vals.append(predict_price(tmp))
            
            fig = px.line(x=[f"+{y} Thn" for y in years], y=vals, markers=True)
            fig.update_layout(xaxis_title="Waktu", yaxis_title="Estimasi Harga ($)", showlegend=False)
            fig.update_traces(line_color='#e74c3c', line_width=4)
            st.plotly_chart(fig, use_container_width=True)
            
        with t2:
            st.info("Gunakan skrip ini saat tawar-menawar:")
            buyer_script = f"""
            "Halo, saya lihat mobil {input_data['manufacturer']} tahun {input_data['year']} Anda. 
            Berdasarkan data pasar untuk kondisi {input_data['condition']}, harga wajarnya di kisaran **${ai_price:,.0f}**.
            Mengingat kilometernya sudah {input_data['odometer']:,}, saya tawar di **${low_bound:,.0f}**."
            """
            st.code(buyer_script, language="text")

else:
    if not status["model"]:
        st.error("⚠️ Sistem Error: File model tidak ditemukan.")
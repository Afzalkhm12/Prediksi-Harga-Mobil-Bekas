import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings
import random
import os

# 1. SETUP & CONFIG
warnings.filterwarnings("ignore")
st.set_page_config(page_title="AutoValue Pro", page_icon="🚘", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background-color: white; padding: 25px; border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); text-align: center;
        border-top: 5px solid #2c3e50;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-5px); }
    .verdict-box {
        padding: 20px; border-radius: 12px; text-align: center; 
        font-weight: 800; color: white; margin-bottom: 25px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1); font-size: 1.3rem;
    }
    .stButton>button {
        width: 100%; border-radius: 8px; height: 3.5em; font-weight: bold;
        transition: all 0.3s;
        background: linear-gradient(90deg, #2c3e50 0%, #34495e 100%);
        color: white; border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #34495e 0%, #2c3e50 100%);
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# 2. LOAD ASSETS
@st.cache_resource
def load_assets():
    status = {"model": False, "scaler": False}
    model, cols, ref_data, scaler = None, None, None, None
    try:
        if os.path.exists('model_final.pkl'):
            model = joblib.load('model_final.pkl'); status["model"] = True
        if os.path.exists('model_columns.pkl'): cols = joblib.load('model_columns.pkl')
        if os.path.exists('reference_data.pkl'): ref_data = joblib.load('reference_data.pkl')
        if os.path.exists('scaler.pkl'): 
            try: scaler = joblib.load('scaler.pkl'); status["scaler"] = True
            except: scaler = None
        return model, cols, ref_data, scaler, status
    except: return None, None, None, None, status

model, model_cols, ref_data, scaler, status = load_assets()

# HELPER MAPS
CONDITION_MAP_UI = {'Baru (New)': 'new', 'Sangat Bagus (Excellent)': 'excellent', 'Bagus (Good)': 'good', 'Layak (Fair)': 'fair'}
CONDITION_ORDER = ['salvage', 'fair', 'good', 'excellent', 'like new', 'new']

# 3. PREDICTION ENGINE
def predict_price(data_dict):
    if 'year' not in data_dict: return 0
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
        
    return max(model.predict(final_input)[0], 0)

# ==============================================================================
# UI DASHBOARD
# ==============================================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3202/3202926.png", width=70)
st.sidebar.title("Parameter Mobil")

# LOGIKA RANDOMIZER UTAMA
if st.sidebar.button("🎲 Isi Data Acak (Demo)"):
    st.session_state['demo_run'] = True
else:
    if 'demo_run' not in st.session_state: st.session_state['demo_run'] = False

# Helper: Fungsi untuk memilih index acak jika tombol ditekan
def get_random_index(options_list):
    if st.session_state['demo_run']:
        return random.randint(0, len(options_list) - 1)
    return 0

# Helper: Fungsi untuk angka acak
def get_random_val(default, min_v, max_v):
    if st.session_state['demo_run']:
        return random.randint(min_v, max_v)
    return default

input_data = {}

if status["model"] and ref_data:
    # 1. MERK (FULLY RANDOM)
    man_opts = ref_data.get('manufacturer', ['ford'])
    man_idx = get_random_index(man_opts)
    input_data['manufacturer'] = st.sidebar.selectbox("Merk Mobil", man_opts, index=man_idx)
    
    # 2. ANGKA (TAHUN & ODO)
    c1, c2 = st.sidebar.columns(2)
    with c1: input_data['year'] = st.number_input("Tahun Pembuatan", 1990, 2026, get_random_val(2018, 2000, 2025))
    with c2: input_data['odometer'] = st.number_input("Jarak Tempuh (Miles)", 0, 500000, get_random_val(45000, 5000, 150000), step=1000)
    
    # 3. KONDISI (FULLY RANDOM)
    cond_opts = list(CONDITION_MAP_UI.keys())
    cond_idx = get_random_index(cond_opts)
    cond_label = st.sidebar.selectbox("Kondisi Fisik", cond_opts, index=cond_idx)
    input_data['condition'] = CONDITION_MAP_UI[cond_label]
    
    # 4. SILINDER (FULLY RANDOM)
    cyl_opts = ['4 Silinder', '6 Silinder', '8 Silinder', '10 Silinder']
    cyl_idx = get_random_index(cyl_opts)
    selected_cyl = st.sidebar.selectbox("Jumlah Silinder", cyl_opts, index=cyl_idx)
    input_data['cylinders'] = selected_cyl.replace(" Silinder", " cylinders")
    
    # 5. SPESIFIKASI LAINNYA (FULLY RANDOM SEMUA)
    with st.sidebar.expander("⚙️ Spesifikasi Teknis (Advanced)"):
        # Fuel
        fuel_opts = ['gas', 'diesel', 'hybrid', 'electric']
        fuel_idx = get_random_index(fuel_opts)
        input_data['fuel'] = st.selectbox("Bahan Bakar", fuel_opts, index=fuel_idx)
        
        # Transmisi
        trans_opts = ['automatic', 'manual', 'other']
        trans_idx = get_random_index(trans_opts)
        input_data['transmission'] = st.selectbox("Transmisi", trans_opts, index=trans_idx)
        
        # Drive
        drive_opts = ['fwd', 'rwd', '4wd']
        drive_idx = get_random_index(drive_opts)
        input_data['drive'] = st.selectbox("Penggerak", drive_opts, index=drive_idx)
        
        # Tipe
        type_opts = ['sedan', 'SUV', 'truck', 'coupe', 'pickup', 'hatchback']
        type_idx = get_random_index(type_opts)
        input_data['type'] = st.selectbox("Tipe Bodi", type_opts, index=type_idx)
        
        # State
        state_opts = ['ca', 'tx', 'fl', 'ny', 'wa', 'il', 'oh']
        state_idx = get_random_index(state_opts)
        input_data['state'] = st.selectbox("Kode Wilayah", state_opts, index=state_idx)
        
        # Paint Color
        color_opts = ['white', 'black', 'silver', 'red', 'blue', 'grey']
        color_idx = get_random_index(color_opts)
        input_data['paint_color'] = st.selectbox("Warna", color_opts, index=color_idx)
        
        # Title
        title_opts = ['clean', 'rebuilt', 'salvage']
        title_idx = get_random_index(title_opts) # Biasanya clean, tapi buat demo random aja
        input_data['title_status'] = st.selectbox("Status Dokumen", title_opts, index=title_idx)

    st.sidebar.markdown("---")
    
    # OPSIONAL: INFLASI
    st.sidebar.subheader("💰 Pengaturan Pasar")
    use_inflation = st.sidebar.checkbox("📈 Sesuaikan Inflasi 2026 (+20%)", value=True)
    
    # OPSIONAL: HARGA IKLAN (JUGA DI-RANDOM JIKA DEMO)
    rand_listing = get_random_val(0, 5000, 60000)
    listing_price = st.sidebar.number_input("Harga Iklan Penjual ($)", 0, 100000, rand_listing, step=500)
    
    st.sidebar.markdown("###")
    btn_calc = st.sidebar.button("🔍 PREDIKSI HARGA PASAR")

    # OUTPUT AREA
    st.markdown("<h1 style='text-align: center; margin-bottom:10px;'>🤖 AutoValue Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color:#7f8c8d; margin-bottom:30px;'>Sistem Prediksi Harga Mobil Bekas Berbasis Machine Learning (XGBoost)</p>", unsafe_allow_html=True)
    
    if btn_calc or st.session_state['demo_run']:
        # Penting: Jangan reset demo_run di sini agar nilai form tetap bertahan saat re-run
        # st.session_state['demo_run'] = False (Dihapus agar UI tidak flicker kembali ke default)
        
        with st.spinner("Sedang menganalisis ribuan data pasar..."):
            # 1. PREDIKSI
            base_price = predict_price(input_data)
            
            # 2. INFLASI
            inflation_factor = 1.20 if use_inflation else 1.0 
            final_price = base_price * inflation_factor
            
            low_bound, high_bound = final_price * 0.92, final_price * 1.08
            
            # --- 1. VONIS BOX ---
            if listing_price > 0:
                gap_val = final_price - listing_price
                if listing_price < low_bound:
                    bg, txt, icon = "#27ae60", f"GREAT DEAL! (Hemat ${gap_val:,.0f})", "🔥"
                elif low_bound <= listing_price <= high_bound:
                    bg, txt, icon = "#f39c12", "FAIR PRICE (Harga Wajar)", "✅"
                else:
                    bg, txt, icon = "#c0392b", f"OVERPRICED (Kemahalan ${abs(gap_val):,.0f})", "⚠️"
                
                st.markdown(f"<div class='verdict-box' style='background-color: {bg};'>{icon} ANALISIS HARGA: {txt}</div>", unsafe_allow_html=True)

            # --- 2. METRICS ---
            col1, col2, col3 = st.columns(3)
            with col1:
                label_model = "🏷️ Prediksi AI (Adjusted)" if use_inflation else "🏷️ Prediksi AI (Raw 2020)"
                st.markdown(f"""<div class='metric-card'><h4 style='color:#7f8c8d; margin:0;'>{label_model}</h4><h1 style='color:#2c3e50; font-size:2.5em; margin:0;'>${final_price:,.0f}</h1></div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class='metric-card'><h4 style='color:#7f8c8d; margin:0;'>🛡️ Rentang Pasar Wajar</h4><h2 style='color:#34495e; margin:10px 0;'>${low_bound:,.0f} - ${high_bound:,.0f}</h2></div>""", unsafe_allow_html=True)
            with col3:
                if listing_price > 0:
                    sign_gap = "+" if gap_val > 0 else "-"
                    color_gap = "#27ae60" if gap_val > 0 else "#c0392b"
                    st.markdown(f"""<div class='metric-card'><h4 style='color:#7f8c8d; margin:0;'>Gap Harga</h4><h2 style='color:{color_gap}; margin:10px 0;'>{sign_gap}${abs(gap_val):,.0f}</h2></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class='metric-card'><h4 style='color:#7f8c8d; margin:0;'>Gap Harga</h4><h2 style='color:#95a5a6; margin:10px 0;'>-</h2></div>""", unsafe_allow_html=True)

            if use_inflation:
                st.info("ℹ️ **Info:** Hasil prediksi telah disesuaikan (+20%) untuk merefleksikan inflasi pasar tahun 2026 dibandingkan dataset tahun 2021.")

            st.markdown("---")

            # --- 3. ANALISIS TABS ---
            t1, t2, t3 = st.tabs(["📉 Depresiasi Aset", "💬 Strategi Negosiasi", "💡 Analisis Sensitivitas"])
            
            with t1:
                st.subheader("Proyeksi Penurunan Nilai (5 Tahun)")
                years, future_data = [0, 1, 2, 3, 4, 5], []
                for y in years:
                    tmp = input_data.copy(); tmp['year'] -= y; tmp['odometer'] += (y*15000)
                    raw_fut = predict_price(tmp)
                    adj_fut = raw_fut * inflation_factor
                    future_data.append({"Tahun": f"+{y} Thn", "Harga": adj_fut})
                
                df_fut = pd.DataFrame(future_data)
                fig_line = px.area(df_fut, x='Tahun', y='Harga', markers=True)
                fig_line.update_traces(line_color='#e74c3c', fillcolor="rgba(231, 76, 60, 0.2)")
                fig_line.update_layout(height=300)
                st.plotly_chart(fig_line, use_container_width=True)

            with t2:
                st.info("📋 **Saran AI untuk Negosiasi:**")
                if listing_price > high_bound:
                    script = f'"Harga pasar wajarnya ${final_price:,.0f}. Tawaran ${listing_price:,.0f} terlalu tinggi. Yuk ketemu di ${high_bound:,.0f}."'
                elif listing_price > 0 and listing_price < low_bound:
                    script = f'"Harganya **${listing_price:,.0f}** sudah murah. Saya siap cek unit hari ini."'
                else:
                    script = f'"Harganya **${listing_price:,.0f}** wajar. Boleh kurang dikit ke **${low_bound:,.0f}**?"'
                st.code(script, language="text")

            with t3:
                st.write("**Simulasi Penghematan:**")
                col_s1, col_s2 = st.columns(2)
                
                tmp_odo = input_data.copy(); tmp_odo['odometer'] += 20000
                p_odo = predict_price(tmp_odo) * inflation_factor
                save_odo = final_price - p_odo
                col_s1.info(f"Jika beli unit dengan KM +20k: **Hemat ${save_odo:,.0f}**")
                
                try:
                    curr_idx = CONDITION_ORDER.index(input_data['condition'])
                    if curr_idx > 0:
                        lower_cond = CONDITION_ORDER[curr_idx - 1]
                        tmp_cond = input_data.copy(); tmp_cond['condition'] = lower_cond
                        p_cond = predict_price(tmp_cond) * inflation_factor
                        save_cond = final_price - p_cond
                        col_s2.warning(f"Jika beli kondisi '{lower_cond}': **Hemat ${save_cond:,.0f}**")
                except: pass
        
        # Reset state setelah render selesai agar klik berikutnya bisa random baru lagi
        st.session_state['demo_run'] = False
        
    else:
        st.info("👈 Masukkan parameter mobil di sidebar kiri.")
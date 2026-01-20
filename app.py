import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os

# 1. SETUP AWAL
warnings.filterwarnings("ignore")
st.set_page_config(page_title="AutoValue Pro", page_icon="🚘", layout="wide")

# CSS Agar Tampilan Profesional
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background-color: white; padding: 20px; border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;
        border-top: 5px solid #2ecc71;
    }
    .error-card {
        background-color: #ffebee; padding: 20px; border-radius: 10px;
        border: 1px solid #ffcdd2; color: #c62828;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. LOAD ASSETS (DENGAN DIAGNOSIS ERROR)
# ==============================================================================
@st.cache_resource
def load_assets():
    status = {"model": False, "scaler": False, "cols": False, "ref": False}
    error_msg = ""
    
    try:
        model = joblib.load('model_final.pkl')
        status["model"] = True
        
        cols = joblib.load('model_columns.pkl')
        status["cols"] = True
        
        ref_data = joblib.load('reference_data.pkl')
        status["ref"] = True
        
        # SCALER ITU KRUSIAL UNTUK CHART GARIS LURUS
        try:
            scaler = joblib.load('scaler.pkl')
            status["scaler"] = True
        except Exception as e:
            scaler = None
            error_msg += f"\n[SCALER ERROR] {str(e)}"
            
        return model, cols, ref_data, scaler, status, error_msg
        
    except Exception as e:
        return None, None, None, None, status, str(e)

model, model_cols, ref_data, scaler, status, load_error = load_assets()

# ==============================================================================
# 3. SIDEBAR INPUT
# ==============================================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3202/3202926.png", width=80)
st.sidebar.title("Parameter Mobil")
st.sidebar.markdown("---")

input_data = {}
btn_calc = False

# Cek Kesehatan Aset Dulu
if status["model"] and status["cols"]:
    # 1. INPUT
    input_data['manufacturer'] = st.sidebar.selectbox("Merk", ref_data.get('manufacturer', ['ford']))
    input_data['year'] = st.sidebar.number_input("Tahun", 1990, 2026, 2018)
    input_data['odometer'] = st.sidebar.number_input("Odometer (Miles)", 0, 500000, 50000, step=1000)
    input_data['condition'] = st.sidebar.select_slider("Kondisi", options=['salvage', 'fair', 'good', 'excellent', 'like new', 'new'], value='good')
    
    # SILINDER: BIARKAN STRING (JANGAN DIUBAH KE ANGKA)
    cyl_opts = ['3 cylinders', '4 cylinders', '5 cylinders', '6 cylinders', '8 cylinders', '10 cylinders', '12 cylinders']
    input_data['cylinders'] = st.sidebar.selectbox("Silinder", cyl_opts, index=3)
    
    input_data['fuel'] = st.sidebar.selectbox("Bahan Bakar", ref_data.get('fuel', ['gas']))
    input_data['transmission'] = st.sidebar.selectbox("Transmisi", ref_data.get('transmission', ['automatic']))
    
    # Opsi Tambahan
    with st.sidebar.expander("Detail Lainnya"):
        input_data['drive'] = st.selectbox("Penggerak", ['4wd', 'fwd', 'rwd'], index=0)
        input_data['type'] = st.selectbox("Tipe Bodi", ['sedan', 'SUV', 'truck', 'coupe', 'pickup'], index=1)
        input_data['paint_color'] = st.selectbox("Warna", ['white', 'black', 'silver', 'red', 'blue'], index=0)
        input_data['title_status'] = st.selectbox("Status", ['clean', 'rebuilt', 'salvage'], index=0)
        input_data['size'] = st.selectbox("Ukuran", ['full-size', 'mid-size', 'compact'], index=0)
        input_data['state'] = st.selectbox("Wilayah", ['ca', 'tx', 'fl', 'ny'], index=0)

    st.sidebar.markdown("###")
    btn_calc = st.sidebar.button("ANALISIS HARGA")

else:
    st.sidebar.error("SISTEM CRITICAL ERROR: Aset tidak lengkap.")

# ==============================================================================
# 4. PREDICTION ENGINE (LOGIKA TEPAT SASARAN)
# ==============================================================================
def predict_price(data_dict):
    # 1. Buat DataFrame
    df = pd.DataFrame([data_dict])
    
    # 2. Feature Engineering (Sama persis dengan Notebook)
    df['car_age'] = 2026 - df['year']
    
    cond_map = {'salvage': 0, 'unknown': 1, 'fair': 2, 'good': 3, 'excellent': 4, 'like new': 5, 'new': 6}
    df['condition_score'] = df['condition'].map(cond_map)
    
    # Hapus kolom mentah
    df = df.drop(columns=['year', 'condition'])
    
    # 3. One-Hot Encoding (Otomatis deteksi '6 cylinders' sebagai kategori)
    df_encoded = pd.get_dummies(df)
    
    # 4. Alignment (PENTING!)
    # Paksa kolom sama dengan training. Kolom hilang diisi 0.
    df_aligned = df_encoded.reindex(columns=model_cols, fill_value=0)
    
    # 5. Scaling (KUNCI AGAR GRAFIK TIDAK LURUS)
    if scaler:
        try:
            # Pastikan urutan kolom sesuai scaler
            final_input = scaler.transform(df_aligned)
        except Exception as e:
            st.error(f"SCALER ERROR: {e}")
            final_input = df_aligned.values # Fallback (Bahaya, tapi jalan)
    else:
        # Jika scaler mati, kita pakai data mentah (Hasil pasti aneh, tapi jalan)
        final_input = df_aligned.values 
        
    # 6. Prediksi
    price = model.predict(final_input)[0]
    return max(price, 0)

# ==============================================================================
# 5. DASHBOARD UTAMA
# ==============================================================================
st.markdown("<h1 style='text-align: center;'>🤖 AutoValue Pro</h1>", unsafe_allow_html=True)

# --- PANEL DIAGNOSIS (MUNCUL JIKA ADA MASALAH) ---
if not status["scaler"]:
    st.warning("⚠️ PERINGATAN: 'scaler.pkl' TIDAK AKTIF! Prediksi akan tidak akurat & Grafik mungkin datar.")
    if load_error:
        with st.expander("Lihat Detail Error Scaler"):
            st.code(load_error)

if btn_calc and model:
    # Lakukan Prediksi
    price = predict_price(input_data)
    
    # --- HASIL UTAMA ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"<div class='metric-card'><h3>🏷️ Harga</h3><h1 style='color:#27ae60'>${price:,.0f}</h1></div>", unsafe_allow_html=True)
        
    with col2:
        low, high = price * 0.92, price * 1.08
        st.markdown(f"<div class='metric-card'><h3>⚖️ Rentang</h3><h3>${low:,.0f} - ${high:,.0f}</h3></div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"<div class='metric-card'><h3>📊 Status Data</h3><h3>{'✅ Scaled' if status['scaler'] else '⚠️ Unscaled'}</h3></div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # --- CHART DEPRESIASI ---
    st.subheader("📉 Proyeksi Harga (5 Tahun)")
    
    # Loop Simulasi
    years = [0, 1, 2, 3, 4]
    vals = []
    for y in years:
        temp = input_data.copy()
        temp['year'] -= y           # Kurangi tahun
        temp['odometer'] += (y * 15000) # Tambah KM
        p = predict_price(temp)
        vals.append(p)
        
    # Plotting
    df_chart = pd.DataFrame({'Tahun': [f"+{y} Thn" for y in years], 'Harga': vals})
    
    # Cek apakah hasil datar (Flat Line Detector)
    if max(vals) == min(vals):
        st.error("🛑 GRAFIK DATAR TERDETEKSI! Ini terjadi karena Scaler rusak/hilang. Model menganggap semua input sama.")
    
    fig = px.line(df_chart, x='Tahun', y='Harga', markers=True, title="Trend Penurunan Harga")
    fig.update_traces(line_color='#e74c3c', line_width=4)
    st.plotly_chart(fig, use_container_width=True)

else:
    if not btn_calc:
        st.info("👈 Masukkan data di sidebar untuk memulai.")
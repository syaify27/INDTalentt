import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai

# --- Konfigurasi Halaman --- 
st.set_page_config(
    page_title="TALENTNAVI - Manajemen Talenta ASN",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS Kustom untuk Tampilan Profesional ---
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"] { background-color: #0A192F; }
    [data-testid="stSidebar"] h1 { color: #FFFFFF; font-size: 24px; }
    .st-emotion-cache-183lzff { color: #0A192F; }
    #MainMenu, footer, .stDeployButton, #stDecoration { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- Pemuatan Artefak (dengan Caching) ---
ARTIFACT_DIR = Path("artifacts")
if not ARTIFACT_DIR.exists():
    st.error(f"Direktori 'artifacts' tidak ditemukan. Pastikan folder tersebut berada di direktori root aplikasi.")
    st.stop()

@st.cache_resource
def load_all_artifacts():
    try:
        model = joblib.load(ARTIFACT_DIR / "xgb_jobfit.model")
        le_pendidikan = joblib.load(ARTIFACT_DIR / "le_pendidikan.joblib")
        df_asn_full = pd.read_csv(ARTIFACT_DIR / "asn.csv")
        df_jab_full = pd.read_csv(ARTIFACT_DIR / "jabatan.csv")
        emb_df = pd.read_csv(ARTIFACT_DIR / "embeddings_node2vec.csv")
        embeddings = {row.node: row.drop("node").values for _, row in emb_df.iterrows()}
        
        explainer = None
        try:
            df_pairwise_full = pd.read_parquet(ARTIFACT_DIR / "pairwise_dataset.parquet")
            X_train = df_pairwise_full[[c for c in df_pairwise_full.columns if c not in ["id_asn","id_jabatan","jobfit_label"]]]
            explainer = shap.Explainer(model.predict, X_train)
        except Exception as e:
            st.warning(f"Gagal memuat data SHAP: {e}. Penjelasan fitur mungkin tidak akan tersedia.")
            
        return model, le_pendidikan, df_asn_full, df_jab_full, embeddings, explainer, model.feature_names_in_
    except FileNotFoundError as e:
        st.error(f"File artifact tidak ditemukan: {e}. Periksa kelengkapan file.")
        st.stop()

# --- Fungsi Helper ---
def build_feature_vector(asn_id, job_id, df_asn, df_jab, le_pend, emb_dict, model_feat_names):
    asn_row = df_asn[df_asn.id_asn == asn_id].iloc[0]
    job_row = df_jab[df_jab.id_jabatan == job_id].iloc[0]
    asn_emb = emb_dict.get(f"ASN:{asn_id}", np.zeros(64))
    job_emb = emb_dict.get(f"JOB:{job_id}", np.zeros(64))
    feature_dict = {
        "usia": asn_row.usia, "pendidikan_enc": le_pend.transform([asn_row.pendidikan])[0],
        "masa_kerja": asn_row.masa_kerja, "dim_teknis": asn_row.dim_teknis, 
        "dim_sosial": asn_row.dim_sosial, "dim_manaj": asn_row.dim_manaj,
        "jumlah_pelatihan": asn_row.jumlah_pelatihan, "w_teknis": job_row.w_teknis,
        "w_sosial": job_row.w_sosial, "w_manaj": job_row.w_manaj,
        "tech_diff": abs(asn_row.dim_teknis - job_row.w_teknis * 100),
        "social_diff": abs(asn_row.dim_sosial - job_row.w_sosial * 100),
        "manaj_diff": abs(asn_row.dim_manaj - job_row.w_manaj * 100),
        "same_unit": 1 if asn_row.unit_kerja == job_row.unit_req else 0,
    }
    for i, val in enumerate(asn_emb): feature_dict[f"asn_emb_{i}"] = val
    for i, val in enumerate(job_emb): feature_dict[f"job_emb_{i}"] = val
    return pd.Series(feature_dict)[model_feat_names]

def create_radar_chart(labels, asn_values, job_values, title):
    num_vars, angles = len(labels), np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    asn_norm = asn_values.tolist() + [asn_values[0]]
    job_norm = job_values.tolist() + [job_values[0]]
    ax.plot(angles, asn_norm, 'o-', label="Profil ASN", color='#1f77b4')
    ax.fill(angles, asn_norm, '#1f77b4', alpha=0.25)
    ax.plot(angles, job_norm, 'o-', label="Profil Jabatan", color='#ff7f0e')
    ax.fill(angles, job_norm, '#ff7f0e', alpha=0.25)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1); ax.set_rlabel_position(0)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(np.arange(0, 101, 25)); ax.set_ylim(0, 100)
    ax.set_title(title, y=1.1, fontdict={'size': 12, 'weight': 'bold'})
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    fig.tight_layout(pad=3.0)
    return fig

def generate_ai_explanation(api_key, asn_data, job_data, score, shap_values):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        shap_summary = []
        for i in range(min(5, len(shap_values.values))):
            feature_name = shap_values.feature_names[i]
            shap_val = shap_values.values[i]
            data_val = shap_values.data[i]
            impact = "meningkatkan" if shap_val > 0 else "menurunkan"
            shap_summary.append(f"- **{feature_name}** (nilai: {data_val:.2f}): Berkontribusi {impact} skor.")
        shap_text = "\n".join(shap_summary)

        prompt = f"""Anda adalah seorang Analis HR senior yang ahli dalam manajemen talenta.
        Tugas Anda adalah memberikan penjelasan naratif yang mendalam namun mudah dimengerti mengenai kecocokan seorang Aparatur Sipil Negara (ASN) untuk sebuah jabatan.
        
        Berikut adalah data yang perlu Anda analisis:
        
        **1. PROFIL ASN:**
        - Usia: {asn_data['usia']} tahun
        - Pendidikan: {asn_data['pendidikan']}
        - Masa Kerja: {asn_data['masa_kerja']} tahun
        - Unit Kerja Saat Ini: {asn_data['unit_kerja']}
        - Kompetensi Teknis: {asn_data['dim_teknis']}
        - Kompetensi Sosial: {asn_data['dim_sosial']}
        - Kompetensi Manajerial: {asn_data['dim_manaj']}

        **2. PROFIL JABATAN YANG DITUJU:**
        - Nama Jabatan: {job_data['nama_jabatan']}
        - Unit Kerja yang Dibutuhkan: {job_data['unit_req']}
        - Bobot Kompetensi Teknis: {job_data['w_teknis'] * 100}
        - Bobot Kompetensi Sosial: {job_data['w_sosial'] * 100}
        - Bobot Kompetensi Manajerial: {job_data['w_manaj'] * 100}

        **3. HASIL ANALISIS MODEL:**
        - **Skor Kecocokan (Job Fit Score): {score:.3f}** (dari skala 100)
        - **Faktor-Faktor Utama yang Mempengaruhi Skor (analisis SHAP):**
        {shap_text}

        **TUGAS ANDA:**
        Berdasarkan semua data di atas, tuliskan analisis Anda dalam format berikut:
        
        **Kesimpulan Rekomendasi:** (Berikan salah satu dari: `Sangat Direkomendasikan`, `Direkomendasikan`, `Direkomendasikan dengan Pertimbangan`, atau `Kurang Direkomendasikan`)
        
        **Analisis Komprehensif:**
        (Jelaskan secara naratif **mengapa** ASN tersebut mendapatkan skor tersebut. Hubungkan profil ASN, kebutuhan jabatan, dan temuan dari SHAP. Uraikan kekuatan utama ASN untuk posisi ini dan sebutkan potensi area pengembangan atau *gap* kompetensi yang perlu diperhatikan. Gunakan bahasa yang profesional dan konstruktif.)
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gagal menghasilkan analisis AI. Pastikan kunci API Anda valid dan coba lagi. Error: {e}"

# --- Memuat Data --- 
model, le_pendidikan, df_asn, df_jab, embeddings, explainer, model_feature_names = load_all_artifacts()

# --- Sidebar --- 
with st.sidebar:
    st.title("🇮🇩 TALENTNAVI")
    st.caption("Navigasi Talenta ASN Masa Depan")
    st.markdown("--- ")
    gemini_api_key = st.text_input("Masukkan Kunci API Gemini Anda:", type="password", help="Dapatkan kunci API dari Google AI Studio.")
    st.markdown("--- ")
    action = st.radio("PILIH MODE ANALISIS:", ("Peringkat Kandidat", "Rekomendasi Jabatan"))
    st.markdown("--- ")
    st.info("Produk dari **SIKECE TEAM** untuk **AI HACKATHON BKN 2025**.")

# --- Aplikasi Utama ---
st.title("INDONESIA DIGITAL TALENT MANAGEMENT")
st.markdown("Platform cerdas untuk membantu menemukan kesesuaian antara talenta ASN dengan kebutuhan jabatan strategis.")
st.markdown("--- ")

# --- Logika untuk Setiap Mode --- 
def run_analysis(mode):
    if mode == "Peringkat Kandidat":
        st.header("🏅 Peringkat Kandidat untuk Jabatan")
        with st.container(border=True):
            selected_id_primary = st.selectbox("**Pilih Jabatan Target:**", df_jab['id_jabatan'].unique(), format_func=lambda x: f"{x} - {df_jab[df_jab.id_jabatan == x]['nama_jabatan'].iloc[0]}")
        df_target = df_asn; id_primary_col, id_secondary_col = 'id_jabatan', 'id_asn'
    else: # Rekomendasi Jabatan
        st.header("🎯 Rekomendasi Jabatan untuk ASN")
        with st.container(border=True):
            selected_id_primary = st.selectbox("**Pilih ASN yang akan dianalisis:**", df_asn['id_asn'].unique(), format_func=lambda x: f"{x} - ASN {x.split('ASN')[-1].lstrip('0')}")
        df_target = df_jab; id_primary_col, id_secondary_col = 'id_asn', 'id_jabatan'

    if not selected_id_primary: return

    with st.spinner(f'Menghitung skor untuk semua {df_target.shape[0]} item...'):
        args = {id_primary_col: selected_id_primary}
        results = [dict(args, **{id_secondary_col: target_id, 'Job Fit Score': round(model.predict(build_feature_vector(asn_id=args.get('id_asn', target_id), job_id=args.get('id_jabatan', target_id), df_asn=df_asn, df_jab=df_jab, le_pend=le_pendidikan, emb_dict=embeddings, model_feat_names=model_feature_names).to_frame().T)[0], 3)}) for target_id in df_target[id_secondary_col]]
        df_ranking = pd.DataFrame(results).sort_values(by="Job Fit Score", ascending=False).reset_index(drop=True)
        df_ranking.index += 1
        if mode == "Peringkat Kandidat": df_ranking['Nama Kandidat'] = df_ranking['id_asn'].apply(lambda x: f"ASN {x.split('ASN')[-1].lstrip('0')}")
        else: df_ranking['Nama Jabatan'] = df_ranking['id_jabatan'].apply(lambda x: df_jab.loc[df_jab.id_jabatan == x, 'nama_jabatan'].iloc[0])
    
    st.dataframe(df_ranking, use_container_width=True)

    with st.expander("🔍 **Lihat Analisis Detail dan Penjelasan Model**"):
        options = df_ranking[id_secondary_col].unique()
        if not options.size > 0: st.warning("Tidak ada data untuk dianalisis."); return
        
        col_select, col_metric = st.columns([2, 1])
        with col_select:
            selected_id_secondary = st.selectbox(f"Pilih item dari peringkat untuk dianalisis:", options, format_func=lambda x: f"{x} (Skor: {df_ranking[df_ranking[id_secondary_col]==x]['Job Fit Score'].iloc[0]})" )
        with col_metric:
            score = df_ranking[df_ranking[id_secondary_col]==selected_id_secondary]['Job Fit Score'].iloc[0]
            st.metric(label="Job Fit Score", value=f"{score:.3f}")

        asn_id = selected_id_primary if mode == "Rekomendasi Jabatan" else selected_id_secondary
        job_id = selected_id_primary if mode == "Peringkat Kandidat" else selected_id_secondary
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_radar, col_shap = st.columns(2)
        
        asn_row = df_asn[df_asn.id_asn == asn_id].iloc[0]
        job_row = df_jab[df_jab.id_jabatan == job_id].iloc[0]

        with col_radar:
            st.subheader("Perbandingan Profil Kompetensi")
            radar_labels = ["Teknis", "Sosial", "Manajerial"]
            asn_values = np.array([asn_row.dim_teknis, asn_row.dim_sosial, asn_row.dim_manaj])
            job_values = np.array([job_row.w_teknis*100, job_row.w_sosial*100, job_row.w_manaj*100])
            fig_radar = create_radar_chart(radar_labels, asn_values, job_values, "Profil Kompetensi")
            st.pyplot(fig_radar, use_container_width=True); plt.close(fig_radar)

        with col_shap:
            st.subheader("Faktor Penentu Kecocokan")
            if explainer:
                feat_vec = build_feature_vector(asn_id, job_id, df_asn, df_jab, le_pendidikan, embeddings, model_feature_names)
                shap_values = explainer(feat_vec.to_frame().T)
                fig_waterfall, _ = plt.subplots(); shap.plots.waterfall(shap_values[0], max_display=8, show=False)
                fig_waterfall.tight_layout(); st.pyplot(fig_waterfall, use_container_width=True); plt.close(fig_waterfall)
            else: st.warning("SHAP Explainer tidak tersedia.")

        if gemini_api_key:
            st.markdown("--- ")
            st.subheader("🤖 Analisis Komprehensif oleh AI")
            with st.spinner("Gemini sedang menganalisis data, mohon tunggu..."):
                ai_explanation = generate_ai_explanation(gemini_api_key, asn_row, job_row, score, shap_values[0])
                st.markdown(ai_explanation)

# --- Menjalankan Aplikasi Sesuai Mode Pilihan ---
run_analysis(action)

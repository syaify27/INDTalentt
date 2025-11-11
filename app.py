import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

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
    /* Mengurangi padding atas container utama */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Gaya Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0A192F; /* Warna biru navy gelap */
    }
    [data-testid="stSidebar"] h1 {
        color: #FFFFFF;
        font-size: 24px;
    }
    [data-testid="stSidebar"] .st-emotion-cache-17l2y9t {
        color: #E0E0E0;
    }
    /* Gaya Judul Utama */
    .st-emotion-cache-183lzff {
        color: #0A192F; 
    }
    /* Menyembunyikan elemen bawaan Streamlit */
    #MainMenu, footer, .stDeployButton, #stDecoration {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- Direktori & Pemuatan Artefak (dengan Caching) ---
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
            feature_cols = [c for c in df_pairwise_full.columns if c not in ["id_asn","id_jabatan","jobfit_label"]]
            X_train = df_pairwise_full[feature_cols]
            explainer = shap.Explainer(model.predict, X_train)
        except Exception:
            st.warning("Gagal memuat data untuk SHAP Explainer. Penjelasan fitur mungkin tidak akan tersedia.")
            
        model_feature_names = model.feature_names_in_
        return model, le_pendidikan, df_asn_full, df_jab_full, embeddings, explainer, model_feature_names
    except FileNotFoundError as e:
        st.error(f"File artifact tidak ditemukan: {e}. Periksa kembali kelengkapan file di folder 'artifacts'.")
        st.stop()

# --- Fungsi Helper --- 
def build_feature_vector(asn_id, job_id, df_asn, df_jab, le_pend, emb_dict, model_feat_names):
    asn_row = df_asn[df_asn.id_asn == asn_id].iloc[0]
    job_row = df_jab[df_jab.id_jabatan == job_id].iloc[0]
    asn_emb = emb_dict.get(f"ASN:{asn_id}", np.zeros(64))
    job_emb = emb_dict.get(f"JOB:{job_id}", np.zeros(64))

    feature_dict = {
        "usia": asn_row.usia,
        "pendidikan_enc": le_pend.transform([asn_row.pendidikan])[0],
        "masa_kerja": asn_row.masa_kerja,
        "dim_teknis": asn_row.dim_teknis, "dim_sosial": asn_row.dim_sosial, "dim_manaj": asn_row.dim_manaj,
        "jumlah_pelatihan": asn_row.jumlah_pelatihan,
        "w_teknis": job_row.w_teknis, "w_sosial": job_row.w_sosial, "w_manaj": job_row.w_manaj,
        "tech_diff": abs(asn_row.dim_teknis - job_row.w_teknis * 100),
        "social_diff": abs(asn_row.dim_sosial - job_row.w_sosial * 100),
        "manaj_diff": abs(asn_row.dim_manaj - job_row.w_manaj * 100),
        "same_unit": 1 if asn_row.unit_kerja == job_row.unit_req else 0,
    }
    for i, val in enumerate(asn_emb): feature_dict[f"asn_emb_{i}"] = val
    for i, val in enumerate(job_emb): feature_dict[f"job_emb_{i}"] = val
    return pd.Series(feature_dict)[model_feat_names]

def create_radar_chart(labels, asn_values, job_values, title):
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    asn_values_norm = (asn_values).tolist() + [asn_values[0]]
    job_values_norm = (job_values).tolist() + [job_values[0]]

    ax.plot(angles, asn_values_norm, 'o-', linewidth=2, label="Profil ASN", color='#1f77b4')
    ax.fill(angles, asn_values_norm, '#1f77b4', alpha=0.25)
    ax.plot(angles, job_values_norm, 'o-', linewidth=2, label="Profil Jabatan", color='#ff7f0e')
    ax.fill(angles, job_values_norm, '#ff7f0e', alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(np.arange(0, 101, 25))
    ax.set_ylim(0, 100)
    ax.set_title(title, y=1.1, fontdict={'size': 12, 'weight': 'bold'})
    # Perbaikan: Mengatur posisi legenda agar tidak tumpang tindih
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    fig.tight_layout(pad=3.0)
    return fig

# --- Memuat Data --- 
model, le_pendidikan, df_asn, df_jab, embeddings, explainer, model_feature_names = load_all_artifacts()

# --- Sidebar --- 
with st.sidebar:
    st.title("🇮🇩 TALENTNAVI")
    st.caption("Navigasi Talenta ASN Masa Depan")
    st.markdown("--- ")
    action = st.radio(
        "PILIH MODE ANALISIS:",
        ("Peringkat Kandidat", "Rekomendasi Jabatan"),
        captions=("Melihat kandidat terbaik untuk satu jabatan.", "Mencari jabatan terbaik untuk satu ASN."),
    )
    st.markdown("--- ")
    st.subheader("Tentang Proyek")
    st.info("Produk dari **SIKECE TEAM** untuk **AI HACKATHON BKN 2025**. "
            "Membantu manajemen talenta ASN berbasis data.")

# --- Aplikasi Utama ---
st.title("INDONESIA DIGITAL TALENT MANAGEMENT")
st.markdown("Platform cerdas untuk membantu menemukan kesesuaian antara talenta ASN dengan kebutuhan jabatan strategis.")
st.markdown("--- ")

# === MODE 1: PERINGKAT KANDIDAT ===
if action == "Peringkat Kandidat":
    st.header("🏅 Peringkat Kandidat untuk Jabatan")
    
    with st.container(border=True):
        selected_job_id = st.selectbox(
            "**Pilih Jabatan Target:**",
            df_jab['id_jabatan'].unique(),
            format_func=lambda x: f"{x} - {df_jab[df_jab.id_jabatan == x]['nama_jabatan'].iloc[0]}",
        )
    
    if selected_job_id:
        with st.spinner('Menghitung dan membuat peringkat untuk semua kandidat...'):
            results = [{"ID ASN": asn_id, "Nama Kandidat": f"ASN {asn_id.split('ASN')[-1].lstrip('0')}", 
                        "Job Fit Score": round(model.predict(build_feature_vector(asn_id, selected_job_id, df_asn, df_jab, le_pendidikan, embeddings, model_feature_names).to_frame().T)[0], 3)}
                       for asn_id in df_asn['id_asn']]
            df_ranking = pd.DataFrame(results).sort_values(by="Job Fit Score", ascending=False).reset_index(drop=True)
            df_ranking.index += 1
        
        st.dataframe(df_ranking, use_container_width=True)

        with st.expander("🔍 **Lihat Analisis Detail dan Penjelasan Model (SHAP)**"):
            col_select, col_metric = st.columns([2, 1])
            with col_select:
                selected_asn_id = st.selectbox(
                    "Pilih ASN dari peringkat untuk dianalisis:",
                    df_ranking['ID ASN'].unique(),
                    format_func=lambda x: f"{x} (Skor: {df_ranking[df_ranking['ID ASN']==x]['Job Fit Score'].iloc[0]})",
                )
            with col_metric:
                score = df_ranking[df_ranking['ID ASN']==selected_asn_id]['Job Fit Score'].iloc[0]
                st.metric(label="Job Fit Score", value=f"{score:.3f}", help="Skor kecocokan antara 0-100")

            st.markdown("<br>", unsafe_allow_html=True)
            col_radar, col_shap = st.columns(2)
            asn_details_row = df_asn[df_asn.id_asn == selected_asn_id].iloc[0]
            current_job_row = df_jab[df_jab.id_jabatan == selected_job_id].iloc[0]

            with col_radar:
                st.subheader("Perbandingan Profil Kompetensi")
                radar_labels = ["Teknis", "Sosial", "Manajerial"]
                asn_values = np.array([asn_details_row.dim_teknis, asn_details_row.dim_sosial, asn_details_row.dim_manaj])
                job_values = np.array([current_job_row.w_teknis*100, current_job_row.w_sosial*100, current_job_row.w_manaj*100])
                fig_radar = create_radar_chart(radar_labels, asn_values, job_values, f"Profil Kompetensi")
                st.pyplot(fig_radar, use_container_width=True)
                plt.close(fig_radar)

            with col_shap:
                st.subheader("Faktor Penentu Kecocokan")
                if explainer:
                    feat_vec_shap = build_feature_vector(selected_asn_id, selected_job_id, df_asn, df_jab, le_pendidikan, embeddings, model_feature_names)
                    shap_values = explainer(feat_vec_shap.to_frame().T)
                    fig_waterfall, ax = plt.subplots()
                    shap.plots.waterfall(shap_values[0], max_display=8, show=False)
                    fig_waterfall.tight_layout()
                    st.pyplot(fig_waterfall, use_container_width=True)
                    plt.close(fig_waterfall)
                else:
                    st.warning("SHAP Explainer tidak tersedia.")

# === MODE 2: REKOMENDASI JABATAN ===
elif action == "Rekomendasi Jabatan":
    st.header("🎯 Rekomendasi Jabatan untuk ASN")

    with st.container(border=True):
        selected_asn_id_reco = st.selectbox(
            "**Pilih ASN yang akan dianalisis:**",
            df_asn['id_asn'].unique(),
            format_func=lambda x: f"{x} - ASN {x.split('ASN')[-1].lstrip('0')}",
        )

    if selected_asn_id_reco:
        with st.spinner('Menganalisis dan mencari jabatan yang paling sesuai...'):
            results = [{"ID Jabatan": job_id, "Nama Jabatan": df_jab.loc[df_jab.id_jabatan == job_id, 'nama_jabatan'].iloc[0],
                        "Job Fit Score": round(model.predict(build_feature_vector(selected_asn_id_reco, job_id, df_asn, df_jab, le_pendidikan, embeddings, model_feature_names).to_frame().T)[0], 3)}
                       for job_id in df_jab['id_jabatan']]
            df_reco = pd.DataFrame(results).sort_values(by="Job Fit Score", ascending=False).reset_index(drop=True)
            df_reco.index += 1
        
        st.dataframe(df_reco, use_container_width=True)
        
        with st.expander("🔍 **Lihat Analisis Detail dan Penjelasan Model (SHAP)**"):
            col_select, col_metric = st.columns([2, 1])
            with col_select:
                selected_job_id_reco = st.selectbox(
                    "Pilih jabatan dari rekomendasi untuk dianalisis:",
                    df_reco['ID Jabatan'].unique(),
                    format_func=lambda x: f"{x} - {df_reco[df_reco['ID Jabatan']==x]['Nama Jabatan'].iloc[0]} (Skor: {df_reco[df_reco['ID Jabatan']==x]['Job Fit Score'].iloc[0]})",
                )
            with col_metric:
                score = df_reco[df_reco['ID Jabatan']==selected_job_id_reco]['Job Fit Score'].iloc[0]
                st.metric(label="Job Fit Score", value=f"{score:.3f}", help="Skor kecocokan antara 0-100")
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_radar, col_shap = st.columns(2)
            current_asn_row = df_asn[df_asn.id_asn == selected_asn_id_reco].iloc[0]
            job_details_row = df_jab[df_jab.id_jabatan == selected_job_id_reco].iloc[0]

            with col_radar:
                st.subheader("Perbandingan Profil Kompetensi")
                radar_labels = ["Teknis", "Sosial", "Manajerial"]
                asn_values = np.array([current_asn_row.dim_teknis, current_asn_row.dim_sosial, current_asn_row.dim_manaj])
                job_values = np.array([job_details_row.w_teknis*100, job_details_row.w_sosial*100, job_details_row.w_manaj*100])
                fig_radar = create_radar_chart(radar_labels, asn_values, job_values, f"Profil Kompetensi")
                st.pyplot(fig_radar, use_container_width=True)
                plt.close(fig_radar)

            with col_shap:
                st.subheader("Faktor Penentu Kecocokan")
                if explainer:
                    feat_vec_shap = build_feature_vector(selected_asn_id_reco, selected_job_id_reco, df_asn, df_jab, le_pendidikan, embeddings, model_feature_names)
                    shap_values = explainer(feat_vec_shap.to_frame().T)
                    fig_waterfall, ax = plt.subplots()
                    shap.plots.waterfall(shap_values[0], max_display=8, show=False)
                    fig_waterfall.tight_layout()
                    st.pyplot(fig_waterfall, use_container_width=True)
                    plt.close(fig_waterfall)
                else:
                    st.warning("SHAP Explainer tidak tersedia.")

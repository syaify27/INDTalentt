import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import matplotlib.patches as mpatches

# --- Page Configuration ---
st.set_page_config(
    page_title="INDONESIA DIGITAL TALENT MANAGEMENT",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Professional Look ---
st.markdown("""
<style>
    /* Main Font */
    html, body, [class*="st-"] {
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0E1117; /* Darker sidebar */
        border-right: 2px solid #28a745; /* Green accent border */
    }
    [data-testid="stSidebar"] h1 {
        color: #FFFFFF;
        font-weight: bold;
    }
    [data-testid="stSidebar"] .st-emotion-cache-17l2y9t { /* Radio button labels */
        color: #E0E0E0;
    }

    /* Main Content Styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }

    /* Titles and Headers */
    h1, h2, h3 {
        color: #2c3e50; /* Dark blue-gray for headers */
    }
    
    .st-emotion-cache-183lzff { /* Main title color */
        color: #28a745; 
        font-weight: bold;
    }
    
    /* Expander Styling */
    .st-expander {
        border: 1px solid #DDDDDD;
        border-radius: 10px;
    }
    .st-expander header {
        font-weight: bold;
        color: #28a745;
    }
    
    /* Metric styling */
    [data-testid="stMetric"] {
        background-color: #F0F2F6;
        border: 1px solid #F0F2F6;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu, footer, .stDeployButton, #stDecoration {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# --- Artifacts Directory ---
ARTIFACT_DIR = Path("artifacts")
if not ARTIFACT_DIR.exists():
    st.error(f"Direktori artifacts tidak ditemukan di: {ARTIFACT_DIR}. Pastikan folder 'artifacts' ada di root aplikasi.")
    st.stop()

# --- Helper & Loading Functions (Cached) ---
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
        except Exception as e:
            st.warning(f"Gagal memuat data untuk SHAP Explainer: {e}. Penjelasan fitur tidak akan tersedia.")
            
        model_feature_names = model.feature_names_in_
        return model, le_pendidikan, df_asn_full, df_jab_full, embeddings, explainer, model_feature_names
    except FileNotFoundError as e:
        st.error(f"File artifact tidak ditemukan: {e}. Harap periksa folder 'artifacts'.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat artifacts: {e}")
        st.stop()

def build_feature_vector(asn_id, job_id, df_asn, df_jab, le_pend, emb_dict, model_feat_names):
    asn_row = df_asn[df_asn.id_asn == asn_id].iloc[0]
    job_row = df_jab[df_jab.id_jabatan == job_id].iloc[0]

    asn_emb = emb_dict.get(f"ASN:{asn_id}", np.zeros(64))
    job_emb = emb_dict.get(f"JOB:{job_id}", np.zeros(64))

    feature_dict = {
        "usia": asn_row.usia,
        "pendidikan_enc": le_pend.transform([asn_row.pendidikan])[0],
        "masa_kerja": asn_row.masa_kerja,
        "dim_teknis": asn_row.dim_teknis,
        "dim_sosial": asn_row.dim_sosial,
        "dim_manaj": asn_row.dim_manaj,
        "jumlah_pelatihan": asn_row.jumlah_pelatihan,
        "w_teknis": job_row.w_teknis,
        "w_sosial": job_row.w_sosial,
        "w_manaj": job_row.w_manaj,
        "tech_diff": abs(asn_row.dim_teknis - job_row.w_teknis * 100),
        "social_diff": abs(asn_row.dim_sosial - job_row.w_sosial * 100),
        "manaj_diff": abs(asn_row.dim_manaj - job_row.w_manaj * 100),
        "same_unit": 1 if asn_row.unit_kerja == job_row.unit_req else 0,
    }
    for i, val in enumerate(asn_emb): feature_dict[f"asn_emb_{i}"] = float(val)
    for i, val in enumerate(job_emb): feature_dict[f"job_emb_{i}"] = float(val)
    
    return pd.Series(feature_dict)[model_feat_names]

def create_radar_chart(labels, asn_values, job_values, title):
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    
    asn_values_norm = (asn_values / 100 * 100).tolist() + [asn_values[0]]
    job_values_norm = (job_values / 100 * 100).tolist() + [job_values[0]]

    ax.plot(angles, asn_values_norm, 'o-', linewidth=2, label="Profil ASN", color='#3498db')
    ax.fill(angles, asn_values_norm, '#3498db', alpha=0.25)
    
    ax.plot(angles, job_values_norm, 'o-', linewidth=2, label="Profil Jabatan", color='#e74c3c')
    ax.fill(angles, job_values_norm, '#e74c3c', alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(np.arange(0, 101, 25))
    ax.set_ylim(0, 100)

    ax.set_title(title, y=1.1, fontdict={'size': 12, 'weight': 'bold'})
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15))
    
    return fig

# --- Load Data ---
model, le_pendidikan, df_asn, df_jab, embeddings, explainer, model_feature_names = load_all_artifacts()

# --- Sidebar ---
with st.sidebar:
    st.title("🇮🇩 TALENTNAVI")
    st.markdown("Navigasi Talenta ASN Masa Depan")
    st.markdown("---")
    
    action = st.radio(
        "PILIH MODE ANALISIS:",
        ("Ranking Kandidat", "Rekomendasi Jabatan"),
        captions=("Peringkat kandidat untuk suatu jabatan.", "Rekomendasi jabatan untuk ASN."),
        index=0,
    )
    st.markdown("---")
    st.subheader("Tentang Proyek")
    st.info(
        "Aplikasi ini adalah produk dari **SIKECE TEAM** untuk "
        "**AI HACKATHON BKN 2025**. "
        "Dibangun untuk membantu manajemen talenta ASN yang lebih cerdas dan berbasis data."
    )

# --- Main Application ---
st.title("INDONESIA DIGITAL TALENT MANAGEMENT")
st.markdown("---")

# === PART 1: RANKING KANDIDAT ===
if action == "Ranking Kandidat":
    st.header("🏅 Peringkat Kandidat untuk Jabatan")
    
    # --- Job Selection ---
    selected_job_id = st.selectbox(
        "Pilih Jabatan Target:",
        df_jab['id_jabatan'].unique(),
        format_func=lambda x: f"{x} - {df_jab[df_jab['id_jabatan'] == x]['nama_jabatan'].iloc[0]}",
        key="ranking_job_select"
    )
    
    if selected_job_id:
        current_job_row = df_jab[df_jab['id_jabatan'] == selected_job_id].iloc[0]
        
        # --- Calculate Ranks ---
        results = []
        for asn_id in df_asn['id_asn']:
            feat_vec = build_feature_vector(asn_id, selected_job_id, df_asn, df_jab, le_pendidikan, embeddings, model_feature_names)
            score = model.predict(feat_vec.to_frame().T)[0]
            results.append({"ID ASN": asn_id, "Nama Kandidat": f"ASN {asn_id.split('ASN')[-1].lstrip('0')}", "Job Fit Score": round(score, 3)})
        
        df_ranking = pd.DataFrame(results).sort_values(by="Job Fit Score", ascending=False).reset_index(drop=True)
        df_ranking.index = df_ranking.index + 1
        
        st.dataframe(df_ranking, use_container_width=True)

        # --- Detailed Analysis Expander ---
        with st.expander("🔍 Lihat Analisis Detail & Penjelasan Model (SHAP)"):
            col_select, col_metric = st.columns([2,1])
            with col_select:
                selected_asn_id = st.selectbox(
                    "Pilih ASN dari peringkat untuk dianalisis:",
                    df_ranking['ID ASN'].unique(),
                    format_func=lambda x: f"{x} (Score: {df_ranking[df_ranking['ID ASN']==x]['Job Fit Score'].iloc[0]})",
                    key="ranking_asn_select_detail"
                )
            with col_metric:
                score = df_ranking[df_ranking['ID ASN']==selected_asn_id]['Job Fit Score'].iloc[0]
                st.metric(label="Job Fit Score", value=f"{score:.3f}")

            st.markdown("---")
            col_radar, col_shap = st.columns(2)
            
            asn_details_row = df_asn[df_asn['id_asn'] == selected_asn_id].iloc[0]
            
            with col_radar:
                st.subheader("Perbandingan Profil")
                radar_labels = ["Teknis", "Sosial", "Manajerial"]
                asn_values = np.array([asn_details_row.dim_teknis, asn_details_row.dim_sosial, asn_details_row.dim_manaj])
                job_values = np.array([current_job_row.w_teknis*100, current_job_row.w_sosial*100, current_job_row.w_manaj*100])
                fig_radar = create_radar_chart(radar_labels, asn_values, job_values, f"Profil {selected_asn_id} vs Jabatan")
                st.pyplot(fig_radar)
                plt.close(fig_radar)

            with col_shap:
                st.subheader("Faktor Penentu Kecocokan (SHAP)")
                if explainer:
                    feat_vec_shap = build_feature_vector(selected_asn_id, selected_job_id, df_asn, df_jab, le_pendidikan, embeddings, model_feature_names)
                    with st.spinner("Menghasilkan penjelasan model..."):
                        shap_values = explainer(feat_vec_shap.to_frame().T)
                        fig_waterfall, _ = plt.subplots()
                        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                        plt.tight_layout()
                        st.pyplot(fig_waterfall)
                        plt.close(fig_waterfall)
                else:
                    st.warning("SHAP Explainer tidak tersedia.")

# === PART 2: REKOMENDASI JABATAN ===
elif action == "Rekomendasi Jabatan":
    st.header("🎯 Rekomendasi Jabatan untuk ASN")

    selected_asn_id_reco = st.selectbox(
        "Pilih ASN:",
        df_asn['id_asn'].unique(),
        format_func=lambda x: f"{x} - ASN {x.split('ASN')[-1].lstrip('0')}",
        key="reco_asn_select"
    )

    if selected_asn_id_reco:
        current_asn_row = df_asn[df_asn['id_asn'] == selected_asn_id_reco].iloc[0]

        results = []
        for job_id in df_jab['id_jabatan']:
            feat_vec = build_feature_vector(selected_asn_id_reco, job_id, df_asn, df_jab, le_pendidikan, embeddings, model_feature_names)
            score = model.predict(feat_vec.to_frame().T)[0]
            results.append({"ID Jabatan": job_id, "Nama Jabatan": df_jab[df_jab.id_jabatan == job_id]['nama_jabatan'].iloc[0], "Job Fit Score": round(score, 3)})
            
        df_reco = pd.DataFrame(results).sort_values(by="Job Fit Score", ascending=False).reset_index(drop=True)
        df_reco.index = df_reco.index + 1
        
        st.dataframe(df_reco, use_container_width=True)
        
        with st.expander("🔍 Lihat Analisis Detail & Penjelasan Model (SHAP)"):
            col_select, col_metric = st.columns([2,1])
            with col_select:
                selected_job_id_reco = st.selectbox(
                    "Pilih jabatan dari rekomendasi untuk dianalisis:",
                    df_reco['ID Jabatan'].unique(),
                    format_func=lambda x: f"{x} - {df_reco[df_reco['ID Jabatan']==x]['Nama Jabatan'].iloc[0]} (Score: {df_reco[df_reco['ID Jabatan']==x]['Job Fit Score'].iloc[0]})",
                    key="reco_job_select_detail"
                )
            with col_metric:
                score = df_reco[df_reco['ID Jabatan']==selected_job_id_reco]['Job Fit Score'].iloc[0]
                st.metric(label="Job Fit Score", value=f"{score:.3f}")
            
            st.markdown("---")
            col_radar, col_shap = st.columns(2)
            
            job_details_row = df_jab[df_jab['id_jabatan'] == selected_job_id_reco].iloc[0]

            with col_radar:
                st.subheader("Perbandingan Profil")
                radar_labels = ["Teknis", "Sosial", "Manajerial"]
                asn_values = np.array([current_asn_row.dim_teknis, current_asn_row.dim_sosial, current_asn_row.dim_manaj])
                job_values = np.array([job_details_row.w_teknis*100, job_details_row.w_sosial*100, job_details_row.w_manaj*100])
                fig_radar = create_radar_chart(radar_labels, asn_values, job_values, f"Profil ASN vs {job_details_row.nama_jabatan}")
                st.pyplot(fig_radar)
                plt.close(fig_radar)
                
            with col_shap:
                st.subheader("Faktor Penentu Kecocokan (SHAP)")
                if explainer:
                    feat_vec_shap = build_feature_vector(selected_asn_id_reco, selected_job_id_reco, df_asn, df_jab, le_pendidikan, embeddings, model_feature_names)
                    with st.spinner("Menghasilkan penjelasan model..."):
                        shap_values = explainer(feat_vec_shap.to_frame().T)
                        fig_waterfall, _ = plt.subplots()
                        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                        plt.tight_layout()
                        st.pyplot(fig_waterfall)
                        plt.close(fig_waterfall)
                else:
                    st.warning("SHAP Explainer tidak tersedia.")

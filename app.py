import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from gensim.models import Word2Vec # Digunakan untuk tipe, bukan load model utuh
import matplotlib.patches as mpatches

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Sistem Manajemen Talenta ASN",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Tema Kustom ---
st.markdown('''
<style>
    .reportview-container {
        margin-top: -2em;
    }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
</style>
''', unsafe_allow_html=True)

# --- Direktori Artifacts ---
ARTIFACT_DIR = Path("artifacts")

# Pastikan direktori artifacts ada
if not ARTIFACT_DIR.exists():
    st.error(f"Direktori artifacts tidak ditemukan di: {ARTIFACT_DIR}. Pastikan Anda telah menempatkan folder 'artifacts' di root aplikasi Streamlit.")
    st.stop()

# --- Helper Functions ---
def summarize_shap_values_new(shap_explanation, feature_row, model_predict_fn=None, top_pct=0.6):
    sv = np.array(shap_explanation.values)
    feat_names = shap_explanation.feature_names
    abs_vals = np.abs(sv)
    total = abs_vals.sum()
    if total == 0:
        return "Tidak ada kontribusi fitur yang signifikan untuk prediksi ini."
    df_feat = pd.DataFrame({
        "feature": feat_names,
        "value": feature_row.values,
        "shap": sv,
        "abs_shap": abs_vals
    })
    df_feat = df_feat.sort_values("abs_shap", ascending=False)
    df_feat["cum_pct"] = df_feat["abs_shap"].cumsum() / total
    dominant = df_feat[df_feat["cum_pct"] <= top_pct]
    if dominant.empty:
        dominant = df_feat.head(3) # fallback to top 3 if top_pct yields nothing
    
    reasons = []
    for _, r in dominant.iterrows():
        fname = r.feature
        fval = r.value
        sign = "meningkatkan" if r.shap > 0 else "menurunkan"
        # Membersihkan nama fitur untuk tampilan yang lebih baik
        pretty = fname.replace("dim_","kompetensi ").replace("w_","bobot ").replace("_"," ")
        # Penanganan khusus untuk feature_name seperti 'asn_emb_0'
        if "asn_emb_" in fname or "job_emb_" in fname:
            pretty = f"embedding fitur (dim {fname.split('_')[-1]})"
            reasons.append(f"{pretty} {sign} kecocokan")
        else:
            reasons.append(f"{pretty} (nilai {np.round(fval,2)}) {sign} kecocokan")

    reasons_text = "; ".join(reasons)
    score_pred = None
    if model_predict_fn is not None:
        try:
            score_pred = float(model_predict_fn(feature_row.to_frame().T)[0])
        except Exception:
            pass # handle cases where model_predict_fn might fail
    score_text = f"Prediksi Job Fit Score ≈ {score_pred:.2f}." if score_pred is not None else ""
    sentence = f"{score_text} Faktor utama: {reasons_text}."
    return sentence

# Fungsi untuk membuat radar chart
def create_radar_chart(labels, asn_values, job_values, title="", max_val=100):
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Lengkapi lingkaran

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Plot ASN data
    if asn_values is not None:
        asn_values = (asn_values / max_val * 100).tolist() # Normalisasi ke 0-100
        asn_values += asn_values[:1]
        ax.plot(angles, asn_values, 'o-', linewidth=2, label="Profil ASN", color='skyblue')
        ax.fill(angles, asn_values, 'skyblue', alpha=0.25)
    
    # Plot Job data
    if job_values is not None:
        job_values = (job_values / max_val * 100).tolist() # Normalisasi ke 0-100
        job_values += job_values[:1]
        ax.plot(angles, job_values, 'o-', linewidth=2, label="Profil Jabatan", color='orange')
        ax.fill(angles, job_values, 'orange', alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    
    # Atur ticks dan label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(np.arange(0, 101, 25))
    ax.set_yticklabels([f"{y}%" for y in np.arange(0, 101, 25)], color="gray", size=8)
    ax.set_ylim(0, 100) # Pastikan skala 0-100

    ax.set_title(title, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    return fig

# --- Fungsi untuk memuat semua artifacts ---
@st.cache_resource
def load_all_artifacts():
    try:
        model = joblib.load(ARTIFACT_DIR / "xgb_jobfit.model")
        le_pendidikan = joblib.load(ARTIFACT_DIR / "le_pendidikan.joblib")
        # scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib") # No need, features are unscaled
        # scaler_jab = joblib.load(ARTIFACT_DIR / "scaler_jab.joblib") # No need, features are unscaled

        df_asn_full = pd.read_csv(ARTIFACT_DIR / "asn.csv")
        df_jab_full = pd.read_csv(ARTIFACT_DIR / "jabatan.csv")
        
        # Load embeddings
        emb_df = pd.read_csv(ARTIFACT_DIR / "embeddings_node2vec.csv")
        embeddings = {row.node: row.drop("node").values for _, row in emb_df.iterrows()}

        # Re-create SHAP explainer
        try:
            df_pairwise_full = pd.read_parquet(ARTIFACT_DIR / "pairwise_dataset.parquet")
            feature_cols_for_explainer = [c for c in df_pairwise_full.columns if c not in ["id_asn","id_jabatan","jobfit_label"]]
            X_train_for_explainer = df_pairwise_full[feature_cols_for_explainer]
            explainer = shap.Explainer(model.predict, X_train_for_explainer)
        except Exception as e:
            st.warning(f"Gagal memuat df_pairwise.parquet untuk SHAP Explainer: {e}. SHAP explanation mungkin tidak tersedia.")
            explainer = None
        
        # Simpan kolom fitur yang digunakan model
        model_feature_names = model.feature_names_in_

        return model, le_pendidikan, df_asn_full, df_jab_full, embeddings, explainer, model_feature_names
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat artifacts: {e}")
        st.stop()

# Load artifacts
model, le_pendidikan, df_asn, df_jab, embeddings, explainer, model_feature_names = load_all_artifacts()

# --- Fungsi untuk membangun feature vector untuk prediksi ---
def build_feature_vector(asn_id, job_id, df_asn_data, df_jab_data, le_pendidikan_obj, embeddings_dict, model_feat_names):
    asn_row = df_asn_data[df_asn_data.id_asn == asn_id].iloc[0]
    job_row = df_jab_data[df_jab_data.id_jabatan == job_id].iloc[0]

    asn_feat_dict = {
        "usia": asn_row.usia,
        "pendidikan_enc": le_pendidikan_obj.transform([asn_row.pendidikan])[0],
        "masa_kerja": asn_row.masa_kerja,
        "dim_teknis": asn_row.dim_teknis,
        "dim_sosial": asn_row.dim_sosial,
        "dim_manaj": asn_row.dim_manaj,
        "jumlah_pelatihan": asn_row.jumlah_pelatihan,
        "unit_kerja": asn_row.unit_kerja
    }
    job_feat_dict = {
        "w_teknis": job_row.w_teknis,
        "w_sosial": job_row.w_sosial,
        "w_manaj": job_row.w_manaj,
        "unit_req": job_row.unit_req
    }

    asn_node = f"ASN:{asn_id}"
    job_node = f"JOB:{job_id}"
    asn_emb = embeddings_dict.get(asn_node, np.zeros(64))
    job_emb = embeddings_dict.get(job_node, np.zeros(64))

    tech_diff = abs(asn_feat_dict["dim_teknis"] - job_feat_dict["w_teknis"] * 100)
    social_diff = abs(asn_feat_dict["dim_sosial"] - job_feat_dict["w_sosial"] * 100)
    manaj_diff = abs(asn_feat_dict["dim_manaj"] - job_feat_dict["w_manaj"] * 100)
    same_unit = 1 if asn_feat_dict["unit_kerja"] == job_feat_dict["unit_req"] else 0

    feature_dict = {
        "usia": asn_feat_dict["usia"],
        "pendidikan_enc": asn_feat_dict["pendidikan_enc"],
        "masa_kerja": asn_feat_dict["masa_kerja"],
        "dim_teknis": asn_feat_dict["dim_teknis"],
        "dim_sosial": asn_feat_dict["dim_sosial"],
        "dim_manaj": asn_feat_dict["dim_manaj"],
        "jumlah_pelatihan": asn_feat_dict["jumlah_pelatihan"],
        "w_teknis": job_feat_dict["w_teknis"],
        "w_sosial": job_feat_dict["w_sosial"],
        "w_manaj": job_feat_dict["w_manaj"],
        "tech_diff": tech_diff,
        "social_diff": social_diff,
        "manaj_diff": manaj_diff,
        "same_unit": same_unit,
    }
    for i, val in enumerate(asn_emb):
        feature_dict[f"asn_emb_{i}"] = float(val)
    for i, val in enumerate(job_emb):
        feature_dict[f"job_emb_{i}"] = float(val)
    
    feature_vector = pd.Series(feature_dict)[model_feat_names]
    return feature_vector

# --- Sidebar ---
with st.sidebar:
    st.image("https://www.streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
    st.title("Menu Navigasi")
    action = st.radio(
        "Pilih Aksi:",
        ("Ranking Kandidat", "Rekomendasi Jabatan"),
        captions=("Peringkat kandidat untuk suatu jabatan", "Rekomendasi jabatan untuk seorang kandidat")
    )
    st.markdown("---")
    st.info("Aplikasi ini membantu dalam pengambilan keputusan terkait manajemen talenta ASN.")

# --- Main App ---
st.title("💼 Sistem Manajemen Talenta ASN")
st.markdown("""
Aplikasi ini dirancang untuk membantu pengelolaan talenta Aparatur Sipil Negara (ASN)
dengan memberikan **ranking kandidat** terbaik untuk suatu jabatan dan **rekomendasi jabatan**
yang paling sesuai untuk seorang kandidat.
""")

if action == "Ranking Kandidat":
    st.header("🏅 Ranking Kandidat untuk Jabatan Tertentu")
    st.markdown("Pilih sebuah jabatan di bawah untuk melihat ASN mana yang memiliki tingkat kecocokan tertinggi.")

    col1_select_job, col2_job_desc = st.columns([1, 2])

    with col1_select_job:
        selected_job_id = st.selectbox(
            "Pilih Jabatan:",
            df_jab['id_jabatan'].unique(),
            format_func=lambda x: f"{x} - {df_jab[df_jab['id_jabatan'] == x]['nama_jabatan'].iloc[0]}",
            key="ranking_job_select"
        )
    
    if selected_job_id:
        current_job_row = df_jab[df_jab['id_jabatan'] == selected_job_id].iloc[0]
        with col2_job_desc:
            st.markdown(f"**Detail Jabatan: {current_job_row['nama_jabatan']}**")
            st.write(f"Unit Kerja yang Dibutuhkan: {current_job_row['unit_req']}")
            st.write(f"Bobot Kompetensi: Teknis={current_job_row['w_teknis']:.2f}, Sosial={current_job_row['w_sosial']:.2f}, Manajerial={current_job_row['w_manaj']:.2f}")

        st.subheader(f"Daftar ASN Paling Cocok untuk Jabatan {current_job_row['nama_jabatan']}")
        st.markdown("Hasil ranking disusun berdasarkan `Predicted Job Fit Score` dari tertinggi ke terendah.")
        
        results = []
        progress_text = "🔄 Menghitung job fit score untuk semua kandidat..."
        my_bar = st.progress(0, text=progress_text)
        
        total_asn = len(df_asn)
        for i, asn_id in enumerate(df_asn['id_asn']):
            feat_vec = build_feature_vector(asn_id, selected_job_id, df_asn, df_jab, le_pendidikan, embeddings, model_feature_names)
            pred_score = model.predict(feat_vec.to_frame().T)[0]
            results.append({
                "id_asn": asn_id,
                "nama_asn": f"ASN {asn_id.split('ASN')[-1].lstrip('0')}",
                "predicted_jobfit_score": round(pred_score, 2)
            })
            my_bar.progress((i + 1) / total_asn, text=progress_text)
        
        my_bar.empty()
        df_ranking = pd.DataFrame(results).sort_values(by="predicted_jobfit_score", ascending=False).reset_index(drop=True)
        df_ranking.index = df_ranking.index + 1

        st.dataframe(df_ranking, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Analisis Profil dan Penjelasan SHAP")
        
        col_select_asn, col_metric = st.columns([2,1])
        with col_select_asn:
            selected_asn_for_shap = st.selectbox(
                "Pilih ASN dari ranking untuk melihat detail:",
                df_ranking['id_asn'].unique(),
                format_func=lambda x: f"{x} (Score: {df_ranking[df_ranking['id_asn']==x]['predicted_jobfit_score'].iloc[0]})",
                key="ranking_asn_select"
            )
        
        with col_metric:
            score = df_ranking[df_ranking['id_asn']==selected_asn_for_shap]['predicted_jobfit_score'].iloc[0]
            st.metric(label="Predicted Job Fit Score", value=f"{score:.2f}")


        if selected_asn_for_shap:
            with st.expander("Lihat Analisis Detail"):
                col_radar_chart, col_shap = st.columns(2)
                with col_radar_chart:
                    st.markdown("##### Perbandingan Profil ASN dan Jabatan (Radar Chart)")
                    asn_details_row = df_asn[df_asn['id_asn'] == selected_asn_for_shap].iloc[0]
                    # Data untuk ASN (kompetensi inti)
                    asn_radar_labels = ["Teknis", "Sosial", "Manajerial"]
                    asn_radar_values = np.array([asn_details_row['dim_teknis'], asn_details_row['dim_sosial'], asn_details_row['dim_manaj']])
                    # Data untuk Jabatan (bobot kompetensi)
                    job_radar_labels = ["Teknis", "Sosial", "Manajerial"]
                    # Bobot jabatan 0-1, kita skalakan ke 0-100 untuk perbandingan visual
                    job_radar_values = np.array([current_job_row['w_teknis']*100, current_job_row['w_sosial']*100, current_job_row['w_manaj']*100])
                    
                    # Gunakan max_val 100 karena kompetensi dan bobot sudah dalam skala 0-100 (setelah dikali 100)
                    fig_radar = create_radar_chart(asn_radar_labels, asn_radar_values, job_radar_values,
                                                   title=f"Profil {selected_asn_for_shap} vs {current_job_row['nama_jabatan']}",
                                                   max_val=100)
                    st.pyplot(fig_radar)
                    plt.close(fig_radar)

                with col_shap:
                    st.markdown("##### Penjelasan Faktor Pendorong Job Fit (SHAP)")
                    if explainer:
                        feat_vec_shap = build_feature_vector(selected_asn_for_shap, selected_job_id, df_asn, df_jab, le_pendidikan, embeddings, model_feature_names)
                        with st.spinner("Menghasilkan penjelasan SHAP..."):
                            shap_values_obj = explainer(feat_vec_shap.to_frame().T)
                        
                        summary_text = summarize_shap_values_new(shap_values_obj[0], feat_vec_shap, model.predict)
                        st.info(summary_text)
                        
                        fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))
                        shap.plots.waterfall(shap_values_obj[0], show=False)
                        plt.tight_layout()
                        st.pyplot(fig_waterfall)
                        plt.close(fig_waterfall)

                    else:
                        st.warning("SHAP Explainer tidak tersedia. Tidak dapat menampilkan penjelasan.")


elif action == "Rekomendasi Jabatan":
    st.header("🎯 Rekomendasi Jabatan untuk Kandidat Tertentu")
    st.markdown("Pilih seorang ASN di bawah untuk melihat jabatan apa saja yang paling cocok.")

    col1_select_asn_reco, col2_asn_desc = st.columns([1, 2])

    with col1_select_asn_reco:
        selected_asn_id = st.selectbox(
            "Pilih ASN:",
            df_asn['id_asn'].unique(),
            format_func=lambda x: f"{x} - ASN {x.split('ASN')[-1].lstrip('0')}",
            key="reco_asn_select"
        )

    if selected_asn_id:
        current_asn_row = df_asn[df_asn['id_asn'] == selected_asn_id].iloc[0]
        with col2_asn_desc:
            st.markdown(f"**Profil ASN: {selected_asn_id}**")
            st.write(f"Usia: {current_asn_row['usia']} thn")
            st.write(f"Pendidikan: {current_asn_row['pendidikan']}")
            st.write(f"Masa Kerja: {current_asn_row['masa_kerja']} thn")
            st.write(f"Unit Kerja: {current_asn_row['unit_kerja']}")
            st.write(f"Kompetensi: Teknis={current_asn_row['dim_teknis']:.1f}, Sosial={current_asn_row['dim_sosial']:.1f}, Manajerial={current_asn_row['dim_manaj']:.1f}")

        st.subheader(f"Rekomendasi Jabatan untuk ASN: {selected_asn_id}")
        st.markdown("Daftar jabatan di bawah disusun berdasarkan `Predicted Job Fit Score` dari tertinggi ke terendah.")
        
        results = []
        progress_text = "🔄 Menghitung job fit score untuk semua jabatan..."
        my_bar = st.progress(0, text=progress_text)
        
        total_jobs = len(df_jab)
        for i, job_id in enumerate(df_jab['id_jabatan']):
            feat_vec = build_feature_vector(selected_asn_id, job_id, df_asn, df_jab, le_pendidikan, embeddings, model_feature_names)
            pred_score = model.predict(feat_vec.to_frame().T)[0]
            results.append({
                "id_jabatan": job_id,
                "nama_jabatan": df_jab[df_jab['id_jabatan'] == job_id]['nama_jabatan'].iloc[0],
                "predicted_jobfit_score": round(pred_score, 2)
            })
            my_bar.progress((i + 1) / total_jobs, text=progress_text)

        my_bar.empty()
        df_reco = pd.DataFrame(results).sort_values(by="predicted_jobfit_score", ascending=False).reset_index(drop=True)
        df_reco.index = df_reco.index + 1

        st.dataframe(df_reco, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Analisis Profil dan Penjelasan SHAP")

        col_select_job_reco, col_metric_reco = st.columns([2,1])
        with col_select_job_reco:
            selected_job_for_shap = st.selectbox(
                "Pilih Jabatan dari rekomendasi untuk melihat detail:",
                df_reco['id_jabatan'].unique(),
                format_func=lambda x: f"{x} - {df_reco[df_reco['id_jabatan']==x]['nama_jabatan'].iloc[0]} (Score: {df_reco[df_reco['id_jabatan']==x]['predicted_jobfit_score'].iloc[0]})",
                key="reco_job_select"
            )
        
        with col_metric_reco:
            score = df_reco[df_reco['id_jabatan']==selected_job_for_shap]['predicted_jobfit_score'].iloc[0]
            st.metric(label="Predicted Job Fit Score", value=f"{score:.2f}")

        if selected_job_for_shap:
            with st.expander("Lihat Analisis Detail"):
                col_radar_chart_reco, col_shap_reco = st.columns(2)
                with col_radar_chart_reco:
                    st.markdown("##### Perbandingan Profil ASN dan Jabatan (Radar Chart)")
                    job_details_row = df_jab[df_jab['id_jabatan'] == selected_job_for_shap].iloc[0]
                    # Data untuk ASN (kompetensi inti)
                    asn_radar_labels = ["Teknis", "Sosial", "Manajerial"]
                    asn_radar_values = np.array([current_asn_row['dim_teknis'], current_asn_row['dim_sosial'], current_asn_row['dim_manaj']])
                    # Data untuk Jabatan (bobot kompetensi)
                    job_radar_labels = ["Teknis", "Sosial", "Manajerial"]
                    job_radar_values = np.array([job_details_row['w_teknis']*100, job_details_row['w_sosial']*100, job_details_row['w_manaj']*100])
                    
                    fig_radar = create_radar_chart(asn_radar_labels, asn_radar_values, job_radar_values,
                                                   title=f"Profil {selected_asn_id} vs {job_details_row['nama_jabatan']}",
                                                   max_val=100)
                    st.pyplot(fig_radar)
                    plt.close(fig_radar)
                
                with col_shap_reco:
                    st.markdown("##### Penjelasan Faktor Pendorong Job Fit (SHAP)")
                    if explainer:
                        feat_vec_shap = build_feature_vector(selected_asn_id, selected_job_for_shap, df_asn, df_jab, le_pendidikan, embeddings, model_feature_names)
                        with st.spinner("Menghasilkan penjelasan SHAP..."):
                            shap_values_obj = explainer(feat_vec_shap.to_frame().T)
                        
                        summary_text = summarize_shap_values_new(shap_values_obj[0], feat_vec_shap, model.predict)
                        st.info(summary_text)
                        
                        fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))
                        shap.plots.waterfall(shap_values_obj[0], show=False)
                        plt.tight_layout()
                        st.pyplot(fig_waterfall)
                        plt.close(fig_waterfall)

                    else:
                        st.warning("SHAP Explainer tidak tersedia. Tidak dapat menampilkan penjelasan.")

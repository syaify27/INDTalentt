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

# --- Konfigurasi Streamlit ---
st.set_page_config(layout="wide", page_title="INDTalent")

# --- Direktori Artifacts ---
ARTIFACT_DIR = Path("artifacts")

# Pastikan direktori artifacts ada
if not ARTIFACT_DIR.exists():
    st.error(f"Direktori artifacts tidak ditemukan di: {ARTIFACT_DIR}. Pastikan Anda telah menempatkan folder 'artifacts' di root aplikasi Streamlit.")
    st.stop()

# --- Helper Functions (dari kode asli Anda) ---
def summarize_shap_values_new(shap_explanation, feature_row, model_predict_fn=None, top_pct=0.6):
    """
    shap_explanation: satu elemen dari shap.Explanation, i.e. shap_explanation[idx]
    feature_row: pd.Series of original (preprocessed) features aligned with shap_explanation.feature_names
    model_predict_fn: callable to get predicted score (optional)
    """
    sv = np.array(shap_explanation.values)   # 1D array
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

# --- Fungsi untuk memuat semua artifacts ---
@st.cache_resource
def load_all_artifacts():
    try:
        model = joblib.load(ARTIFACT_DIR / "xgb_jobfit.model")
        le_pendidikan = joblib.load(ARTIFACT_DIR / "le_pendidikan.joblib")
        scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib") # Note: original script didn't use this for pairwise DF
        scaler_jab = joblib.load(ARTIFACT_DIR / "scaler_jab.joblib") # Note: original script didn't use this for pairwise DF

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

        return model, le_pendidikan, scaler, scaler_jab, df_asn_full, df_jab_full, embeddings, explainer, model_feature_names
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat artifacts: {e}")
        st.stop()

model, le_pendidikan, scaler, scaler_jab, df_asn, df_jab, embeddings, explainer, model_feature_names = load_all_artifacts()

# --- Fungsi untuk membangun feature vector untuk prediksi ---
def build_feature_vector(asn_id, job_id, df_asn_data, df_jab_data, le_pendidikan_obj, embeddings_dict, model_feat_names):
    asn_row = df_asn_data[df_asn_data.id_asn == asn_id].iloc[0]
    job_row = df_jab_data[df_jab_data.id_jabatan == job_id].iloc[0]

    # Ambil fitur ASN (unscaled, sesuai cara df_pairwise dibuat di script asli)
    asn_feat_dict = {
        "usia": asn_row.usia,
        "pendidikan_enc": le_pendidikan_obj.transform([asn_row.pendidikan])[0],
        "masa_kerja": asn_row.masa_kerja,
        "dim_teknis": asn_row.dim_teknis,
        "dim_sosial": asn_row.dim_sosial,
        "dim_manaj": asn_row.dim_manaj,
        "jumlah_pelatihan": asn_row.jumlah_pelatihan,
        "unit_kerja": asn_row.unit_kerja # Hanya untuk same_unit
    }

    # Ambil fitur Jabatan (unscaled, sesuai cara df_pairwise dibuat di script asli)
    job_feat_dict = {
        "w_teknis": job_row.w_teknis,
        "w_sosial": job_row.w_sosial,
        "w_manaj": job_row.w_manaj,
        "unit_req": job_row.unit_req # Hanya untuk same_unit
    }

    # Node2Vec embeddings
    asn_node = f"ASN:{asn_id}"
    job_node = f"JOB:{job_id}"
    asn_emb = embeddings_dict.get(asn_node, np.zeros(64))
    job_emb = embeddings_dict.get(job_node, np.zeros(64))

    # Feature engineering: competency difference weighted
    # Penting: Gunakan nilai unscaled untuk perhitungan ini karena model dilatih dengan ini.
    tech_diff = abs(asn_feat_dict["dim_teknis"] - job_feat_dict["w_teknis"] * 100)
    social_diff = abs(asn_feat_dict["dim_sosial"] - job_feat_dict["w_sosial"] * 100)
    manaj_diff = abs(asn_feat_dict["dim_manaj"] - job_feat_dict["w_manaj"] * 100)
    same_unit = 1 if asn_feat_dict["unit_kerja"] == job_feat_dict["unit_req"] else 0

    # Gabungkan semua fitur ke dalam dictionary
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
    
    # Pastikan urutan kolom sesuai dengan yang diharapkan model
    feature_vector = pd.Series(feature_dict)[model_feat_names]
    return feature_vector

# --- Aplikasi Streamlit ---
st.title("🚀 INDTalent - Indonesia Digital Talent Management")
st.markdown("""
Aplikasi ini membantu dalam melihat ranking kandidat untuk suatu jabatan dan merekomendasikan jabatan untuk seorang kandidat, 
dilengkapi dengan penjelasan faktor-faktor penentu menggunakan SHAP.
""")

# Sidebar untuk navigasi atau input utama
st.sidebar.header("Pilihan Aksi")
action = st.sidebar.radio("Pilih Aksi:", ("Ranking Kandidat", "Rekomendasi Jabatan"))

if action == "Ranking Kandidat":
    st.header("Ranking Kandidat untuk Jabatan Tertentu")
    st.markdown("Pilih jabatan dari daftar di bawah untuk melihat ASN mana yang paling cocok.")

    # Dropdown untuk memilih jabatan
    selected_job_id = st.selectbox(
        "Pilih Jabatan:",
        df_jab['id_jabatan'].unique(),
        format_func=lambda x: f"{x} - {df_jab[df_jab['id_jabatan'] == x]['nama_jabatan'].iloc[0]}"
    )

    if selected_job_id:
        st.subheader(f"ASN Paling Cocok untuk {df_jab[df_jab['id_jabatan'] == selected_job_id]['nama_jabatan'].iloc[0]}")
        
        results = []
        progress_text = "Menghitung job fit score untuk semua kandidat..."
        my_bar = st.progress(0, text=progress_text)
        
        total_asn = len(df_asn)
        for i, asn_id in enumerate(df_asn['id_asn']):
            feat_vec = build_feature_vector(asn_id, selected_job_id, df_asn, df_jab, le_pendidikan, embeddings, model_feature_names)
            pred_score = model.predict(feat_vec.to_frame().T)[0]
            results.append({
                "id_asn": asn_id,
                "nama_asn": f"ASN {asn_id.split('ASN')[-1].lstrip('0')}", # Contoh nama
                "predicted_jobfit_score": round(pred_score, 2)
            })
            my_bar.progress((i + 1) / total_asn, text=progress_text)
        
        my_bar.empty()
        df_ranking = pd.DataFrame(results).sort_values(by="predicted_jobfit_score", ascending=False).reset_index(drop=True)
        df_ranking.index = df_ranking.index + 1 # Start index from 1

        st.dataframe(df_ranking, use_container_width=True)

        st.markdown("---")
        st.subheader("Penjelasan SHAP untuk Kandidat Pilihan")
        selected_asn_for_shap = st.selectbox(
            "Pilih ASN untuk melihat penjelasan:",
            df_ranking['id_asn'].unique(),
            format_func=lambda x: f"{x} (Score: {df_ranking[df_ranking['id_asn']==x]['predicted_jobfit_score'].iloc[0]})"
        )
        
        if selected_asn_for_shap and explainer:
            feat_vec_shap = build_feature_vector(selected_asn_for_shap, selected_job_id, df_asn, df_jab, le_pendidikan, embeddings, model_feature_names)
            with st.spinner("Menghasilkan penjelasan SHAP..."):
                shap_values_obj = explainer(feat_vec_shap.to_frame().T)
            
            st.write(f"**Penjelasan untuk ASN: {selected_asn_for_shap} dan Jabatan: {df_jab[df_jab['id_jabatan'] == selected_job_id]['nama_jabatan'].iloc[0]}**")
            summary_text = summarize_shap_values_new(shap_values_obj[0], feat_vec_shap, model.predict)
            st.info(summary_text)
            
            # Waterfall plot
            st.markdown("##### Visualisasi Faktor Pendorong (SHAP Waterfall Plot)")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values_obj[0], show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig) # Penting untuk menutup plot agar tidak tumpang tindih

        elif not explainer:
            st.warning("SHAP Explainer tidak tersedia. Tidak dapat menampilkan penjelasan.")


elif action == "Rekomendasi Jabatan":
    st.header("Rekomendasi Jabatan untuk Kandidat Tertentu")
    st.markdown("Pilih seorang ASN dari daftar di bawah untuk melihat jabatan apa saja yang paling cocok.")

    # Dropdown untuk memilih ASN
    selected_asn_id = st.selectbox(
        "Pilih ASN:",
        df_asn['id_asn'].unique(),
        format_func=lambda x: f"{x} - ASN {x.split('ASN')[-1].lstrip('0')}" # Contoh nama
    )

    if selected_asn_id:
        st.subheader(f"Jabatan yang Cocok untuk ASN: {selected_asn_id}")
        
        results = []
        progress_text = "Menghitung job fit score untuk semua jabatan..."
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
        df_reco.index = df_reco.index + 1 # Start index from 1

        st.dataframe(df_reco, use_container_width=True)

        st.markdown("---")
        st.subheader("Penjelasan SHAP untuk Jabatan Pilihan")
        selected_job_for_shap = st.selectbox(
            "Pilih Jabatan untuk melihat penjelasan:",
            df_reco['id_jabatan'].unique(),
            format_func=lambda x: f"{x} - {df_reco[df_reco['id_jabatan']==x]['nama_jabatan'].iloc[0]} (Score: {df_reco[df_reco['id_jabatan']==x]['predicted_jobfit_score'].iloc[0]})"
        )

        if selected_job_for_shap and explainer:
            feat_vec_shap = build_feature_vector(selected_asn_id, selected_job_for_shap, df_asn, df_jab, le_pendidikan, embeddings, model_feature_names)
            with st.spinner("Menghasilkan penjelasan SHAP..."):
                shap_values_obj = explainer(feat_vec_shap.to_frame().T)
            
            st.write(f"**Penjelasan untuk ASN: {selected_asn_id} dan Jabatan: {df_jab[df_jab['id_jabatan'] == selected_job_for_shap]['nama_jabatan'].iloc[0]}**")
            summary_text = summarize_shap_values_new(shap_values_obj[0], feat_vec_shap, model.predict)
            st.info(summary_text)
            
            # Waterfall plot
            st.markdown("##### Visualisasi Faktor Pendorong (SHAP Waterfall Plot)")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values_obj[0], show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig) # Penting untuk menutup plot agar tidak tumpang tindih

        elif not explainer:
            st.warning("SHAP Explainer tidak tersedia. Tidak dapat menampilkan penjelasan.")

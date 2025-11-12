*INDTalent* Indonesia Digital Talent Management 
deployment : https://indtalentt.streamlit.app/
https://ind-talent.vercel.app/

model training script : https://colab.research.google.com/drive/1Rn4EPhwL74Hda_HQKJY-GdRUV6wGnuYi?usp=sharing
see TRAINED MODEL : /artifacts/

# 🇮🇩 INDTalent: Indonesia Digital Talent Management

**Platform Cerdas untuk Manajemen Talenta ASN**

Sebuah Proyek oleh **SIKECE TEAM** untuk **AI HACKATHON BKN 2025**

---

## 📖 Deskripsi Proyek

**INDTalent** adalah aplikasi web berbasis AI yang dirancang untuk merevolusi manajemen talenta di lingkungan Aparatur Sipil Negara (ASN). Aplikasi ini menggunakan model *machine learning* untuk memprediksi dan menilai kesesuaian antara profil seorang ASN dengan kebutuhan strategis sebuah jabatan.

Tidak hanya memberikan skor, INDTalent juga memanfaatkan kekuatan Model Bahasa (LLM) dari Google Gemini untuk memberikan **analisis komprehensif** yang menjelaskan alasan di balik rekomendasi, mengidentifikasi kekuatan, dan menyoroti potensi area pengembangan bagi ASN.

## ✨ Fitur Utama

-   **Peringkat Kandidat:** Memilih satu jabatan dan mendapatkan peringkat kandidat ASN terbaik yang paling sesuai untuk posisi tersebut.
-   **Rekomendasi Jabatan:** Memilih satu ASN dan mendapatkan rekomendasi jabatan yang paling cocok berdasarkan profil dan kompetensinya.
-   **Skor Prediksi:** Memberikan skor kesesuaian kuantitatif (0-100) yang dihasilkan oleh model *machine learning* (XGBoost).
-   **Analisis oleh AI (Explainable AI):**
    -   Menggunakan **SHAP** untuk mengidentifikasi faktor-faktor utama yang memengaruhi skor.
    -   Menggunakan **Google Gemini** untuk menerjemahkan data teknis menjadi narasi analisis HR yang mendalam, profesional, dan mudah dimengerti.
-   **Antarmuka Interaktif:** Dibangun menggunakan Streamlit untuk pengalaman pengguna yang intuitif dan responsif.

## 🛠️ Teknologi yang Digunakan

-   **Backend & ML:** Python
-   **Web Framework:** Streamlit
-   **Data Manipulation:** Pandas
-   **Machine Learning Model:** XGBoost
-   **Model Explainability:** SHAP
-   **Generative AI:** Google Generative AI (Gemini)

## 🚀 Cara Menjalankan Aplikasi

### 1. Prasyarat

-   Python 3.8+
-   Akses ke terminal atau command prompt
-   API Key dari Google AI Studio (untuk Gemini)

### 2. Instalasi Dependensi

Klona repositori ini, lalu instal semua library yang dibutuhkan dengan menjalankan perintah berikut di terminal:

```bash
pip install -r requirements.txt


by *SIKECE*




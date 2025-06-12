import streamlit as st
import pandas as pd
import numpy as np
import os
import nltk
import joblib 
import gdown
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from PIL import Image

st.markdown(
    """
    <style>
    /* Background utama */
    html, body, [data-testid="stApp"], .main, .block-container {
        background-color: #fff4e6 !important;
        color: #3e2f1c;
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > textarea,
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stNumberInput > div > div > input {
        background-color: #ffe6ba !important;
        border: 1px solid #f5b971 !important;
        color: #3e2f1c !important;
        font-weight: 500;
    }

    /* Button style */
    div.stButton > button {
        background-color: #fff0c7 !important;
        color: #a04c00 !important;
        border: 2px solid #fbbf24 !important;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1.2rem;
    }

    div.stButton > button:hover {
        background-color: #fcdca4 !important;
        color: #78350f !important;
    }

    /* Expander style */
    [data-testid="stExpander"] {
        background-color: #fff9ea !important;
        border: 1px solid #f2c185;
        border-radius: 10px;
        padding: 1rem;
    }

    [data-testid="stExpander"] .stDataFrame div[data-testid="stTable"] {
        background-color: #fff9ea !important;
    }

    [data-testid="stExpander"] .stDataFrame table tbody tr:nth-child(even) {
        background-color: #fff3d6 !important;
    }

    [data-testid="stExpander"] .stDataFrame table tbody tr:nth-child(odd) {
        background-color: #ffeccf !important;
    }

    /* Header Streamlit hilang */
    header, .css-18ni7ap.e8zbici2 {
        display: none !important;
    }

    [data-testid="stAppViewContainer"] > .main {
        padding-top: 0rem !important;
    }

    /* Logo + Judul */
    .title-container {
        text-align: center;
        margin-top: 10px;
        margin-bottom: 25px;
    }

    .title-container h1 {
        font-size: 1.8em;
        font-weight: bold;
        margin-bottom: 0.2rem;
    }

    .title-container p {
        font-size: 1.1em;
        color: gray;
        margin-top: 0;
    }

    .highlight-text {
        color: #cc0000;
        font-weight: bold;
        text-transform: uppercase;
        margin-top: 10px;
    }

    .icon-row {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin-top: 20px;
        flex-wrap: wrap;
    }

    .icon-box {
        text-align: center;
    }

    .icon-label {
        font-weight: 600;
        font-size: 0.9em;
        margin-top: 4px;
    }
    </style>
    """
)
st.markdown(
    """
    <!-- Judul dan subtitle -->
    <div class="title-container">
        <img src="https://cdn-icons-png.flaticon.com/512/1828/1828884.png" width="60"/>
        <h1>Social Media Caption & Posting Analytics</h1>
        <p>Boost Your Engagement with Smart Caption Analysis and Optimal Posting Times</p>
        <p class="highlight-text">ONLY FOR ENGLISH CAPTION</p>
    </div>

    <!-- Logo Media Sosial -->
    <div class="icon-row">
        <div class="icon-box">
            <img src="https://cdn-icons-png.flaticon.com/512/733/733547.png" width="45"/>
            <div class="icon-label">Facebook</div>
        </div>
        <div class="icon-box">
            <img src="https://cdn-icons-png.flaticon.com/512/2111/2111463.png" width="45"/>
            <div class="icon-label">Instagram</div>
        </div>
        <div class="icon-box">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="45"/>
            <div class="icon-label">LinkedIn</div>
        </div>
        <div class="icon-box">
            <img src="https://github.com/error404-sudo/NewCapstone/raw/main/X.png" width="45"/>
            <div class="icon-label">X</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)



@st.cache_data
def download_and_load_data():
    file_path = "social_media_engagement_data.xlsx"
    if not os.path.exists(file_path):
        file_id = '1E4W1RvNGgyawc6I4TxQk76n289FX9kCK'
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner("📥 Mengunduh data dari Google Drive..."):
            gdown.download(url, file_path, quiet=False)

    df = pd.read_excel(file_path, sheet_name='Working File')
    return df

# --- Load dataset ---
try:
    df = download_and_load_data()
except Exception as e:
    st.error(f"❌ Gagal memuat data: {e}")
    st.stop()

# --- Preprocessing ---
cols_to_drop = [
    'Post ID', 'Date', 'Time',
    'Audience Location', 'Audience Continent', 'Audience Interests',
    'Campaign ID', 'Influencer ID', 'Weekday Type'
]
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True, errors='ignore')

for col in ['Platform', 'Post Type', 'Audience Gender', 'Age Group', 'Sentiment', 'Time Periods']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()

if 'Post Timestamp' in df.columns:
    df['Post Timestamp'] = pd.to_datetime(df['Post Timestamp'], errors='coerce')
    df['Post Hour'] = df['Post Timestamp'].dt.hour
    df['Post Day Name'] = df['Post Timestamp'].dt.day_name()

# --- Sentimen Analyzer ---
nltk.download('vader_lexicon')
vader_analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(caption):
    score = vader_analyzer.polarity_scores(caption)
    sentiment = "Positive" if score['compound'] >= 0.05 else "Negative" if score['compound'] <= -0.05 else "Neutral"
    return sentiment

def hybrid_recommendation_pipeline_super_adaptive(post_type, audience_gender, age_group, sentiment=None, platform_input=None):
    warning_text = ""
    filtered = df[
        (df['Post Type'] == post_type) &
        (df['Audience Gender'] == audience_gender) &
        (df['Age Group'] == age_group)
    ]
    if sentiment:
        filtered = filtered[filtered['Sentiment'] == sentiment]
    if platform_input.lower() != 'all':
        filtered = filtered[filtered['Platform'] == platform_input.title()]
        group_cols = ['Time Periods', 'Post Day Name', 'Post Hour']
    else:
        group_cols = ['Platform', 'Time Periods', 'Post Day Name', 'Post Hour']

    main_reco = filtered.groupby(group_cols).agg({'Engagement Rate': 'mean'}).sort_values('Engagement Rate', ascending=False).reset_index()
    if main_reco.empty:
        warning_text = "⚠️ Data terlalu sempit, fallback ke post type."
        main_reco = df[df['Post Type'] == post_type].groupby(['Platform', 'Time Periods', 'Post Day Name', 'Post Hour']).agg({'Engagement Rate': 'mean'}).sort_values('Engagement Rate', ascending=False).reset_index()
    
    return main_reco.head(5), warning_text

def strategy_recommendation(post_type, gender, age_group):
    filtered = df[(df['Post Type'] == post_type) & (df['Audience Gender'] == gender) & (df['Age Group'] == age_group)]
    strategy = filtered.groupby('Sentiment').agg({'Engagement Rate': 'mean', 'Post Content': 'count'}).rename(columns={'Post Content': 'Jumlah Post'}).sort_values('Engagement Rate', ascending=False).reset_index()
    return strategy

def alternative_platform_suggestion(post_type, gender, age_group, platform_input):
    filtered = df[(df['Post Type'] == post_type) & (df['Audience Gender'] == gender) & (df['Age Group'] == age_group)]
    if platform_input.lower() != 'all':
        filtered = filtered[filtered['Platform'] != platform_input.title()]
    alt = filtered.groupby('Platform').agg({'Engagement Rate': 'mean', 'Post Content': 'count'}).rename(columns={'Post Content': 'Jumlah Post'}).sort_values('Engagement Rate', ascending=False).reset_index()
    return alt.head(3)


@st.cache_resource
def load_model_from_file():
    model_path = "engagement_rate.pkl"
    model_file_id = '1KnVCvxqNc3qPQ6LxsYndH_RTNio9JnaU'
    model_url = f"https://drive.google.com/uc?id={model_file_id}"

    if not os.path.exists(model_path):
        with st.spinner("📥 Mengunduh model..."):
            gdown.download(model_url, model_path, quiet=False)

    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"❌ File model tidak ditemukan di path: {model_path}")
        st.stop()

# Load model ke dalam variabel
model_engage = load_model_from_file()


def estimate_engagement_from_user_input(post_type, gender, age_group, platform, sentiment):
    filtered = df[(df['Post Type'] == post_type) & (df['Audience Gender'] == gender) & (df['Age Group'] == age_group) & (df['Sentiment'] == sentiment)]
    if platform.lower() != 'all':
        filtered = filtered[filtered['Platform'] == platform.title()]
    if filtered.empty:
        return None
    top_post = filtered.sort_values(by='Engagement Rate', ascending=False).iloc[0]
    X = pd.DataFrame([{
        'Likes': top_post['Likes'], 'Comments': top_post['Comments'],
        'Shares': top_post['Shares'], 'Impressions': top_post['Impressions'], 'Reach': top_post['Reach']
    }])
    return model_engage.predict(X)[0]

# UI Simulasi


with st.form("recommendation_form"):
    caption_input = st.text_input("Masukkan Caption Anda")
    post_type_input = st.selectbox("Jenis Post:", ["Video", "Image", "Link"])
    gender_input = st.selectbox("Gender Audiens:", ["Male", "Female", "Other"])
    age_group_input = st.selectbox(
    "Kelompok Umur:  \n"
    "*_Adolescent (18–25) • Mature (26–45) • Senior (60+)_*",
    ["Senior Adults", "Mature Adults", "Adolescent Adults"]
    )
    platform_input = st.selectbox("Platform:", ["All", "Instagram", "Facebook", "Twitter", "LinkedIn"])
    submitted = st.form_submit_button("🔍 Jalankan Rekomendasi")

if submitted:
    sentiment_detected = analyze_sentiment(caption_input)
    st.success(f"✅ Sentimen Caption: **{sentiment_detected}**")

    reco, warning = hybrid_recommendation_pipeline_super_adaptive(post_type_input, gender_input, age_group_input, sentiment_detected, platform_input)
    strategy = strategy_recommendation(post_type_input, gender_input, age_group_input)
    alt_platform = alternative_platform_suggestion(post_type_input, gender_input, age_group_input, platform_input)
    estimated = estimate_engagement_from_user_input(post_type_input, gender_input, age_group_input, platform_input, sentiment_detected)

    st.markdown("---")
    st.markdown("### ⏰ Rekomendasi Waktu Posting")

    if not reco.empty:
        best = reco.iloc[0]
        st.success(f"""
            Post pada pukul **{int(best['Post Hour']):02d}:00 WIB** di hari **{best['Post Day Name']}**
            melalui platform **{best.get('Platform', 'Unknown')}** untuk engagement maksimal.
        """)

        with st.expander("📊 Lihat 5 Rekomendasi Teratas"):
            reco_display = reco[['Time Periods', 'Post Day Name', 'Post Hour', 'Engagement Rate']].copy()
            reco_display['Engagement Rate'] = (reco_display['Engagement Rate'] / 20).apply(lambda x: f"{x:.2f}%")
            st.dataframe(reco_display.head(5), use_container_width=True)

    else:
        st.warning("Tidak ada rekomendasi yang cukup relevan.")


    # --- Strategi Konten ---
    st.markdown("### 🎯 Strategi Konten")
    if not strategy.empty:
        top = strategy.iloc[0]
        st.success(f"Gunakan konten **{post_type_input.lower()}** dengan sentimen **{top['Sentiment'].lower()}** untuk **{age_group_input.lower()}**.")

        with st.expander("📊 Lihat Detail Strategi Caption"):
            strategy_display = strategy[['Sentiment', 'Engagement Rate']].copy()
            strategy_display['Engagement Rate'] = (strategy_display['Engagement Rate'] / 20).apply(lambda x: f"{x:.2f}%")
            st.dataframe(strategy_display.sort_values('Engagement Rate', ascending=False), use_container_width=True)


    else:
        st.warning("Tidak ada strategi caption ditemukan.")

    # --- Platform Alternatif ---
    st.markdown("### 🔄 Saran Platform Alternatif")
    if not alt_platform.empty:
        top_platform = alt_platform.iloc[0]["Platform"]
        st.success(f"Platform alternatif yang dapat Anda pertimbangkan: **{top_platform}**.")

        with st.expander("📊 Lihat 3 Platform Teratas"):
            alt_display = alt_platform[['Platform', 'Engagement Rate']].copy()
            alt_display['Engagement Rate'] = (alt_display['Engagement Rate'] / 20).apply(lambda x: f"{x:.2f}%")
            st.dataframe(alt_display, use_container_width=True)

    else:
        st.warning("Tidak ada platform alternatif yang disarankan.")


    st.markdown("### 📈 Estimasi Engagement Rate")
    if estimated:
        scaled_estimated = estimated / 20  # misal dibagi 20 untuk tampil lebih masuk akal
        st.success(f"Perkiraan engagement rate tertinggi: **{scaled_estimated:.2f}%**")

    else:
        st.warning("Tidak ditemukan data historis untuk estimasi engagement.")

    st.caption("📌 *Catatan: Semua nilai engagement rate hanya ditampilkan dalam skala 0–10% agar mudah dibaca.*")

    if warning:
        st.warning(warning)

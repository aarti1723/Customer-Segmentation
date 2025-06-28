import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mall Segmentation", layout="wide")
st.title("🛍️ Mall Customer Clustering App")

@st.cache_resource
def load_model():
    with open("model/kmeans_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Data Preview")
    st.write(df.head())

    if 'Annual Income (k$)' in df.columns and 'Spending Score (1-100)' in df.columns:
        X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
        df['Cluster'] = model.predict(X)

        st.subheader("🎯 Clustered Data")
        st.write(df)

        st.subheader("📊 Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster',
                        palette='Set2', data=df, s=70, ax=ax)
        st.pyplot(fig)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download Clustered CSV", csv, "clusters.csv", "text/csv")
    else:
        st.warning("❗ CSV me 'Annual Income (k$)' aur 'Spending Score (1-100)' column hona chahiye")
else:
    st.info("👈 Upload CSV file to begin.")
import streamlit as st
import pickle
import re
import pandas as pd
from sklearn.metrics import confusion_matrix

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load dataset
df = pd.read_csv("dataset.csv")

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

st.title("Language Identification System")

# -----------------------------
# 📊 1. Dataset Distribution
# -----------------------------
st.subheader("Dataset Language Distribution")

lang_counts = df["language"].value_counts()

# ✅ Streamlit chart (no seaborn/matplotlib)
st.bar_chart(lang_counts)

# -----------------------------
# 🧠 Prediction Section
# -----------------------------
st.subheader("Predict Language")

user_input = st.text_input("Enter text")

if st.button("Predict"):
    clean = preprocess(user_input)
    vec = vectorizer.transform([clean])

    prediction = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]

    st.success(f"Predicted Language: {prediction}")

    # -----------------------------
    # 📈 2. Prediction Probabilities
    # -----------------------------
    st.subheader("Prediction Confidence")

    labels = model.classes_

    prob_df = pd.DataFrame({
        "Language": labels,
        "Probability": probs
    }).set_index("Language")

    st.bar_chart(prob_df)

# -----------------------------
# 📉 3. Confusion Matrix
# -----------------------------
st.subheader("Model Confusion Matrix")

X_all = vectorizer.transform(df["text"].apply(preprocess))
y_true = df["language"]
y_pred = model.predict(X_all)

cm = confusion_matrix(y_true, y_pred, labels=model.classes_)

cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)

# ✅ Show as table instead of heatmap
st.dataframe(cm_df)
import os

if not os.path.exists("model.pkl"):
    st.error("model.pkl not found")
    st.stop()

if not os.path.exists("vectorizer.pkl"):
    st.error("vectorizer.pkl not found")
    st.stop()

if not os.path.exists("dataset.csv"):
    st.error("dataset.csv not found")
    st.stop()
    if user_input.strip() == "":
    st.warning("Please enter text")
    st.stop()
    sample_df = df.sample(min(len(df), 1000), random_state=42)

X_all = vectorizer.transform(sample_df["text"].apply(preprocess))
y_true = sample_df["language"]

import streamlit as st
import pickle
import re
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load dataset (for graphs)
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

fig1, ax1 = plt.subplots()
sns.barplot(x=lang_counts.index, y=lang_counts.values, ax=ax1)
ax1.set_ylabel("Count")
ax1.set_xlabel("Language")
st.pyplot(fig1)

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

    fig2, ax2 = plt.subplots()
    sns.barplot(x=labels, y=probs, ax=ax2)
    ax2.set_ylabel("Probability")
    ax2.set_xlabel("Language")

    st.pyplot(fig2)

# -----------------------------
# 📉 3. Confusion Matrix
# -----------------------------
st.subheader("Model Confusion Matrix")

# Generate predictions on full dataset
X_all = vectorizer.transform(df["text"].apply(preprocess))
y_true = df["language"]
y_pred = model.predict(X_all)

cm = confusion_matrix(y_true, y_pred, labels=model.classes_)

fig3, ax3 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=model.classes_,
            yticklabels=model.classes_,
            cmap="Blues", ax=ax3)

ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")

st.pyplot(fig3)

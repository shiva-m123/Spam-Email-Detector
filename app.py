import re
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="centered")

# Small fallback dataset so the app can run before you add a larger dataset.
# For a proper project evaluation, use a larger labelled email/SMS dataset.
DEMO_DATA = [
    ("ham", "Hi, are we still meeting for lunch today?"),
    ("ham", "Please find the meeting notes attached."),
    ("ham", "Your order has been shipped and will arrive tomorrow."),
    ("ham", "Can you call me when you are free?"),
    ("ham", "The assignment deadline is Friday."),
    ("ham", "Thanks for your help with the project."),
    ("ham", "Your bank statement is ready to view in the app."),
    ("ham", "I will reach the office in ten minutes."),
    ("ham", "The team meeting starts at 3 PM."),
    ("ham", "Happy birthday! Hope you have a wonderful day."),
    ("ham", "Please review the document and share your feedback."),
    ("ham", "Your appointment is confirmed for Monday morning."),
    ("ham", "Dinner is ready, come downstairs."),
    ("ham", "The report looks good. I have added two comments."),
    ("ham", "Can we reschedule our call to tomorrow?"),
    ("ham", "Your payment receipt is available in your account."),
    ("ham", "I have shared the class notes with everyone."),
    ("ham", "Remember to bring your ID card tomorrow."),
    ("ham", "Thank you for contacting customer support."),
    ("ham", "The package was delivered to your address."),
    ("spam", "Congratulations! You have won a free prize. Claim now."),
    ("spam", "URGENT: Your account is suspended. Click this link now."),
    ("spam", "You are selected for a cash reward. Send your details."),
    ("spam", "Win a brand new phone by entering this free contest."),
    ("spam", "Claim your exclusive offer before it expires."),
    ("spam", "You have won money. Call this number immediately."),
    ("spam", "Get a loan approved instantly with no documents."),
    ("spam", "Limited time offer! Buy now and get 90 percent off."),
    ("spam", "Verify your password immediately using this link."),
    ("spam", "You are our lucky winner. Claim your gift today."),
    ("spam", "Earn thousands from home. Sign up now."),
    ("spam", "Your prize is waiting. Click here to collect it."),
    ("spam", "Final warning: update your bank details now."),
    ("spam", "Free vacation available. Pay a small processing fee."),
    ("spam", "Exclusive deal! Act now before this offer ends."),
    ("spam", "You have been selected for a cash giveaway."),
    ("spam", "Congratulations, claim your reward by replying YES."),
    ("spam", "Get cheap medicines online. Order today."),
    ("spam", "Your account will be blocked unless you verify now."),
    ("spam", "Click here to get a free gift card.")
]

def normalize_label(value):
    value = str(value).strip().lower()
    if value in {"spam", "1", "junk"}:
        return "spam"
    if value in {"ham", "0", "not spam", "legitimate"}:
        return "ham"
    return None

@st.cache_resource
def train_model(data_signature):
    # data_signature is used to refresh the cached model when the data changes.
    data = load_dataset()
    X = data["text"].astype(str)
    y = data["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english",
                                  ngram_range=(1, 2), max_features=10000)),
        ("classifier", MultinomialNB())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy, y_test, predictions

def load_dataset():
    path = "spam.csv"
    try:
        raw = pd.read_csv(path, encoding="latin-1")
        # Supports the common SMS Spam Collection format (v1, v2)
        if {"v1", "v2"}.issubset(raw.columns):
            raw = raw.rename(columns={"v1": "label", "v2": "text"})
        elif not {"label", "text"}.issubset(raw.columns):
            raise ValueError("CSV must contain columns label,text or v1,v2.")
        raw = raw[["label", "text"]].dropna()
        raw["label"] = raw["label"].apply(normalize_label)
        raw = raw.dropna()
        if raw["label"].nunique() != 2:
            raise ValueError("Dataset must contain both ham and spam examples.")
        return raw
    except FileNotFoundError:
        return pd.DataFrame(DEMO_DATA, columns=["label", "text"])

st.title("📧 Spam Email Detector")
st.write("Enter an email message and the machine-learning model will classify it as **Spam** or **Ham (Not Spam)**.")

with st.sidebar:
    st.header("About the project")
    st.write("Algorithm: Multinomial Naive Bayes")
    st.write("Text features: TF-IDF")
    st.write("Language: Python")
    st.divider()
    st.caption("For a stronger model, place a larger labelled dataset named spam.csv beside app.py.")

data = load_dataset()
signature = str((len(data), int(pd.util.hash_pandas_object(data, index=True).sum())))
model, accuracy, y_test, predictions = train_model(signature)

st.subheader("Test an email")
message = st.text_area(
    "Email content",
    placeholder="Example: Congratulations! You have won a prize. Click here to claim...",
    height=150
)

if st.button("Detect Spam", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter an email message first.")
    else:
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        classes = model.named_steps["classifier"].classes_
        confidence = float(probabilities[list(classes).index(prediction)]) * 100

        if prediction == "spam":
            st.error("🚨 This message looks like SPAM.")
        else:
            st.success("✅ This message looks like HAM (NOT SPAM).")
        st.metric("Model confidence", f"{confidence:.1f}%")
        st.caption("Confidence is the model's estimated probability, not a guarantee. Always inspect suspicious links and attachments.")

with st.expander("Model evaluation"):
    st.metric("Test accuracy", f"{accuracy * 100:.1f}%")
    st.write("Classification report:")
    st.code(classification_report(y_test, predictions, zero_division=0))
    st.write("Confusion matrix (rows = actual, columns = predicted; order: ham, spam):")
    cm = confusion_matrix(y_test, predictions, labels=["ham", "spam"])
    st.dataframe(pd.DataFrame(cm, index=["Actual ham", "Actual spam"],
                              columns=["Predicted ham", "Predicted spam"]))

st.divider()
st.caption(f"Training dataset: {len(data)} messages | Ham: {(data['label'] == 'ham').sum()} | Spam: {(data['label'] == 'spam').sum()}")

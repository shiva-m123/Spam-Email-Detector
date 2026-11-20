# Spam Email Detector (Python + Machine Learning)

## Features
- Classifies a message as Spam or Ham (Not Spam)
- Uses TF-IDF text features and Multinomial Naive Bayes
- Shows a test-set accuracy, classification report and confusion matrix
- Runs with a small built-in demonstration dataset; supports a larger CSV dataset

## Setup (Windows)
1. Install Python 3.10 or later.
2. Extract this ZIP and open the project folder in VS Code.
3. Open Terminal in VS Code and run:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```

4. The app opens in your browser. Paste an email message and click **Detect Spam**.

## Use a real dataset (recommended)
For meaningful project results, download a labelled spam dataset such as the UCI SMS Spam Collection.
Save it in this folder as `spam.csv`.

Supported CSV formats:
- `label,text` (labels: `ham` / `spam`)
- `v1,v2` (the common SMS Spam Collection format)

The included fallback examples are only for demonstrating that the application runs.
Do not report the fallback dataset's accuracy as real-world performance.

## How it works
1. Text is converted into TF-IDF numeric features.
2. Multinomial Naive Bayes learns word-pattern probabilities from labelled examples.
3. The model predicts whether a new message is spam or ham.
4. Evaluation uses a held-out test split.

## Important limitation
This is an educational classifier, not a production email security gateway. It can make mistakes. Never click suspicious links or open unexpected attachments based only on its prediction.

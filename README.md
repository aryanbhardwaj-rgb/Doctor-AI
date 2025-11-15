# 🩺 AI Doctor: Symptom Checker & Triage Assistant

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-orange.svg)
![spaCy](https://img.shields.io/badge/spaCy-3.7-brightgreen.svg)

An intelligent, prototype health assistant built with Streamlit and Scikit-learn. This app takes user symptoms in natural language, predicts potential diseases with probabilities, and provides actionable triage advice.

---

## ✨ Features

* **Natural Language Symptom Entry:** Users can describe their symptoms in plain English (e.g., "I have a high fever, a bad cough, and a headache").
* **Probabilistic Diagnosis:** The app uses a Multinomial Naive Bayes model to output the top 4 most likely conditions with a confidence percentage (e.g., "Flu: 78.5%").
* **Actionable Triage:** For each potential diagnosis, the app provides a triage level (High, Medium, or Low) and clear next steps, such as "Consult nearest hospital" or "Manage at home."
* **Session-Based Health Profile:** A demo feature that allows users to enter their age, gender, and chronic conditions to simulate how a personalized model would work.
* **Diagnosis History:** The app uses Streamlit's session state to remember all diagnoses performed during the current session, which can be reviewed in the "History" tab.

---

## 🔧 Tech Stack & File Structure

This project is built with Python and relies on the following core libraries:

* **Frontend:** **Streamlit** (for the web app UI)
* **Machine Learning:** **Scikit-learn** (for training and running the `MultinomialNB` model)
* **NLP (Symptom Extraction):** **spaCy** (for parsing user input)
* **Data Handling:** **Pandas**, **Joblib** (for loading the model)

### File Structure
├── 📄 main.py # The main Streamlit application script ├── 📦 disease_model.pkl # The pre-trained scikit-learn (Naive Bayes) model ├── 📄 symptom_columns.json # "Contract" file mapping model features (symptoms) ├── 📄 symptom_precaution.csv # Database of precautions and triage advice ├── 📄 requirements.txt # List of Python dependencies └── 📄 README.md # You are here!
---

## 🚀 Setup & Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a `requirements.txt` file** with the following content:
    ```
    streamlit
    pandas
    numpy
    scikit-learn
    spacy
    requests
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the spaCy Language Model:**
    This is a **critical** step for the app to understand English.
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Run the Streamlit app:**
    ```bash
    streamlit run main.py
    ```
    Your app will now be running at `http://localhost:8501` in your browser.

---

## ⚙️ How It Works: The Data Pipeline

The app's logic flows in a clear sequence from user input to diagnosis:

1.  **Load:** When the app starts, it loads the `disease_model.pkl`, `symptom_columns.json`, `symptom_precaution.csv`, and the `spaCy` model into memory.

2.  **Input (UI):** The user types their symptoms into the `st.text_area` and clicks "Analyze."

3.  **Extract (NLP):** The `extract_symptoms` function uses `spaCy` to parse the text. It finds all known symptoms (from `symptom_list_clean`) that are present in the user's string and returns a clean list (e.g., `['high_fever', 'cough']`).

4.  **Vectorize (Model Prep):** The `get_ml_diagnosis` function creates a zero-filled vector with 130+ columns (based on `symptom_columns.json`). It then "flips" the index for `high_fever` and `cough` from 0 to 1. This 0/1 vector is the final input for the model.

5.  **Predict (ML):** The input vector is fed into `ml_model.predict_proba()`. This outputs a full list of probabilities for all 40+ diseases (e.g., `[Flu: 0.78, Cold: 0.12, ...]`).

6.  **Filter & Enrich (Data):** The app sorts these results, takes the top 4, and filters out any with a probability below 5%. It then looks up each remaining disease in the `symptom_precaution.csv` to get its triage level and advice.

7.  **Display (UI):** Streamlit dynamically renders the results in `st.tabs`, using `st.error`, `st.warning`, or `st.success` to color-code the triage warnings.

---

## ⚖️ Limitations & Future Work

This is a prototype and has several important limitations:

* **"Naive" Model:** The Multinomial Naive Bayes model assumes all symptoms are independent, which is not medically accurate (e.g., "fever" and "chills" are related).
* **Fixed Vocabulary:** The model *only* knows the symptoms in `symptom_columns.json`. It cannot understand synonyms (e.g., "dizziness" vs. "unsteadiness").
* **No Severity/Duration:** The model treats "a mild cough" and "a severe, persistent cough" as the same.
* **Demo Profile:** The "Health Profile" is only a UI demo. The model is **not** trained to use this data for a more personalized diagnosis.

### Future Improvements

* **Symptom Mapping:** Implement a "mapper" to link synonyms and general terms to the model's known symptoms (e.g., "I feel cold" -> `shivering`, `chills`).
* **Explainable AI (XAI):** Integrate **LIME** or **SHAP** to show the user *which* of their symptoms most influenced the diagnosis.
* **Real Database:** Replace the temporary `st.session_state` with a real database (like SQLite or Firebase) to enable permanent user profiles and diagnosis history.
* **Re-train Model:** Work with the Data Researcher to get a better dataset that includes age, gender, and severity, and re-train a more powerful model (e.g., Random Forest or a neural network).

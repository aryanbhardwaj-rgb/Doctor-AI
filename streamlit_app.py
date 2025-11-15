# main.py
import streamlit as st
import requests
import spacy
from bs4 import BeautifulSoup
from streamlit_lottie import st_lottie
import time
import joblib
import pandas as pd
import json


# ========== NLP MODEL (Load once) ==========
@st.cache_resource
def load_nlp_model():
    return spacy.load("en_core_web_sm")


with st.spinner("Warming up the AI Doctor's NLP... Please wait."):
    nlp = load_nlp_model()


# ========== ML MODEL & SYMPTOMS (Load once) ==========
@st.cache_resource
def load_model_and_symptoms():
    try:
        model = joblib.load("disease_model.pkl")
        with open('symptom_columns.json', 'r') as f:
            symptom_cols_model = json.load(f)
        symptom_list_clean = [s.replace('_', ' ') for s in symptom_cols_model]
        return model, symptom_cols_model, symptom_list_clean
    except FileNotFoundError:
        return None, None, None


@st.cache_resource
def load_precaution_data():
    """
    Loads the precaution data from the CSV and sets the 'Disease'
    column as the index for fast lookups.
    """
    try:
        df = pd.read_csv("symptom_precaution.csv")
        # Clean disease names to match model output (if necessary, e.g., extra spaces)
        df['Disease'] = df['Disease'].str.strip()
        df.set_index('Disease', inplace=True)
        return df
    except FileNotFoundError:
        return None


ml_model, symptom_columns_model, symptom_list_clean = load_model_and_symptoms()
precaution_df = load_precaution_data()


# ========== LOTTIE ANIMATION LOADER ==========
@st.cache_data
def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Lottie: {e}")
        return None


# ========== PROJECT DESCRIPTION ==========
project_description = """
AI Doctor (Symptom-to-Diagnosis with Explanation) is an intelligent healthcare 
assistant that takes symptoms from the user and predicts possible diseases...
"""  # (Shortened for brevity)


# ========== CORE FUNCTIONS ==========

def extract_symptoms(text, known_symptoms_list):
    """
    Extracts symptoms from text *only* if they exist in the
    model's known symptom list.
    """
    if known_symptoms_list is None:
        st.error("Model vocabulary not loaded!")
        return []

    doc = nlp(text.lower())
    found = set()
    known_symptoms_set = set(known_symptoms_list)
    text_lower = text.lower()

    for symptom in known_symptoms_set:
        if symptom in text_lower:
            found.add(symptom)

    for chunk in doc.noun_chunks:
        if chunk.text in known_symptoms_set:
            found.add(chunk.text)

    return list(sorted(found))


def get_ml_diagnosis(user_symptoms_list, precaution_data):
    """
    Uses the trained ML model to predict diseases and appends
    actionable advice and precautions.
    """
    if ml_model is None or symptom_columns_model is None:
        return None

    if precaution_data is None:
        # This check is important in case the file failed to load
        st.error("Precaution data not loaded. Triage info will be unavailable.")
        precaution_data = pd.DataFrame()  # Create empty df to avoid errors

    input_data = pd.DataFrame(0, index=[0], columns=symptom_columns_model)

    symptoms_found_in_model = 0
    for symptom in user_symptoms_list:
        s_clean = symptom.replace(' ', '_')
        if s_clean in input_data.columns:
            input_data[s_clean] = 1
            symptoms_found_in_model += 1

    if symptoms_found_in_model == 0:
        return []

    probabilities = ml_model.predict_proba(input_data)[0]

    top_4_indices = probabilities.argsort()[-4:][::-1]
    top_4_diseases = ml_model.classes_[top_4_indices]
    top_4_scores = probabilities[top_4_indices]

    results = []
    MIN_PROBABILITY = 0.15  # 15% Threshold

    for disease, score in zip(top_4_diseases, top_4_scores):
        if score > MIN_PROBABILITY:
            # --- NEW TRIAGE & PRECAUTION LOGIC ---
            precautions = []
            triage_level = "Low"  # Default
            triage_message = "These symptoms can likely be managed at home. Monitor your condition."

            # Use .strip() to remove any leading/trailing whitespace from disease name
            disease_clean = disease.strip()

            if not precaution_data.empty and disease_clean in precaution_data.index:
                row = precaution_data.loc[disease_clean]
                # Get all non-null precautions from Precaution_1, Precaution_2, etc.
                precautions = [row[col] for col in precaution_data.columns if pd.notna(row[col])]

                # Simple Triage Logic based on keywords in precautions
                precaution_text = ' '.join(precautions).lower()
                if "consult nearest hospital" in precaution_text or "call ambulance" in precaution_text:
                    triage_level = "High"
                    triage_message = "Based on these symptoms, immediate medical attention is recommended."
                elif "consult doctor" in precaution_text or "follow up" in precaution_text or "medication" in precaution_text:
                    triage_level = "Medium"
                    triage_message = "It is advisable to consult a doctor to confirm this diagnosis and discuss treatment."
                # --- END NEW LOGIC ---

            results.append({
                "disease": disease,
                "score": f"{score * 100:.2f}%",
                "summary": f"The AI model predicts a **{score * 100:.2f}%** probability for **{disease}** based on your symptoms.",
                "precautions": precautions,
                "triage_level": triage_level,
                "triage_message": triage_message
            })

    return results


# ========== UI & STYLING ==========
st.set_page_config(page_title="AI Doctor", page_icon="🩺", layout="wide")

# (Lottie URLs and CSS are unchanged)
lottie_sidebar = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_v1_xtKk4p.json")
lottie_welcome = load_lottie_url("https://assets3.lottiefiles.com/packages/lf20_q5BUPd.json")
lottie_history = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_nw1ssb8k.json")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, .main {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #E0F7FA 0%, #B2EBF2 50%, #E8F5E9 100%);
    }

    .css-1d391kg {
        background: linear-gradient(180deg, #FFFFFF 0%, #F1F8FF 100%);
        border-right: 2px solid #FFFFFF;
        box-shadow: 2px 0px 10px rgba(0,0,0,0.05);
    }

    h1 { color: #005A9C; font-weight: 700; }
    h2, h3 { color: #1E3A5F; font-weight: 600; }

    .stButton > button {
        background-color: #0072C6; color: white; border-radius: 50px;
        border: none; padding: 10px 24px; font-weight: 600;
        transition: all 0.3s ease; box-shadow: 0px 4px 12px rgba(0, 114, 198, 0.3);
    }
    .stButton > button:hover {
        background-color: #005A9C; box-shadow: 0px 6px 16px rgba(0, 90, 156, 0.4);
        transform: translateY(-2px);
    }
    .stTextArea > div > textarea {
        background-color: #FFFFFF; border: 2px solid #B0C4DE; border-radius: 12px;
        font-family: 'Poppins', sans-serif; font-size: 1.1em;
    }
    .stTextArea > div > textarea:focus {
        border-color: #0072C6; box-shadow: 0px 0px 8px rgba(0, 114, 198, 0.3);
    }

    .stTabs [role="tab"] {
        background-color: #F0F8FF; border-radius: 8px 8px 0 0;
        font-weight: 600; color: #1E3A5F;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #FFFFFF; color: #0072C6; border-bottom: 2px solid #0072C6;
    }

    .stTabs [role="tabpanel"] {
        background-color: #FFFFFF;
        color: #333333; /* <-- Dark mode UI Fix */
        border-radius: 0 0 10px 10px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        padding: 20px;
    }

    .st-expander-content {
        background-color: #FFFFFF;
        color: #333333; /* <-- Dark mode UI Fix */
        border-radius: 0 0 10px 10px;
        padding: 20px;
    }

    .symptom-box {
        background-color: #FFFFFF; border: 2px solid #0072C6; border-radius: 12px;
        padding: 16px; margin-bottom: 15px; box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
    }
    .symptom-box strong { color: #005A9C; }
    </style>
    """, unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    if lottie_sidebar:
        st_lottie(lottie_sidebar, speed=1, height=150, key="sidebar_anim")
    st.header("AI Doctor 🩺")
    st.markdown("*Your Intelligent Health Assistant*")
    nav = st.radio("Navigation", ["Symptom Checker", "History", "About this Project"])
    st.markdown("---")

    # === NEW HEALTH PROFILE SECTION ===
    with st.expander("My Health Profile (Demo)"):
        st.info("This info is only for this session and is not stored.")
        st.text_input("Your Age", key="profile_age")
        st.selectbox(
            "Your Gender",
            ["Prefer not to say", "Male", "Female", "Other"],
            key="profile_gender"
        )
        st.text_area(
            "Your Chronic Conditions (e.g., Diabetes, Hypertension)",
            key="profile_conditions",
            height=100
        )
        st.text_area(
            "Current Medications (e.g., Aspirin)",
            key="profile_meds",
            height=100
        )
        st.warning("For this prototype, please do **not** enter real personal health information.")
    # === END OF NEW SECTION ===

    st.markdown("---")
    st.success("**This is prototype 1 and soon we are gonna meet the doctor level and approved by them.**")
    st.markdown("---")
    st.info("Project by Aryan Bhardwaj, 2025")

# ========== MAIN PAGE ROUTING ==========
if "history" not in st.session_state:
    st.session_state["history"] = []

# === NEW SESSION STATE KEYS FOR PROFILE ===
if "profile_age" not in st.session_state:
    st.session_state["profile_age"] = ""
if "profile_gender" not in st.session_state:
    st.session_state["profile_gender"] = "Prefer not to say"
if "profile_conditions" not in st.session_state:
    st.session_state["profile_conditions"] = ""
if "profile_meds" not in st.session_state:
    st.session_state["profile_meds"] = ""
# === END OF NEW SECTION ===


# --- PAGE 1: SYMPTOM CHECKER ---
if nav == "Symptom Checker":

    if ml_model is None or symptom_list_clean is None:
        st.error("Fatal Error: 'disease_model.pkl' or 'symptom_columns.json' not found.")
        st.error("Please run `python Trainmodel.py` in your terminal to create these files.")
        st.stop()

    if precaution_df is None:
        st.error("Fatal Error: 'symptom_precaution.csv' not found.")
        st.error("Please make sure the file is in the same directory as main.py.")
        st.stop()

    if lottie_welcome:
        st_lottie(lottie_welcome, speed=1, height=250, key="welcome_anim")
    st.title("Symptom Checker")
    st.markdown("Please describe your symptoms. The more detail, the better our AI can analyze.")

    user_input = st.text_area("Enter symptoms (e.g., 'I have a high fever, a bad cough, and a headache'):", "",
                              height=150)

    if st.button("Analyze Symptoms", use_container_width=True, type="primary"):

        symptoms = extract_symptoms(user_input, symptom_list_clean)

        if not user_input.strip():
            st.warning("Please enter your symptoms in the box above.")

        elif not symptoms:
            st.warning("No symptoms from the model's vocabulary were detected.")
            st.info(
                "Try using specific symptoms the model knows, such as 'runny nose', 'sore throat', 'muscle aches', or 'continuous sneezing' instead of general terms like 'cold'.")

        else:
            # === NEW PROFILE CONTEXT DISPLAY ===
            st.markdown("---")
            st.subheader("Analyzing with Your Profile Context")

            # Check if any profile data has been entered
            has_profile = any([
                st.session_state.profile_age,
                st.session_state.profile_gender != "Prefer not to say",
                st.session_state.profile_conditions,
                st.session_state.profile_meds
            ])

            if has_profile:
                profile_md = ""
                if st.session_state.profile_age:
                    profile_md += f" - **Age:** {st.session_state.profile_age}\n"
                if st.session_state.profile_gender != "Prefer not to say":
                    profile_md += f" - **Gender:** {st.session_state.profile_gender}\n"
                if st.session_state.profile_conditions:
                    profile_md += f" - **Conditions:** {st.session_state.profile_conditions}\n"
                if st.session_state.profile_meds:
                    profile_md += f" - **Medications:** {st.session_state.profile_meds}\n"

                st.markdown(profile_md)
            else:
                st.markdown(
                    "No profile data entered. You can add this in the sidebar for a more personalized (demo) experience.")

            st.info("""
            **Note:** This profile information is saved for your session. 
            Our current AI model is **not** yet trained to personalize its diagnosis based on this data,
            but this demonstrates how a future version would work.
            """)
            st.markdown("---")
            # === END OF NEW SECTION ===

            st.markdown(f'<div class="symptom-box"><strong>Detected Symptoms:</strong> {", ".join(symptoms)}</div>',
                        unsafe_allow_html=True)

            with st.spinner("Analyzing... Contacting AI model..."):
                results = get_ml_diagnosis(symptoms, precaution_df)

            if results:
                st.subheader("🔬 AI-Powered Diagnosis (Prototype)")
                st.markdown("Here are the top potential conditions based on our AI model:")

                tab_titles = [f"Result {i + 1}: {res['disease']}" for i, res in enumerate(results)]
                tabs = st.tabs(tab_titles)

                for i, res in enumerate(results):
                    with tabs[i]:
                        st.subheader(f"{res['disease']} ({res['score']})")
                        st.write(res["summary"])

                        st.markdown("---")
                        st.subheader("Actionable Advice & Next Steps")

                        # 1. Display Triage Message
                        if res['triage_level'] == 'High':
                            st.error(f"**Triage Level: High.** {res['triage_message']}")
                        elif res['triage_level'] == 'Medium':
                            st.warning(f"**Triage Level: Medium.** {res['triage_message']}")
                        else:
                            st.success(f"**Triage Level: Low.** {res['triage_message']}")

                        # 2. Display Precautions
                        if res['precautions']:
                            st.markdown("**Recommended Precautions:**")
                            # Create a bulleted list
                            md_list = ""
                            for p in res['precautions']:
                                md_list += f"- {p.capitalize()}\n"
                            st.markdown(md_list)

                        # 3. Add links to official guidelines
                        st.markdown("**Further Reading:**")
                        st.markdown(f"""
                        * [Search CDC for: {res['disease']}](https://www.cdc.gov/search/?query={res['disease'].replace(' ', '%20')})
                        * [Search WHO for: {res['disease']}](https://www.who.int/search?query={res['disease'].replace(' ', '%20')})
                        """)

                        st.markdown("---")
                        st.info(
                            f"**Note:** This prediction is based on a machine learning model trained on a public dataset and is not a substitute for professional medical advice."
                        )

                st.session_state["history"].append({
                    "input": user_input, "symptoms": symptoms, "results": results
                })
            else:
                # This will now trigger if no result passes the 15% threshold
                st.error("No diseases found with high confidence for these symptoms.")
                st.info("Try adding more specific symptoms for a better analysis.")


# --- PAGE 2: HISTORY ---
elif nav == "History":
    st.header("🕓 Your Diagnosis History")
    if not st.session_state.history:
        st.info("You haven't performed any diagnoses yet.")
        if lottie_history: st_lottie(lottie_history, speed=1, height=300, key="history_anim")
    else:
        st.markdown(f"You have **{len(st.session_state.history)}** saved session(s).")
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

        for i, entry in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"**Session {i}** | Input: *'{entry['input'][:50]}...'*", expanded=False):
                st.markdown(f"**1. Your Input:**");
                st.text(entry['input'])
                st.markdown(f"**2. Detected Symptoms:**");
                st.markdown(f"`{', '.join(entry['symptoms'])}`")
                st.markdown(f"**3. Top Results:**")
                for res in entry["results"]:
                    st.success(f"**{res['disease']} ({res.get('score', 'N/A')})**: {res['summary']}")

                    # --- ADDED HISTORY DISPLAY ---
                    if 'triage_message' in res:
                        if res['triage_level'] == 'High':
                            st.error(f"**Triage:** {res['triage_message']}")
                        elif res['triage_level'] == 'Medium':
                            st.warning(f"**Triage:** {res['triage_message']}")
                        else:
                            st.success(f"**Triage:** {res['triage_message']}")

                    if 'precautions' in res and res['precautions']:
                        st.markdown("**Precautions Given:**")
                        md_list = ""
                        for p in res['precautions']:
                            md_list += f"- {p.capitalize()}\n"
                        st.markdown(md_list)
                    # --- END ADDED HISTORY DISPLAY ---

                st.markdown("---")

# --- PAGE 3: ABOUT ---
elif nav == "About this Project":
    st.header("ℹ️ About the AI Doctor")
    st.image("https://img.icons8.com/fluency/144/brain-connection.png", width=144)
    st.markdown(
        "This project, created by **Aryan Bhardwaj**, is a Streamlit-based prototype for an intelligent health assistant.")
    st.subheader("Project Vision (As Described)")
    st.info(project_description)

    st.subheader("Project Status (Prototype V2.2 - Robust Model)")
    st.warning("""
    This version now uses a **Multinomial Naive Bayes** model. This model is
    statistically better for this task and gives more *sensible* and *robust*
    answers for sparse inputs (like 1-2 symptoms) than the previous Random Forest.

    We also added a **confidence threshold** to filter out low-probability "noise".

    **Next Steps for you:**
    1.  **Symptom Mapping:** Map general words (like 'cold') to the *actual* symptoms in your model (like `['runny_nose', 'congestion']`).
    2.  **Explainable AI (XAI):** Use **LIME** or **SHAP** to show *which* of the user's symptoms most influenced the prediction.
    """)

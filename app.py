import streamlit as st
import pickle

# ================================
# Load your ML Models
# ================================
@st.cache_resource
def load_models():
    disease_model = pickle.load(open("disease_model.pkl", "rb"))
    specialist_model = pickle.load(open("specialist_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    disease_encoder = pickle.load(open("disease_encoder.pkl", "rb"))
    specialist_encoder = pickle.load(open("specialist_encoder.pkl", "rb"))
    return disease_model, specialist_model, vectorizer, disease_encoder, specialist_encoder


disease_model, specialist_model, vectorizer, disease_encoder, specialist_encoder = load_models()

# ================================
# Streamlit Page Config
# ================================
st.set_page_config(page_title="AI Healthcare Assistant", page_icon="🩺", layout="centered")
st.title("🤖 AI-Based Smart Healthcare Appointment System")

st.write("Answer a few questions and I will recommend medication or a doctor.")

# ================================
# SESSION STATE
# ================================
if "step" not in st.session_state:
    st.session_state.step = 0

if "answers" not in st.session_state:
    st.session_state.answers = {}

def reset():
    st.session_state.step = 0
    st.session_state.answers = {}
    st.experimental_rerun()


# ================================
# STEP 0: MAIN SYMPTOM
# ================================
if st.session_state.step == 0:
    symptom = st.text_input("🩹 What is your main health issue? (example: fever, headache, stomach pain)")

    if st.button("Next") and symptom.strip() != "":
        st.session_state.answers["symptom"] = symptom
        st.session_state.step = 1
        st.experimental_rerun()


# ================================
# STEP 1: NUMBER OF DAYS
# ================================
elif st.session_state.step == 1:
    days = st.number_input("📆 How many days have you had this issue?", 1, 60)

    if st.button("Next"):
        st.session_state.answers["days"] = days
        st.session_state.step = 2
        st.experimental_rerun()


# ================================
# STEP 2: SEVERITY & ML PREDICTION
# ================================
elif st.session_state.step == 2:

    symptom = st.session_state.answers["symptom"]
    days = st.session_state.answers["days"]

    st.subheader("🤖 AI Analysis")

    # Convert symptom into ML vector
    vector = vectorizer.transform([symptom])

    # Predict disease
    disease_pred = disease_model.predict(vector)[0]
    disease_name = disease_encoder.inverse_transform([disease_pred])[0]

    # Predict doctor speciality
    specialist_pred = specialist_model.predict(vector)[0]
    specialist_name = specialist_encoder.inverse_transform([specialist_pred])[0]

    # ------------------------------
    # DECISION LOGIC
    # ------------------------------
    if days <= 2:
        st.success("🟢 Your symptoms seem mild. No doctor required now.")

        st.write(f"### Recommended Tablet for **{symptom}**")
        st.write("💊 **Paracetamol 500mg** – Take 1 tablet after food (morning & night).")
        st.write("💧 Drink plenty of water and rest.")

    else:
        st.error("🔴 Your symptoms seem serious. Doctor consultation needed.")

        st.write("### Recommended Doctor")
        st.write(f"👨‍⚕️ **Specialist:** {specialist_name}")
        st.write("🏥 Nearest Hospital: Apollo Hospitals (example)")
        st.write("📞 Contact: +91 99999 88888")

    st.write("---")

    # Show predicted disease
    st.info(f"🧬 **Possible Disease:** {disease_name}")

    # Restart Option
    if st.button("🔄 Start New Diagnosis"):
        reset()

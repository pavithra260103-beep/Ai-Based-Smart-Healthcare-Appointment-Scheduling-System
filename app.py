import streamlit as st
import pickle

# ---------------------------------------------------------
# Load ML Models
# ---------------------------------------------------------
disease_model = pickle.load(open("disease_model.pkl", "rb"))
specialist_model = pickle.load(open("specialist_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
disease_encoder = pickle.load(open("disease_encoder.pkl", "rb"))
specialist_encoder = pickle.load(open("specialist_encoder.pkl", "rb"))

# ---------------------------------------------------------
# Doctor Mapping ONLY doctor name
# ---------------------------------------------------------
doctor_map = {
    "Cardiologist": "Dr. Aravind",
    "General Physician": "Dr. Kishore",
    "Dermatologist": "Dr. Priya",
    "Neurologist": "Dr. Manoj",
    "Gastroenterologist": "Dr. Suresh",
    "ENT Specialist": "Dr. Kavitha",
}

# ---------------------------------------------------------
# Medicine suggestions for minor illness
# ---------------------------------------------------------
tablet_recommend = {
    "fever": "Paracetamol 500mg – 3 times a day",
    "cold": "Cetirizine – 1 tablet at night",
    "headache": "Dolo 650mg – 1 tablet",
    "body pain": "Ibuprofen 400mg – after food",
}

# ---------------------------------------------------------
# Streamlit Session State for Chatbot
# ---------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1
    st.session_state.symptom = ""
    st.session_state.days = 0
    st.session_state.severity = ""

st.title("🤖 AI Symptom Checker & Smart Doctor Recommendation")

# ---------------------------------------------------------
# STEP 1 → ask initial symptom
# ---------------------------------------------------------
if st.session_state.step == 1:
    st.write("👋 Hello! Tell me your main symptom.")
    symptom = st.text_input("Example: fever, headache, cold, cough")

    if st.button("Next"):
        if symptom.strip() == "":
            st.warning("Please enter a symptom!")
        else:
            st.session_state.symptom = symptom
            st.session_state.step = 2
            st.rerun()

# ---------------------------------------------------------
# STEP 2 → ask how many days
# ---------------------------------------------------------
elif st.session_state.step == 2:
    st.write(f"🩺 How many days have you had **{st.session_state.symptom}**?")
    days = st.number_input("Days:", min_value=0, max_value=30, step=1)

    if st.button("Next"):
        st.session_state.days = days
        st.session_state.step = 3
        st.rerun()

# ---------------------------------------------------------
# STEP 3 → ask severity
# ---------------------------------------------------------
elif st.session_state.step == 3:
    st.write("⚠️ How severe is the issue?")
    severity = st.selectbox("Select severity level", ["mild", "moderate", "severe"])

    if st.button("Get Result"):
        st.session_state.severity = severity
        st.session_state.step = 4
        st.rerun()

# ---------------------------------------------------------
# STEP 4 → Final Decision Logic + ML Prediction
# ---------------------------------------------------------
elif st.session_state.step == 4:
    st.subheader("🧠 AI Decision Result")

    symptom = st.session_state.symptom
    days = st.session_state.days
    severity = st.session_state.severity

    # RULE-BASED LOGIC
    if severity == "mild" and days <= 2:
        # Tablet Recommendation
        if symptom in tablet_recommend:
            st.success(f"💊 **Recommended Tablet for {symptom}:** {tablet_recommend[symptom]}")
        else:
            st.info("This seems mild. Try rest, water, and basic medicine.")
        st.stop()

    # If not mild → use ML model
    X = vectorizer.transform([symptom])
    disease_pred = disease_model.predict(X)[0]
    specialist_pred = specialist_model.predict(X)[0]

    disease = disease_encoder.inverse_transform([disease_pred])[0]
    specialist = specialist_encoder.inverse_transform([specialist_pred])[0]

    st.error(f"🧬 Possible Disease: **{disease}**")
    st.warning(f"👨‍⚕️ Required Specialist: **{specialist}**")

    # Show doctor name
    if specialist in doctor_map:
        st.success(f"⭐ Recommended Doctor: **{doctor_map[specialist]}**")
    else:
        st.info("Specialist available, but doctor name missing.")

    # Reset button
    if st.button("Start Over"):
        st.session_state.step = 1
        st.rerun()

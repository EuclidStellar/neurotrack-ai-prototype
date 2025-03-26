import streamlit as st
import google.generativeai as genai
from crewai import Crew, Agent, Task
import pandas as pd
import random
import os
from fpdf import FPDF
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

questions = {
    "social_interactions": [
        "Do you feel comfortable talking to new people or being in groups?",
        "Do you often find it hard to figure out what others are feeling or thinking?",
        "How easy is it for you to make and keep friends?"
    ],
    "communication": [
        "Do you have trouble finding the right words to say what you mean?",
        "How often do you need others to repeat themselves so you can understand them?",
        "Do you like writing your thoughts (like texting or emailing) more than speaking them?"
    ],
    "attention_focus": [
        "Do you often lose track of what you’re doing or forget what you were going to say?",
        "How hard is it for you to follow instructions with more than one step?",
        "Do noises or movements around you make it tough to focus?"
    ],
    "hyperactivity_impulsivity": [
        "Do you feel restless or find it hard to sit still for a long time?",
        "How often do you interrupt others when they’re talking?",
        "Do you often make choices fast without thinking about what might happen?"
    ],
    "sensory_experiences": [
        "Are there sounds, lights, or smells that bother you more than they bother others?",
        "Do you dislike certain textures in clothes, food, or other things?",
        "How much do bright lights or loud places bother you?"
    ],
    "learning_academics": [
        "Do you find any of these hard compared to others your age? (Reading, Writing, Math)",
        "How often do you need extra time to finish tasks or schoolwork?"
    ],
    "emotional_regulation": [
        "Do you get really upset when your day does not go as planned?",
        "How well do you deal with surprises or changes in your routine?",
        "Do strong feelings like anger, sadness, or excitement feel hard to control?"
    ]
}

def generate_patient_responses(num_patients=5):
    responses = []
    for patient_id in range(1, num_patients + 1):
        patient_response = {
            "id": patient_id,
            "social_interactions": random.randint(1, 4),
            "communication": random.randint(1, 4),
            "attention_focus": random.randint(1, 4),
            "hyperactivity_impulsivity": random.randint(1, 4),
            "sensory_experiences": random.randint(1, 4),
            "learning_academics": random.randint(1, 4),
            "emotional_regulation": random.randint(1, 4)
        }
        responses.append(patient_response)
    return responses

def clean_text_for_pdf(text):
    clean = text.replace("**", "").replace("*", "").replace("#", "")
    lines = clean.split('\n')
    filtered_lines = []
    for line in lines:
        if not any(line.strip().startswith(x) for x in ["Note:", "- Note:", "Based on", "Given the", "It's impossible", "Different methods", "Furthermore", "However", "Example IEP Components", "I'll suggest"]):
            filtered_lines.append(line)
    clean = '\n'.join(filtered_lines)
    while "\n\n\n" in clean:
        clean = clean.replace("\n\n\n", "\n\n")
    clean = clean.replace('\u2013', '-').replace('\u2014', '-').replace('\u2018', "'").replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"').replace('\u2022', '*').replace('\u2026', '...')
    return clean

def generate_gemini_response(prompt):
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(prompt)
    return response.text

grouper = Agent(
    name="Student Grouper",
    role="Grouping students based on neurodevelopmental patterns",
    goal="Identify clusters of students with similar symptoms",
    backstory="An expert AI trained in clustering algorithms and pattern recognition, specialized in neurodevelopmental disorders."
)
iep_generator = Agent(
    name="IEP Generator",
    role="Generating personalized learning strategies",
    goal="Create tailored education plans with recommended strategies",
    backstory="An AI with expertise in educational psychology and personalized learning."
)
therapy_recommender = Agent(
    name="Therapy Recommender",
    role="Recommending therapy and specialists",
    goal="Suggest therapies and relevant specialists",
    backstory="An AI specialized in therapy recommendations and healthcare."
)
grouping_task = Task(
    description="Cluster students with similar symptoms based on their questionnaire responses",
    agent=grouper,
    expected_output="Student groups categorized by symptom patterns"
)
iep_task = Task(
    description="Generate personalized IEPs for each student group",
    agent=iep_generator,
    expected_output="Individualized education plans with AI-driven strategies"
)
therapy_task = Task(
    description="Suggest therapies and therapists based on the IEP",
    agent=therapy_recommender,
    expected_output="Recommended therapy types and therapists"
)
crew = Crew(
    agents=[grouper, iep_generator, therapy_recommender],
    tasks=[grouping_task, iep_task, therapy_task]
)

def run_neurotrack(data):
    data_dict = data.to_dict(orient='records')
    group_prompt = f"Group the following students into clusters based on their symptom patterns:\n{data_dict} , give your in short concise points and use good articulation"
    groups = generate_gemini_response(group_prompt)
    iep_prompt = f"Create personalized IEPs for the student groups:\n{groups} , give your in short concise points and use good articulation also include example IEPs in 2 bullet points"
    ieps = generate_gemini_response(iep_prompt)
    therapy_prompt = f"Suggest therapy types and therapists for the IEPs:\n{ieps} , give your in short concise points and use good articulation also include the type of therapy and the therapist"
    therapies = generate_gemini_response(therapy_prompt)
    return {"groups": groups, "ieps": ieps, "therapies": therapies}

st.set_page_config(page_title="Neurotrack AI", layout="wide")
st.title("Neurotrack AI Workflow")
st.markdown("This app groups student data, generates IEPs, and recommends therapies based on user inputs.")

if "patients" not in st.session_state:
    st.session_state.patients = generate_patient_responses(5)

st.subheader("Patient Questionnaire Responses")
df = pd.DataFrame(st.session_state.patients)
st.dataframe(df)

with st.expander("Edit Existing Patient Data"):
    patient_ids = [str(p["id"]) for p in st.session_state.patients]
    selected_id = st.selectbox("Select Patient ID to Edit", patient_ids)
    idx = next(i for i, p in enumerate(st.session_state.patients) if str(p["id"]) == selected_id)
    with st.form("edit_patient_form"):
        patient_id = st.number_input("Patient ID", min_value=1, value=st.session_state.patients[idx]["id"], step=1)
        social_interactions = st.slider("Social Interactions (1 = very easy, 4 = very difficult)", 1, 4, st.session_state.patients[idx]["social_interactions"])
        communication = st.slider("Communication (1 = very easy, 4 = very difficult)", 1, 4, st.session_state.patients[idx]["communication"])
        attention_focus = st.slider("Attention Focus (1 = very easy, 4 = very difficult)", 1, 4, st.session_state.patients[idx]["attention_focus"])
        hyperactivity_impulsivity = st.slider("Hyperactivity & Impulsivity (1 = very easy, 4 = very difficult)", 1, 4, st.session_state.patients[idx]["hyperactivity_impulsivity"])
        sensory_experiences = st.slider("Sensory Experiences (1 = very easy, 4 = very difficult)", 1, 4, st.session_state.patients[idx]["sensory_experiences"])
        learning_academics = st.slider("Learning Academics (1 = very easy, 4 = very difficult)", 1, 4, st.session_state.patients[idx]["learning_academics"])
        emotional_regulation = st.slider("Emotional Regulation (1 = very easy, 4 = very difficult)", 1, 4, st.session_state.patients[idx]["emotional_regulation"])
        if st.form_submit_button("Update Patient"):
            st.session_state.patients[idx] = {
                "id": patient_id,
                "social_interactions": social_interactions,
                "communication": communication,
                "attention_focus": attention_focus,
                "hyperactivity_impulsivity": hyperactivity_impulsivity,
                "sensory_experiences": sensory_experiences,
                "learning_academics": learning_academics,
                "emotional_regulation": emotional_regulation
            }
            st.success("Patient data updated")

with st.expander("Add New Patient Data"):
    with st.form("new_patient_form"):
        patient_id = st.number_input("Patient ID", min_value=1, value=len(st.session_state.patients) + 1, step=1)
        social_interactions = st.slider("Social Interactions (1 = very easy, 4 = very difficult)", 1, 4, 2)
        communication = st.slider("Communication (1 = very easy, 4 = very difficult)", 1, 4, 2)
        attention_focus = st.slider("Attention Focus (1 = very easy, 4 = very difficult)", 1, 4, 2)
        hyperactivity_impulsivity = st.slider("Hyperactivity & Impulsivity (1 = very easy, 4 = very difficult)", 1, 4, 2)
        sensory_experiences = st.slider("Sensory Experiences (1 = very easy, 4 = very difficult)", 1, 4, 2)
        learning_academics = st.slider("Learning Academics (1 = very easy, 4 = very difficult)", 1, 4, 2)
        emotional_regulation = st.slider("Emotional Regulation (1 = very easy, 4 = very difficult)", 1, 4, 2)
        if st.form_submit_button("Add Patient"):
            new_patient = {
                "id": patient_id,
                "social_interactions": social_interactions,
                "communication": communication,
                "attention_focus": attention_focus,
                "hyperactivity_impulsivity": hyperactivity_impulsivity,
                "sensory_experiences": sensory_experiences,
                "learning_academics": learning_academics,
                "emotional_regulation": emotional_regulation
            }
            st.session_state.patients.append(new_patient)
            st.success("New patient added")

if st.button("Run Neurotrack AI Workflow"):
    if st.session_state.patients:
        df = pd.DataFrame(st.session_state.patients)
        with st.spinner("Running Neurotrack AI..."):
            results = run_neurotrack(df)
            st.subheader("Final Results")
            st.markdown("**Student Groups:**")
            st.write(results["groups"])
            st.markdown("**IEPs:**")
            st.write(results["ieps"])
            st.markdown("**Therapies:**")
            st.write(results["therapies"])
            pdf = FPDF()
            pdf.add_page()
            try:
                pdf.add_font('DejaVu', '', '/usr/share/fonts/TTF/DejaVuSans.ttf', uni=True)
                pdf.set_font('DejaVu', '', 16)
            except:
                pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Neurotrack AI Results", ln=True, align="C")
            pdf.ln(10)
            try:
                pdf.set_font('DejaVu', '', 10)
            except:
                pdf.set_font("Arial", "", 10)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pdf.cell(0, 10, f"Generated on: {current_time}", ln=True)
            pdf.ln(5)
            clean_groups = clean_text_for_pdf(results["groups"])
            clean_ieps = clean_text_for_pdf(results["ieps"])
            clean_therapies = clean_text_for_pdf(results["therapies"])
            try:
                pdf.set_font('DejaVu', 'B', 14)
            except:
                pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Student Groups", ln=True)
            try:
                pdf.set_font('DejaVu', '', 10)
            except:
                pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 5, clean_groups)
            pdf.ln(10)
            try:
                pdf.set_font('DejaVu', 'B', 14)
            except:
                pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Individualized Education Plans", ln=True)
            try:
                pdf.set_font('DejaVu', '', 10)
            except:
                pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 5, clean_ieps)
            pdf.ln(10)
            try:
                pdf.set_font('DejaVu', 'B', 14)
            except:
                pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Recommended Therapies", ln=True)
            try:
                pdf.set_font('DejaVu', '', 10)
            except:
                pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 5, clean_therapies)
            pdf.ln(10)
            try:
                pdf.set_font('DejaVu', 'B', 14)
            except:
                pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Patient Assessment Matrix", ln=True)
            try:
                pdf.set_font('DejaVu', 'B', 9)
            except:
                pdf.set_font("Arial", "B", 9)
            categories = ["ID", "Social", "Comm", "Attn", "Hyper", "Sensory", "Learn", "Emotion"]
            col_width = 180 / len(categories)
            for category in categories:
                pdf.cell(col_width, 7, category, border=1, align='C')
            pdf.ln()
            try:
                pdf.set_font('DejaVu', '', 9)
            except:
                pdf.set_font("Arial", "", 9)
            for _, row in df.iterrows():
                pdf.cell(col_width, 7, str(int(row['id'])), border=1, align='C')
                pdf.cell(col_width, 7, str(int(row['social_interactions'])), border=1, align='C')
                pdf.cell(col_width, 7, str(int(row['communication'])), border=1, align='C')
                pdf.cell(col_width, 7, str(int(row['attention_focus'])), border=1, align='C')
                pdf.cell(col_width, 7, str(int(row['hyperactivity_impulsivity'])), border=1, align='C')
                pdf.cell(col_width, 7, str(int(row['sensory_experiences'])), border=1, align='C')
                pdf.cell(col_width, 7, str(int(row['learning_academics'])), border=1, align='C')
                pdf.cell(col_width, 7, str(int(row['emotional_regulation'])), border=1, align='C')
                pdf.ln()
            pdf.ln(5)
            pdf.cell(0, 5, "Rating Scale: 1 = minimal difficulty, 4 = significant difficulty", ln=True)
            pdf_file = pdf.output(dest='S').encode('latin1')
            st.download_button("Download PDF Report", data=pdf_file, file_name="neurotrack_report.pdf", mime="application/pdf")
    else:
        st.error("No patient data available.")

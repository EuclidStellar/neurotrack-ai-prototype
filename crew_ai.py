# ================================================
# Neurotrack AI Workflow for Student Grouping, IEP Generation, and Therapy Recommendations
# Using Gemini API
# ================================================

import google.generativeai as genai
from crewai import Crew, Agent, Task
import pandas as pd
import random
import os
from fpdf import FPDF
from datetime import datetime

# ========== CONFIGURATION ==========
# Replace with your Gemini API key
GEMINI_API_KEY = "your-api-key-here"
genai.configure(api_key=GEMINI_API_KEY)

# ========== SYMPTOM ASSESSMENT QUESTIONS ==========
# Storing questions in a dictionary format
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

# ========== SIMULATING PATIENT RESPONSES ==========
def generate_patient_responses(num_patients=5):
    """Generate random responses for simulated patient data."""
    responses = []
    
    for patient_id in range(1, num_patients + 1):
        patient_response = {
            "id": patient_id,
            "social_interactions": random.randint(1, 4),      # 1 = very easy, 4 = very difficult
            "communication": random.randint(1, 4),
            "attention_focus": random.randint(1, 4),
            "hyperactivity_impulsivity": random.randint(1, 4),
            "sensory_experiences": random.randint(1, 4),
            "learning_academics": random.randint(1, 4),
            "emotional_regulation": random.randint(1, 4)
        }
        responses.append(patient_response)
    
    return pd.DataFrame(responses)

# Generate simulated patient data
patient_data = generate_patient_responses(10)

# ========== GEMINI LLM HELPER FUNCTION ==========
def generate_gemini_response(prompt):
    """Generates response from Gemini API using the given prompt."""
    model = genai.GenerativeModel('gemini-1.5-pro')  # Updated model name
    response = model.generate_content(prompt)
    return response.text

# ========== IMPROVED CLEAN TEXT FUNCTION ==========
def clean_text_for_pdf(text):
    """Remove markdown formatting and unwanted symbols from text."""
    # Remove markdown formatting
    clean = text.replace("**", "")
    clean = clean.replace("*", "")
    clean = clean.replace("#", "")
    
    # Remove any lines starting with common thinking process indicators
    lines = clean.split('\n')
    filtered_lines = []
    for line in lines:
        if not any(line.strip().startswith(x) for x in ["Note:", "- Note:", "Based on", "Given the", 
                                                       "It's impossible", "Different methods", 
                                                       "Furthermore", "However", "Example IEP Components",
                                                       "I'll suggest"]):
            filtered_lines.append(line)
    
    clean = '\n'.join(filtered_lines)
    
    # Clean up extra whitespace
    while "\n\n\n" in clean:
        clean = clean.replace("\n\n\n", "\n\n")
    
    # Replace problematic Unicode characters with ASCII equivalents
    clean = clean.replace('\u2013', '-')  # Replace en dash with hyphen
    clean = clean.replace('\u2014', '-')  # Replace em dash with hyphen
    clean = clean.replace('\u2018', "'")  # Replace left single quote
    clean = clean.replace('\u2019', "'")  # Replace right single quote
    clean = clean.replace('\u201c', '"')  # Replace left double quote
    clean = clean.replace('\u201d', '"')  # Replace right double quote
    clean = clean.replace('\u2022', '*')  # Replace bullet point
    clean = clean.replace('\u2026', '...') # Replace ellipsis
    
    return clean

# ========== CREW AI AGENTS ==========
# Agent 1: Student Grouper
grouper = Agent(
    name="Student Grouper",
    role="Grouping students based on neurodevelopmental patterns",
    goal="Identify clusters of students with similar symptoms",
    backstory="An expert AI trained in clustering algorithms and pattern recognition , specialized in neurodevelopmental disorders."
    
)

# Agent 2: IEP Generator
iep_generator = Agent(
    name="IEP Generator",
    role="Generating personalized learning strategies",
    goal="Create tailored education plans with recommended strategies",
    backstory="An AI with expertise in educational psychology and personalized learning."
)

# Agent 3: Therapy Recommender
therapy_recommender = Agent(
    name="Therapy Recommender",
    role="Recommending therapy and specialists",
    goal="Suggest therapies and relevant specialists",
    backstory="An AI specialized in therapy recommendations and healthcare."
)

# ========== TASKS ==========
# Task 1: Grouping students
grouping_task = Task(
    description="Cluster students with similar symptoms based on their questionnaire responses",
    agent=grouper,
    expected_output="Student groups categorized by symptom patterns"
)

# Task 2: Generate IEPs
iep_task = Task(
    description="Generate personalized IEPs for each student group",
    agent=iep_generator,
    expected_output="Individualized education plans with AI-driven strategies"
)

# Task 3: Recommend therapies
therapy_task = Task(
    description="Suggest therapies and therapists based on the IEP",
    agent=therapy_recommender,
    expected_output="Recommended therapy types and therapists"
)

# ========== CREW WORKFLOW ==========
crew = Crew(
    agents=[grouper, iep_generator, therapy_recommender],
    tasks=[grouping_task, iep_task, therapy_task]
)

# ========== WORKFLOW EXECUTION ==========
def run_neurotrack(data):
    print("🚀 Running Neurotrack AI...\n")

    # Convert the DataFrame to dictionary format for compatibility
    data_dict = data.to_dict(orient='records')

    # Step 1: Grouping students
    print("📊 Grouping students...")
    group_prompt = f"Group the following students into clusters based on their symptom patterns:\n{data_dict} , give your in short concise points and use good articulation"
    groups = generate_gemini_response(group_prompt)

    # Step 2: Generating IEPs
    print("\n📚 Generating IEPs...")
    iep_prompt = f"Create personalized IEPs for the student groups:\n{groups} , give your in short concise points and use good articulation also include example IEPs in 2 bullet points"
    ieps = generate_gemini_response(iep_prompt)

    # Step 3: Recommending Therapies
    print("\n🩺 Recommending Therapies...")
    therapy_prompt = f"Suggest therapy types and therapists for the IEPs:\n{ieps} , give your in short concise points and use good articulation also include the type of therapy and the therapist"
    therapies = generate_gemini_response(therapy_prompt)

    print("\n✅ Neurotrack AI Completed!")

    # Return the results
    return {
        "groups": groups,
        "ieps": ieps,
        "therapies": therapies
    }


# ========== RUN THE WORKFLOW ==========
if __name__ == "__main__":
    print("📝 Patient Questionnaire Responses:")
    print(patient_data)

    # Run the workflow with patient responses
    results = run_neurotrack(patient_data)

    # Display the final results
    print("\n🔍 Final Results:")
    print("\nStudent Groups:\n", results["groups"])
    print("\nIEPs:\n", results["ieps"])
    print("\nTherapies:\n", results["therapies"])
    
    # Create PDF report
    print("\n📄 Generating PDF report...")
    
    # Initialize PDF with Unicode support
    pdf = FPDF()
    pdf.add_page()
    
    # Set font with Unicode support (using DejaVu if available)
    try:
        pdf.add_font('DejaVu', '', '/usr/share/fonts/TTF/DejaVuSans.ttf', uni=True)
        pdf.set_font('DejaVu', '', 16)
    except:
        # Fallback to standard font with careful text cleaning
        pdf.set_font("Arial", "B", 16)
    
    # Title
    pdf.cell(0, 10, "Neurotrack AI Results", ln=True, align="C")
    pdf.ln(10)
    
    # Date and time
    try:
        pdf.set_font('DejaVu', '', 10)
    except:
        pdf.set_font("Arial", "", 10)
        
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 10, f"Generated on: {current_time}", ln=True)
    pdf.ln(5)
    
    # Clean the results to remove unwanted text and problematic characters
    clean_groups = clean_text_for_pdf(results["groups"])
    clean_ieps = clean_text_for_pdf(results["ieps"])
    clean_therapies = clean_text_for_pdf(results["therapies"])
    
    # Student Groups section
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
    
    # IEPs section
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
    
    # Therapies section
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
    
    # Add patient data matrix
    try:
        pdf.set_font('DejaVu', 'B', 14)
    except:
        pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Patient Assessment Matrix", ln=True)
    
    # Table header
    try:
        pdf.set_font('DejaVu', 'B', 9)
    except:
        pdf.set_font("Arial", "B", 9)
    categories = ["ID", "Social", "Comm", "Attn", "Hyper", "Sensory", "Learn", "Emotion"]
    col_width = 180 / len(categories)
    for category in categories:
        pdf.cell(col_width, 7, category, border=1, align='C')
    pdf.ln()
    
    # Table data
    try:
        pdf.set_font('DejaVu', '', 9)
    except:
        pdf.set_font("Arial", "", 9)
    for _, row in patient_data.iterrows():
        pdf.cell(col_width, 7, str(int(row['id'])), border=1, align='C')
        pdf.cell(col_width, 7, str(int(row['social_interactions'])), border=1, align='C')
        pdf.cell(col_width, 7, str(int(row['communication'])), border=1, align='C')
        pdf.cell(col_width, 7, str(int(row['attention_focus'])), border=1, align='C')
        pdf.cell(col_width, 7, str(int(row['hyperactivity_impulsivity'])), border=1, align='C')
        pdf.cell(col_width, 7, str(int(row['sensory_experiences'])), border=1, align='C')
        pdf.cell(col_width, 7, str(int(row['learning_academics'])), border=1, align='C')
        pdf.cell(col_width, 7, str(int(row['emotional_regulation'])), border=1, align='C')
        pdf.ln()
    
    # Rating legend
    pdf.ln(5)
    pdf.cell(0, 5, "Rating Scale: 1 = minimal difficulty, 4 = significant difficulty", ln=True)
    
    
    # Change this path with your desired directory
    
    # Ensure the directory exists
    os.makedirs("/Users/euclidstellar/Downloads/proto/reports", exist_ok=True)
    
    # Save the PDF
    filename = f"/Users/euclidstellar/Downloads/proto/reports/neurotrack_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    try:
        pdf.output(filename)
        print(f"✅ PDF report saved as: {filename}")
    except UnicodeEncodeError:
        # If unicode error still occurs, try an even more aggressive cleaning approach
        print("⚠️ Unicode encoding issue detected. Attempting to fix...")
        
        # Reopen the PDF with stricter character filtering
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Neurotrack AI Results", ln=True, align="C")
        pdf.ln(10)
        
        # Save with ASCII-only content
        for result_text in [clean_groups, clean_ieps, clean_therapies]:
            # Ultra-strict ASCII-only filter
            ascii_text = ''.join(c if ord(c) < 128 else '-' for c in result_text)
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 5, ascii_text)
            pdf.ln(10)
            
        # Still include the matrix
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Patient Assessment Matrix", ln=True)
        
        # Recreate table with simple ASCII
        pdf.set_font("Arial", "B", 9)
        for category in categories:
            pdf.cell(col_width, 7, category, border=1, align='C')
        pdf.ln()
        
        pdf.set_font("Arial", "", 9)
        for _, row in patient_data.iterrows():
            for value in [row['id'], row['social_interactions'], row['communication'], 
                         row['attention_focus'], row['hyperactivity_impulsivity'],
                         row['sensory_experiences'], row['learning_academics'], 
                         row['emotional_regulation']]:
                pdf.cell(col_width, 7, str(int(value)), border=1, align='C')
            pdf.ln()
            
        # Save with a different name
        fallback_filename = f"/Users/euclidstellar/Downloads/proto/reports/neurotrack_report_ascii_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(fallback_filename)
        print(f"✅ PDF report (ASCII-only) saved as: {fallback_filename}")
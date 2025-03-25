# 🧠 Neurotrack AI
**Neurotrack AI** is an intelligent system that analyzes student neurological patterns to create personalized education plans and therapy recommendations.  
Utilizing advanced AI through the **Gemini Flash 1.5model** with **CrewAI AI agents**, it identifies clusters of students with similar neurodevelopmental characteristics, generates **Individualized Education Plans (IEPs)**, and suggests appropriate therapy interventions.

---

## 🚀 **Features**
- **Intelligent Student Grouping:** Analyzes questionnaire responses to identify patterns in neurodevelopmental traits  
- **IEP Generation:** Creates tailored education plans with personalized learning strategies  
- **Therapy Recommendations:** Suggests appropriate therapy types and specialists based on student needs  
- **PDF Report Generation:** Produces comprehensive reports with all findings and recommendations  
- **Data Visualization:** Includes assessment matrices to easily visualize student profiles  

---

## 📋 **Assessment Categories**
The system evaluates students across **seven key neurodevelopmental domains**:
- 👫 **Social interactions**
- 🗨️ **Communication**
- 🔍 **Attention focus**
- ⚡ **Hyperactivity/impulsivity**
- 🎧 **Sensory experiences**
- 📚 **Learning and academics**
- ❤️ **Emotional regulation**

---

## 🔧 **Technical Architecture**
**Neurotrack AI** leverages a **Crew AI** workflow with three specialized agents:
- 🧑‍🏫 **Student Grouper:** Identifies clusters of students with similar symptom patterns  
- 📚 **IEP Generator:** Creates personalized learning strategies for each group  
- 🩺 **Therapy Recommender:** Suggests appropriate therapies and specialists  

---

## 🚀 **Getting Started**

### ✅ **Prerequisites**
- Python 3.11+  
- Google Gemini API key  

### 🛠️ **Installation**
```bash
# Clone the repository
git clone https://github.com/euclidstellar/neurotrack-ai-prototype.git
cd neurotrack-ai-prototype

# Install required packages
pip install -r requirements.txt

# Add your Gemini API key
# Edit the GEMINI_API_KEY variable in crew_ai.py

# Configure PDF output location (optional)
# By default, PDFs are saved to "/Users/euclidstellar/Downloads/proto/reports/"
# Edit the path in the PDF generation section of crew_ai.py to change this location
```

### ▶️ **Usage**
```bash
# Run the Neurotrack AI system
python crew_ai.py
```
The script will:
- Generate simulated patient data (or you can use your own)  
- Group students based on symptom patterns  
- Create personalized IEPs for each group  
- Recommend appropriate therapies  
- Generate a **PDF report** with all findings  

---

## 📊 **Sample Output**
The system produces a comprehensive **PDF report** containing:
- ✅ Identified student groups with similar neurodevelopmental patterns  
- 📝 Personalized education plans for each group  
- 🩺 Recommended therapies and specialists  
- 📈 A **patient assessment matrix** for easy data visualization  

---
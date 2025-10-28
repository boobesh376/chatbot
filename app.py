import json
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import joblib 
import pandas as pd
import numpy as np
import os
import shap 
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv 

# Load environment variables from .env file immediately
load_dotenv()

# --- DATABASE AND APP SETUP ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'default_secret_key_if_not_set')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

# --- USER MODEL (For Login) ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# --- CONFIGURATION & MODEL LOADING ---
# 1. Gemini Configuration 
API_KEY = os.environ.get("GEMINI_API_KEY") 

try:
    if not API_KEY or API_KEY == "AIzaSy...your_actual_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY not found or is placeholder in environment variables.")
    genai.configure(api_key=API_KEY) 
    gemini_model = genai.GenerativeModel("gemini-2.5-flash") 
    print("✅ Gemini Model configured successfully.")
except Exception as e:
    print(f"❌ GEMINI CONFIG ERROR: Check your key and connection. Error: {e}")
    
# 2. ML Model Configuration 
try:
    ADE_MODEL = joblib.load('ade_classifier_logreg.pkl')
    EFFECTIVENESS_MODEL = joblib.load('effectiveness_regressor_rf.pkl')
    SHAP_BACKGROUND_DATA = joblib.load('shap_background_data.pkl')
    
    # Initialize SHAP Explainer
    ADE_EXPLAINER = shap.Explainer(ADE_MODEL.predict_proba, SHAP_BACKGROUND_DATA)
    
    print("✅ All ML Models and SHAP Explainer loaded successfully.")
except FileNotFoundError:
    print("❌ FATAL ERROR: Required ML files not found. Run model_trainer.py first.")
    ADE_MODEL = None
    EFFECTIVENESS_MODEL = None
    ADE_EXPLAINER = None

# --- CONSTANTS (Must match model_trainer.py) ---
MODEL_FEATURES = [
    'Age', 'TotalMedications', 'AlphaDiversity', 'MicrobialMetabolite_A_Level',
    'FB_Ratio', 'RiskDrugCount', 'AntiFibroticUse'
]
ADE_RISK_DRUGS = ["Pantoprazole", "Omeprazole", "Esomeprazole", "Azithromycin", "Ciprofloxacin", "Prednisone", "Warfarin", "Clopidogrel", "Aspirin"]
ANTI_FIBROTIC_DRUGS = ["Pirfenidone", "Nintedanib"]


# --- SIMPLIFIED FEATURE EXTRACTION FUNCTION (LLM output is CSV-like text) ---
def extract_features_from_text(user_input: str) -> dict or None:
    """Uses Gemini to extract structured patient data into a simple CSV-like format."""
    
    # The prompt now asks for a predictable, simple text output, NOT JSON.
    extraction_prompt = f"""
    **TASK:** Analyze the following patient data text. Extract the required features and output ONLY a single line of pipe-separated values (|). If a medication is mentioned, list ALL mentioned drugs under the MedicationList_Text. Use 0 or 0.0 for missing values.
    
    **OUTPUT FORMAT (Example):** <Age>|<TotalMeds>|<AlphaDiversity>|<MetaboliteA>|<BacteroidesAbundance>|<FirmicutesAbundance>|<MedicationList_Text>
    
    **INPUT TEXT:** "{user_input}"
    
    **INSTRUCTIONS:** Output ONLY the single line of pipe-separated data. Do not include headers, explanations, or any other text.
    """
    
    try:
        response = genai.GenerativeModel("gemini-2.5-flash").generate_content(extraction_prompt)
        raw_output = response.text.strip().replace('*', '').replace('`', '').replace('|', '|').strip()
        
        # --- ROBUST PARSING FIX ---
        parts = raw_output.split('|')
        
        if len(parts) < 7:
            app.logger.error(f"LLM Parsing Error: Expected 7 parts, got {len(parts)}. Raw: {raw_output}")
            return None # Safely exit if parsing fails
            
        # Try converting the parts to their expected types
        extracted_data = {}
        try:
            extracted_data['Age'] = int(float(parts[0].strip()))
            extracted_data['TotalMedications'] = int(float(parts[1].strip()))
            extracted_data['AlphaDiversity'] = float(parts[2].strip())
            extracted_data['MicrobialMetabolite_A_Level'] = float(parts[3].strip())
            extracted_data['Bacteroides_Abundance'] = float(parts[4].strip())
            extracted_data['Firmicutes_Abundance'] = float(parts[5].strip())
            extracted_data['MedicationList_Text'] = parts[6].strip()
        except ValueError as ve:
            app.logger.error(f"Value Conversion Error: {ve}. Check the format of LLM output values.")
            return None
        
        return extracted_data
        
    except Exception as e:
        app.logger.error(f"Extraction API Call Error: {e}")
        return None

# --- FEATURE ENGINEERING FUNCTION (Updated to match new parsing logic) ---
def engineer_features(patient_data: dict) -> pd.DataFrame:
    """Processes extracted data into the features required by the ML model."""
    
    age = patient_data.get('Age', 65)
    total_meds = patient_data.get('TotalMedications', 4)
    alpha_diversity = patient_data.get('AlphaDiversity', 4.0)
    metabolite_a = patient_data.get('MicrobialMetabolite_A_Level', 10.0)
    bacteroides = patient_data.get('Bacteroides_Abundance', 0.25)
    firmicutes = patient_data.get('Firmicutes_Abundance', 0.50)
    medication_list_text = patient_data.get("MedicationList_Text", "")
    
    # 1. Feature Calculation: FB_Ratio 
    fb_ratio = firmicutes / (bacteroides if bacteroides > 0 else 0.001)

    # 2. Feature Parsing: RiskDrugCount and AntiFibroticUse (using simple text search)
    risk_drug_count = sum(1 for drug in ADE_RISK_DRUGS if drug.lower() in medication_list_text.lower())
    anti_fibrotic_use = int(any(drug.lower() in medication_list_text.lower() for drug in ANTI_FIBROTIC_DRUGS))
    
    # 3. Create the final DataFrame
    data_row = {
        'Age': age, 
        'TotalMedications': total_meds, 
        'AlphaDiversity': alpha_diversity, 
        'MicrobialMetabolite_A_Level': metabolite_a,
        'FB_Ratio': fb_ratio,
        'RiskDrugCount': risk_drug_count,
        'AntiFibroticUse': anti_fibrotic_use
    }
    
    return pd.DataFrame([data_row])[MODEL_FEATURES], medication_list_text

# --- XAI AND RECOMMENDATION LOGIC ---
def get_xai_and_recommendation(features_df: pd.DataFrame, ade_prob: float, eff_score: float, med_list_text: str):
    """Generates SHAP explanation and synthesizes recommendations."""
    
    shap_values_raw = ADE_EXPLAINER(features_df.values)[0]
    
    shap_contributions = pd.DataFrame({
        'Feature': features_df.columns,
        'Contribution': shap_values_raw.values[0][1]
    })

    shap_contributions['Abs'] = shap_contributions['Contribution'].abs()
    top_contributors = shap_contributions.sort_values(by='Abs', ascending=False).head(5)

    recommendations = []
    
    fb_ratio = features_df['FB_Ratio'].iloc[0]
    alpha_div = features_df['AlphaDiversity'].iloc[0]
    risk_drug_count = features_df['RiskDrugCount'].iloc[0]
    
    # R1: Critical Risk
    if ade_prob > 0.65:
        recommendations.append("🚨 **CRITICAL RISK:** High probability of Adverse Drug Event detected.")
        top_positive_contributor = top_contributors[top_contributors['Contribution'] > 0.15]['Feature'].iloc[0] if not top_contributors[top_contributors['Contribution'] > 0.15].empty else 'Multiple factors'
        recommendations.append(f"Primary risk driver: **{top_positive_contributor}**. Immediate clinical review of dosage/substitution is advised, especially for known DDIs.")

    # R2: Microbiome/Effectiveness Intervention
    if eff_score < 3.5 or alpha_div < 3.5:
        recommendations.append(f"📉 **Efficacy Concern:** Effectiveness score is {eff_score:.1f}. Low Alpha Diversity ({alpha_div:.2f}) suggests potential reduced drug efficacy.")
        recommendations.append("ACTION: Recommend microbiome modulation (e.g., high-fiber diet, specific probiotics) to improve gut health and potentially enhance drug metabolism.")

    # R3: Polypharmacy Review
    if risk_drug_count >= 2 and eff_score < 4.0:
        recommendations.append(f"⚠️ **Polypharmacy Check:** Patient is on {risk_drug_count} high-risk drugs. If appropriate, review necessity of drugs like PPIs (Omeprazole, Pantoprazole) to minimize systemic load and preserve microbiome integrity.")

    if not recommendations:
        recommendations.append("✅ **Optimal Profile:** Continue current regimen with routine monitoring, focusing on maintaining Alpha Diversity.")
        
    return recommendations, top_contributors.to_dict('records')


# --- LOGIN/REGISTRATION ROUTES (Authentication Layer) ---
@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['logged_in'] = True
            session['user_id'] = user.id
            session['user_email'] = user.email
            flash('Login successful!', 'success')
            return redirect(url_for('chat_page'))
        else:
            flash('Login failed. Check your email and password.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email address already registered.', 'danger')
            return redirect(url_for('register'))
            
        new_user = User(email=email)
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('user_id', None)
    session.pop('user_email', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/chat_page')
def chat_page():
    if not session.get('logged_in'):
        flash('Please log in to access the chatbot.', 'warning')
        return redirect(url_for('login'))
        
    return render_template('index.html')

@app.route('/settings')
def settings_page():
    if not session.get('logged_in'):
        flash('Please log in to access settings.', 'warning')
        return redirect(url_for('login'))
        
    return render_template('settings.html')

@app.route('/clear-history', methods=['POST'])
def clear_chat_history():
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    
    flash('Chat history cleared successfully (Browser history needs manual refresh).', 'success')
    return redirect(url_for('settings_page'))

# --- CONSOLIDATED CHAT/ANALYSIS ENDPOINT ---
@app.route('/chat', methods=['POST'])
def handle_chat_query():
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized. Please log in."}), 401

    raw_data = request.get_json()
    user_input = raw_data.get('query', '').strip()
    chat_tone = raw_data.get('chat_tone', 'clinical')
    
    if ADE_MODEL is None or EFFECTIVENESS_MODEL is None:
        return jsonify({"error": "ML Models are not loaded. Run model_trainer.py first."}), 503
    if not user_input:
        return jsonify({"response": "Please enter a query."}), 400
        
    
    # 1. ATTEMPT STRUCTURED ANALYSIS (XAI MODE)
    patient_data = extract_features_from_text(user_input)

    # Check if a meaningful amount of data was extracted (Age > 40, at least 2 meds)
    if patient_data and patient_data.get('Age', 0) > 40 and patient_data.get('TotalMedications', 0) >= 2:
        
        try:
            # 1.1 PHASE 1: Feature Engineering and ML Prediction
            features_df_raw, med_list_text = engineer_features(patient_data)
            
            # Predict ADE Risk Probability (Class 1)
            ade_prob = ADE_MODEL.predict_proba(features_df_raw)[0][1]
            # Predict Effectiveness Score (Regression)
            eff_score_raw = EFFECTIVENESS_MODEL.predict(features_df_raw)[0]
            eff_score = np.clip(round(eff_score_raw, 1), 1.0, 5.0) # Clip score between 1 and 5
            
            ade_percentage = f"{ade_prob * 100:.2f}%"

            # 1.2 PHASE 2: XAI and Recommendation Synthesis
            recommendations, top_contributors = get_xai_and_recommendation(
                features_df_raw, ade_prob, eff_score, med_list_text
            )
            
            # 1.3 PHASE 3: Dynamic Prompting for Explainable AI (XAI)
            
            recommendation_text = "\n* ".join(recommendations)
            contributor_text = "\n".join([f"* **{c['Feature']}**: {c['Contribution']:.4f}" for c in top_contributors])
            
            prompt = f"""
            **ROLE:** You are an Explainable AI (XAI) Assistant for the FibroGuide system. You MUST provide a concise, professional, and personalized report (max 250 words) to an IPF specialist doctor.
            
            **PATIENT PROFILE SUMMARY:** Age {patient_data.get('Age')} years, Total Meds {patient_data.get('TotalMedications')}.
            
            **MODEL PREDICTIONS:**
            - Adverse Drug Event (ADE) Risk Probability: {ade_percentage}
            - Predicted Drug Effectiveness Score (1.0-5.0): {eff_score:.1f}
            
            **MODEL RATIONALE (SHAP/ML):**
            The top 5 factors influencing the ADE prediction were:
            {contributor_text}
            
            **INSTRUCTIONS for Response Structure (Strictly adhere to this format):**
            
            1.  **Recommendation Summary:** Start with this exact heading. State the predicted ADE Risk and Effectiveness Score clearly.
            2.  **Rationale:** Based on the Model Rationale above, describe the primary patient-specific risks (e.g., "High Risk Drug Count" or "Low Alpha Diversity") that drove the predictions.
            3.  **Actionable Recommendations:** Provide a list of actions derived from the synthesized recommendations:
                {recommendation_text}
            4.  **Tone:** Ensure the output uses only professional, clinical language. Do not add introductory or concluding conversational text.
            """
            
            response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
            response_text = response.text
            
            # --- FINAL CLEANUP ---
            response_text = response_text.replace('**', '') 
            response_text = response_text.replace("Recommendation Summary:", "<span class='analysis-header'>Recommendation Summary:</span>")

            final_output = "🔬 <span class='report-header-text'>FibroGuide XAI Analysis Report</span>\n\n" + response_text
            
            return jsonify({"success": True, "response": final_output})

        except Exception as e:
            # THIS IS NOW A FALLBACK ONLY. If this fails, the app still falls back to general chat.
            app.logger.error(f"ML/Prompting Error: {e}")
            pass

    # 2. FALLBACK TO GENERAL CONVERSATIONAL QUERY (CHATBOT MODE)
    
    if chat_tone == 'supportive':
        system_prompt = "You are a warm, supportive, and friendly medical assistant specializing in Idiopathic Pulmonary Fibrosis (IPF). Answer the user's question in an encouraging and helpful tone."
    elif chat_tone == 'detailed':
        system_prompt = "You are a highly detailed and educational medical assistant specializing in Idiopathic Pulmonary Fibrosis (IPF). Answer the user's question with comprehensive, factual information, citing key terms."
    else: # Default is 'clinical'
        system_prompt = "You are a concise, factual, and professional clinical decision support assistant specializing in Idiopathic Pulmonary Fibrosis (IPF). Answer the user's question with high precision."

    try:
        response = genai.GenerativeModel("gemini-2.5-flash").generate_content([system_prompt, user_input])
        response_text = response.text.replace('**', '') 
        return jsonify({"success": True, "response": response_text})
    except Exception as e:
        app.logger.error(f"Gemini Chat Error: {e}")
        return jsonify({"error": "LLM service unavailable or invalid API key."}), 500


if __name__ == '__main__':
    with app.app_context():
        # Initialize the database and create tables
        db.create_all()
        
    app.run(debug=True, host='0.0.0.0')

import json
import google.generativeai as genai
from google.cloud import vision 
from google.genai import types 
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import joblib 
import pandas as pd
import numpy as np
import os
import shap 
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv 
import base64 

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
gemini_model = None # Initialize globally

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


# ----------------------------------------------------------------------
## 🛠️ NEW: CLOUD VISION OCR FUNCTION
# ----------------------------------------------------------------------

def get_text_from_image_vision(image_bytes: bytes) -> str:
    """Uses Google Cloud Vision API to perform robust OCR on image bytes."""
    try:
        # NOTE: This client relies on gcloud auth application-default login
        vision_client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_bytes)
        
        # Use DOCUMENT_TEXT_DETECTION for dense, multi-line text (like prescriptions)
        response = vision_client.document_text_detection(image=image)
        
        if response.full_text_annotation:
            return response.full_text_annotation.text
        else:
            return ""
            
    except Exception as e:
        app.logger.error(f"Cloud Vision API OCR Error: {e}")
        # Return a flag string indicating failure
        return f"OCR_ERROR: Failed to connect or authenticate Vision API. Check gcloud login and API status. Error: {e}" 


# ----------------------------------------------------------------------
## 🔬 STRUCTURED OUTPUT SCHEMA
# ----------------------------------------------------------------------

# Define the exact JSON schema the model MUST return.
EXTRACTION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        'MedicationList_Text': types.Schema(type=types.Type.STRING, description="Comma-separated list of ONLY drug names extracted from text or image. Default to '' if none found."),
        'Age': types.Schema(type=types.Type.INTEGER, description="Patient's age. Default: 65."),
        'TotalMedications': types.Schema(type=types.Type.INTEGER, description="Total number of unique medications. Default: 4."),
        'AlphaDiversity': types.Schema(type=types.Type.NUMBER, description="Alpha Diversity level. Default: 4.0."),
        'MicrobialMetabolite_A_Level': types.Schema(type=types.Type.NUMBER, description="Microbial Metabolite A level. Default: 10.0."),
        'Bacteroides_Abundance': types.Schema(type=types.Type.NUMBER, description="Bacteroides Abundance. Default: 0.25."),
        'Firmicutes_Abundance': types.Schema(type=types.Type.NUMBER, description="Firmicutes Abundance. Default: 0.50.")
    },
    required=['Age', 'TotalMedications', 'AlphaDiversity', 'MicrobialMetabolite_A_Level', 
              'Bacteroides_Abundance', 'Firmicutes_Abundance', 'MedicationList_Text']
)


# ----------------------------------------------------------------------
## 🔬 FEATURE EXTRACTION FUNCTION (TEXT-ONLY)
# ----------------------------------------------------------------------
def run_gemini_extraction(combined_text: str) -> dict or None:
    """Uses Gemini (text-only mode) to extract structured data from a combined text string."""
    global gemini_model
    
    if not gemini_model:
        app.logger.error("Gemini model not initialized.")
        return None

    system_instruction = (
        "You are a strict, professional data extraction specialist. Your task is to extract ALL required data points "
        "from the provided patient text. For any parameter not explicitly found, use the default value. "
        "Your ONLY output MUST be a valid JSON object matching the provided schema, with no additional text or markdown."
    )
    
    try:
        response = gemini_model.generate_content(
            contents=[combined_text], # ONLY text content now
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EXTRACTION_SCHEMA,
                system_instruction=system_instruction
            )
        )
        
        raw_json_output = response.text.strip().replace('```json', '').replace('```', '').strip()
        extracted_data = json.loads(raw_json_output)
        
        app.logger.info(f"LLM Structured JSON Output: {extracted_data}")
        
        # Validation: Check if the extracted data is just the default profile.
        is_meaningful_data = not (
            extracted_data.get('Age') == 65 and 
            extracted_data.get('TotalMedications') == 4 and 
            extracted_data.get('AlphaDiversity') == 4.0 and
            extracted_data.get('Bacteroides_Abundance') == 0.25 and 
            extracted_data.get('MedicationList_Text') == "" 
        )

        if not is_meaningful_data:
            app.logger.warning("Extraction returned only default values. Assuming no meaningful patient data found.")
            return None
            
        return extracted_data
        
    except Exception as e:
        app.logger.error(f"Structured Extraction API Call/Parsing Error: {e}")
        return None

# --- FEATURE ENGINEERING FUNCTION (NO CHANGE) ---
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

# --- XAI AND RECOMMENDATION LOGIC (NO CHANGE) ---
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


# --- LOGIN/REGISTRATION ROUTES (NO CHANGE) ---
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

# ----------------------------------------------------------------------
## 🎯 CRITICAL FIXES FOR IMAGE/CHAT LOGIC (HYBRID IMPLEMENTATION)
# ----------------------------------------------------------------------
@app.route('/chat', methods=['POST'])
def handle_chat_query():
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized. Please log in."}), 401

    raw_data = request.get_json()
    user_input = raw_data.get('query', '').strip()
    image_data_base64 = raw_data.get('image_data', None)
    chat_tone = raw_data.get('chat_tone', 'clinical')
    
    if ADE_MODEL is None or EFFECTIVENESS_MODEL is None:
        return jsonify({"error": "ML Models are not loaded. Run model_trainer.py first."}), 503
    
    if not user_input and not image_data_base64: 
        return jsonify({"response": "Please enter a query or upload an image."}), 400
        
    app.logger.info(f"Image Data Received: {bool(image_data_base64)}")
    
    # --- Data Containers ---
    extracted_ocr_text = ""
    image_bytes = None
    
    if image_data_base64:
        try:
            # 1. ROBUST BASE64 DECODING (TO PREVENT 500 CRASH)
            base64_decoded_string = image_data_base64.replace("-", "+").replace("_", "/")
            missing_padding = len(base64_decoded_string) % 4
            if missing_padding:
                base64_decoded_string += "=" * (4 - missing_padding)
            image_bytes = base64.b64decode(base64_decoded_string)

            # 2. HYBRID OCR: Get text from the image using Cloud Vision API
            # Note: This is the external call that requires gcloud auth application-default login
            extracted_ocr_text = get_text_from_image_vision(image_bytes)
            app.logger.info(f"OCR Extracted Text: {extracted_ocr_text}")
            
            if extracted_ocr_text.startswith("OCR_ERROR"):
                # Critical failure in Vision API, must stop and use the error message
                image_data_base64 = None
                
        except Exception as e:
            app.logger.error(f"FATAL IMAGE PROCESSING CRASH: {e}")
            image_data_base64 = None
    
    # 3. COMBINE ALL TEXT INPUTS for Gemini Extraction
    combined_text_for_gemini = user_input
    if extracted_ocr_text and not extracted_ocr_text.startswith("OCR_ERROR"):
        # Prepend the extracted text to the user's prompt (user_input contains the XAI instruction or the general query)
        combined_text_for_gemini = f"--- OCR RESULTS START ---\n{extracted_ocr_text}\n--- OCR RESULTS END ---\n{user_input}"
    
    # 1. ATTEMPT STRUCTURED ANALYSIS (XAI MODE)
    # Trigger XAI if an image was attempted AND we successfully got OCR text OR the user sent the extraction instruction.
    is_xai_mode_intended = (image_data_base64 and extracted_ocr_text) or ("**INSTRUCTION FOR XAI**" in user_input)
    
    is_meaningful_xai_data = None
    patient_data = None
    
    if is_xai_mode_intended:
        
        # --- Execute Gemini Extraction for features ---
        patient_data = run_gemini_extraction(combined_text_for_gemini)
        is_meaningful_xai_data = patient_data is not None
        
        if is_meaningful_xai_data:
            
            try:
                # 1.1 PHASE 1: Feature Engineering and ML Prediction
                features_df_raw, med_list_text = engineer_features(patient_data)
                
                # Predict ADE Risk Probability (Class 1)
                ade_prob = ADE_MODEL.predict_proba(features_df_raw)[0][1]
                # Predict Effectiveness Score (Regression)
                eff_score_raw = EFFECTIVENESS_MODEL.predict(features_df_raw)[0]
                eff_score = np.clip(round(eff_score_raw, 1), 1.0, 5.0) 
                
                ade_percentage = f"{ade_prob * 100:.2f}%"

                # 1.2 PHASE 2 & 3: XAI Report Generation
                recommendations, top_contributors = get_xai_and_recommendation(
                    features_df_raw, ade_prob, eff_score, med_list_text
                )
                
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
                
                1.  **Recommendation Summary:** Start with this exact heading. State the predicted ADE Risk and Effectiveness Score clearly.
                2.  **Rationale:** Based on the Model Rationale above, describe the primary patient-specific risks (e.g., "High Risk Drug Count" or "Low Alpha Diversity") that drove the predictions.
                3.  **Actionable Recommendations:** Provide a list of actions derived from the synthesized recommendations:
                    {recommendation_text}
                4.  **Tone:** Ensure the output uses only professional, clinical language. Do not add introductory or concluding conversational text.
                """
                
                response = gemini_model.generate_content([prompt])
                response_text = response.text
                
                response_text = response_text.replace('**', '') 
                response_text = response_text.replace("Recommendation Summary:", "<span class='analysis-header'>Recommendation Summary:</span>")

                final_output = "🔬 <span class='report-header-text'>FibroGuide XAI Analysis Report</span>\n\n" + response_text
                
                return jsonify({"success": True, "response": final_output})

            except Exception as e:
                app.logger.error(f"ML/Prompting Error (Attempted XAI): {e}")
                pass
    
    # 2. FALLBACK TO GENERAL CONVERSATIONAL QUERY (CHATBOT MODE)
    
    # 2.1 Specific Extraction Failure Message:
    # This runs if the user intended XAI (image or instruction) but we failed to get structured data.
    if is_xai_mode_intended and not is_meaningful_xai_data:
        required_data = "Age, Total Meds, or Microbiome data"
        if "**INSTRUCTION FOR XAI**" in user_input: 
            required_data = "Medication list and/or " + required_data
            
        fallback_message = f"**EXTRACTION FAILED:** An image was provided, but structured data for the XAI analysis could not be extracted (Missing: {required_data}). Please ensure the image is clear or provide the data in the text box."
        
        # If the failure was due to the OCR service itself, report that specific error
        if extracted_ocr_text.startswith("OCR_ERROR"):
             fallback_message = f"**CRITICAL SERVICE ERROR:** The Google Cloud Vision API failed to extract text. Please ensure the **Cloud Vision API is ENABLED** in your project and you have run **gcloud auth application-default login**."
             return jsonify({"success": True, "response": fallback_message})
        
        # If the user's intent was extraction but it failed, return the failure message
        return jsonify({"success": True, "response": fallback_message})
    
    # 2.2 Standard General Chat:
    
    if chat_tone == 'supportive':
        system_prompt = "You are a warm, supportive, and friendly medical assistant specializing in Idiopathic Pulmonary Fibrosis (IPF). Answer the user's question in an encouraging and helpful tone."
    elif chat_tone == 'detailed':
        system_prompt = "You are a highly detailed and educational medical assistant specializing in Idiopathic Pulmonary Fibrosis (IPF). Answer the user's question with comprehensive, factual information, citing key terms."
    else: # Default is 'clinical'
        system_prompt = "You are a concise, factual, and professional clinical decision support assistant specializing in Idiopathic Pulmonary Fibrosis (IPF). Answer the user's question with high precision."

    # For general chat, only send the original user text, stripped of any hidden instructions
    final_chat_query = raw_data.get('query', '').strip()
    
    # Prepare the content list for the final chat model
    chat_content_list = [system_prompt]
    
    # If we have OCR text (and it didn't crash), include it as context for the chat model
    if extracted_ocr_text and not extracted_ocr_text.startswith("OCR_ERROR"):
        chat_content_list.append(f"Context from attached document: {extracted_ocr_text}")
        
    # Append the user's actual question
    chat_content_list.append(final_chat_query)


    try:
        response = gemini_model.generate_content(chat_content_list)
        response_text = response.text.replace('**', '') 
        return jsonify({"success": True, "response": response_text})
    except Exception as e:
        app.logger.error(f"Gemini Chat Error: {e}")
        return jsonify({"error": "LLM service unavailable or invalid API key."}), 500


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
    app.run(debug=True, host='0.0.0.0')
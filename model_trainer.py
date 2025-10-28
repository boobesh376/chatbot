import pandas as pd
import numpy as np
import json
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error
import shap

# --- CONFIGURATION & CONSTANTS ---
# NOTE: The data file name must match the output from data_augmenter.py
data_file = r'C:\Users\janan\Downloads\FibroGuide-Flask-Chatbot\augmented_ipf_patient_data_1000.csv' 
ADE_MODEL_PATH = 'ade_classifier_logreg.pkl'
EFFECTIVENESS_MODEL_PATH = 'effectiveness_regressor_rf.pkl'
FEATURE_NAMES_PATH = 'feature_names.pkl'

# FEATURES: Define the EXACT list of engineered features for the models
MODEL_FEATURES = [
    'Age', 
    'TotalMedications', 
    'AlphaDiversity', 
    'MicrobialMetabolite_A_Level',
    'FB_Ratio',
    'RiskDrugCount',
    'AntiFibroticUse'
]

# High-risk drug list: Must be consistent across all files
ADE_RISK_DRUGS = ["Pantoprazole", "Omeprazole", "Esomeprazole", "Azithromycin", "Ciprofloxacin", "Prednisone", "Warfarin", "Clopidogrel", "Aspirin"]
ANTI_FIBROTICS = ["Pirfenidone", "Nintedanib"]


# --- FEATURE ENGINEERING FUNCTIONS ---

def count_risk_drugs(med_json_str):
    """Counts high-risk drugs."""
    try:
        med_list = json.loads(str(med_json_str)) 
        return sum(1 for drug in med_list if drug.get('drugName') in ADE_RISK_DRUGS)
    except:
        return 0

def has_antifibrotic(med_json_str):
    """Checks for core anti-fibrotic drugs."""
    try:
        med_list = json.loads(str(med_json_str))
        return any(drug.get('drugName') in ANTI_FIBROTICS for drug in med_list)
    except:
        return 0

# --- MAIN TRAINING FUNCTION ---
def run_training():
    
    try:
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} augmented records for training from '{data_file}'.")
    except FileNotFoundError:
        print(f"❌ ERROR: Training data file '{data_file}' not found. Did you run data_augmenter.py?")
        return

    # 1. FEATURE ENGINEERING 
    df['RiskDrugCount'] = df['MedicationList_JSON'].apply(count_risk_drugs)
    df['AntiFibroticUse'] = df['MedicationList_JSON'].apply(has_antifibrotic).astype(int)
    
    # Calculate the Firmicutes/Bacteroides ratio (FB_Ratio)
    df['FB_Ratio'] = df['Firmicutes_Abundance'] / df['Bacteroides_Abundance'].replace(0, 0.001)
    
    # Save the feature names list (Crucial for the Flask App to ensure column order)
    joblib.dump(MODEL_FEATURES, FEATURE_NAMES_PATH)
    
    # 2. TRAIN ADE CLASSIFIER (Logistic Regression)
    X = df[MODEL_FEATURES]
    Y_ade = df['AdverseDrugEvent_Occurred']
    
    X_train_ade, X_test_ade, Y_train_ade, Y_test_ade = train_test_split(X, Y_ade, test_size=0.2, random_state=42, stratify=Y_ade)

    ade_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42, solver='liblinear'))
    ])
    
    ade_pipeline.fit(X_train_ade, Y_train_ade)
    
    # 3. TRAIN EFFECTIVENESS REGRESSOR (Random Forest Regressor)
    Y_eff = df['DrugEffectiveness_Score']
    X_train_eff, X_test_eff, Y_train_eff, Y_test_eff = train_test_split(X, Y_eff, test_size=0.2, random_state=42)

    eff_pipeline = Pipeline([
        ('scaler', StandardScaler()), # Standardize the features
        ('model', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8))
    ])
    
    eff_pipeline.fit(X_train_eff, Y_train_eff)
    
    # 4. REPORT AND SAVE MODELS
    print(f"\n--- Model Training Summary ---")
    print(f"ADE Classifier Accuracy (Test): {ade_pipeline.score(X_test_ade, Y_test_ade):.4f}")
    eff_preds = eff_pipeline.predict(X_test_eff)
    print(f"Effectiveness Regressor RMSE (Test): {np.sqrt(mean_squared_error(Y_test_eff, eff_preds)):.4f}")

    joblib.dump(ade_pipeline, ADE_MODEL_PATH)
    joblib.dump(eff_pipeline, EFFECTIVENESS_MODEL_PATH)
    print(f"✅ Both trained models saved successfully.")
    
    # 5. SHAP INTEGRATION (Crucial for XAI)
    # We save a sample of the training data features (X_train) for the SHAP Explainer background.
    X_train_sample = X_train_ade.sample(50, random_state=42).reset_index(drop=True)
    joblib.dump(X_train_sample, 'shap_background_data.pkl')
    print("✅ SHAP background data saved for XAI integration.")

if __name__ == '__main__':
    run_training()

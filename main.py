import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)

# 1. Performance Evaluation Logic (100 Iterations)
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Load Data
df_eval = pd.read_excel('/content/alzheimers_disease_data.xlsx')

# Keep a full copy for the UI search before dropping ID columns for training
df_ui_library = df_eval.copy()

# Prepare training data
df_eval_clean = df_eval.drop(columns=["PatientID", "DoctorInCharge"])
X_eval = df_eval_clean.drop(columns=["Diagnosis"])
y_eval = df_eval_clean["Diagnosis"]

for i in range(100):
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_eval, y_eval, test_size=0.2, random_state=i)
    scaler_f = StandardScaler()
    X_train_f_sc = scaler_f.fit_transform(X_train_f)
    X_test_f_sc = scaler_f.transform(X_test_f)

    # Model Training with entropy and tuned hyperparameters
    temp_model = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=26, random_state=42)
    temp_model.fit(X_train_f_sc, y_train_f)

    y_pred_f = temp_model.predict(X_test_f_sc)
    accuracies.append(accuracy_score(y_test_f, y_pred_f))
    precisions.append(precision_score(y_test_f, y_pred_f))
    recalls.append(recall_score(y_test_f, y_pred_f))
    f1_scores.append(f1_score(y_test_f, y_pred_f))

print("===== CDSS PERFORMANCE (DT Diagnostic Engine - 100 Iterations) ====")
print(f"Average Accuracy : {np.mean(accuracies):.4f}")
print(f"Average Precision: {np.mean(precisions):.4f}")
print(f"Average Recall   : {np.mean(recalls):.4f}")
print(f"Average F1 Score : {np.mean(f1_scores):.4f}")

# 2. Clinical Guardrails Engine
def get_clinical_recommendations(patient_row):
    recommendations = []
    if patient_row['BMI'] > 25:
        recommendations.append(('BMI', f"{patient_row['BMI']:.2f}", "Contributes to Risk (Target < 25)"))
    elif patient_row['BMI'] < 18.5:
        recommendations.append(('BMI', f"{patient_row['BMI']:.2f}", "Contributes to Risk (Underweight < 18.5)"))
    else:
        recommendations.append(('BMI', f"{patient_row['BMI']:.2f}", "Protective Factor (Healthy Range)"))

    if patient_row['Smoking'] == 1: recommendations.append(('Smoking', "Yes", "Contributes to Risk (Current Smoker)"))
    else: recommendations.append(('Smoking', "No", "Protective Factor (Non-Smoker)"))

    if patient_row['AlcoholConsumption'] > 10: recommendations.append(('Alcohol Consumption', f"{patient_row['AlcoholConsumption']:.2f}", "Contributes to Risk (Excessive > 10 units/wk)"))
    else: recommendations.append(('Alcohol Consumption', f"{patient_row['AlcoholConsumption']:.2f}", "Protective Factor (Moderate/Low)"))

    if patient_row['PhysicalActivity'] < 5: recommendations.append(('Physical Activity', f"{patient_row['PhysicalActivity']:.2f}", "Contributes to Risk (Target > 5 hrs/wk)"))
    else: recommendations.append(('Physical Activity', f"{patient_row['PhysicalActivity']:.2f}", "Protective Factor (Active)"))

    if patient_row['DietQuality'] < 4: recommendations.append(('Diet Quality', f"{patient_row['DietQuality']:.2f}", "Contributes to Risk (Poor Nutrition)"))
    elif patient_row['DietQuality'] > 7: recommendations.append(('Diet Quality', f"{patient_row['DietQuality']:.2f}", "Protective Factor (High Quality)"))

    if patient_row['SleepQuality'] < 6: recommendations.append(('Sleep Quality', f"{patient_row['SleepQuality']:.2f}", "Contributes to Risk (Poor Sleep)"))
    else: recommendations.append(('Sleep Quality', f"{patient_row['SleepQuality']:.2f}", "Protective Factor (Good Sleep)"))

    if patient_row['SystolicBP'] > 130 or patient_row['DiastolicBP'] > 80: recommendations.append(('Blood Pressure', f"{int(patient_row['SystolicBP'])}/{int(patient_row['DiastolicBP'])}", "Contributes to Risk (Hypertension)"))
    else: recommendations.append(('Blood Pressure', f"{int(patient_row['SystolicBP'])}/{int(patient_row['DiastolicBP'])}", "Protective Factor (Healthy BP)"))

    if patient_row['CholesterolTotal'] > 200: recommendations.append(('Total Cholesterol', f"{patient_row['CholesterolTotal']:.2f}", "Contributes to Risk (High > 200)"))
    if patient_row['CholesterolLDL'] > 100: recommendations.append(('LDL Cholesterol', f"{patient_row['CholesterolLDL']:.2f}", "Contributes to Risk (Target < 100)"))
    if patient_row['CholesterolHDL'] < 40: recommendations.append(('HDL Cholesterol', f"{patient_row['CholesterolHDL']:.2f}", "Contributes to Risk (Target > 40)"))

    recommendations.sort(key=lambda x: "Risk" in x[2], reverse=True)
    return recommendations[:5]

# 3. Final Model Training for Deployment
X_train, X_test_df, y_train, y_test = train_test_split(X_eval, y_eval, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Note: We fit the scaler on the full training set for the UI
final_model = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=26, random_state=42)
final_model.fit(X_train_scaled, y_train)

# --- NEW UI SECTION ---
def run_cdss_ui():
    print("\n" + "="*50)
    print("      ALZHEIMER'S CDSS PATIENT SEARCH UI")
    print("="*50)
    
    user_input = input("Please enter the Patient ID: ").strip()
    
    # Filter the original dataframe for the ID (handling both string and int types)
    patient_query = df_ui_library[df_ui_library['PatientID'].astype(str) == user_input]
    
    if patient_query.empty:
        print(f"\n[!] Error: Patient ID '{user_input}' not found in the database.")
    else:
        # Prepare the single row for prediction
        # We must drop the same columns we dropped during training
        patient_data_only = patient_query.drop(columns=["PatientID", "DoctorInCharge", "Diagnosis"])
        patient_scaled = scaler.transform(patient_data_only)
        
        # Predict
        prediction = final_model.predict(patient_scaled)[0]
        recs = get_clinical_recommendations(patient_query.iloc[0])
        
        # Format Output
        status = "Alzheimer's Detected" if prediction == 1 else "No Alzheimer's Detected"
        
        print(f"\n>>> REPORT FOR PATIENT ID: {user_input}")
        print(f"CDSS Diagnosis: {status}")
        print("-" * 30)
        print("Concerning Panels:" if prediction == 1 else "Panels to Observe:")
        for feat, val, label in recs:
            print(f"   - {feat}: {val} ({label})")
        print("-" * 30)

# Run the UI
run_cdss_ui()

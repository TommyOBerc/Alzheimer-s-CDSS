# IMPORTS:
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)

# 1. Data prep. :
df_full = pd.read_excel('/content/alzheimers_disease_data.xlsx')

# Excludes the final row from evaluation to treat it as a fresh 'unseen' test case (avoids data leakage!)
df_eval = df_full.iloc[:-1].copy()
df_ui_library = df_full.copy()

# Removes non-clinical identifiers that shouldn't influence the model math:
df_eval_clean = df_eval.drop(columns=["PatientID", "DoctorInCharge"])
X = df_eval_clean.drop(columns=["Diagnosis"])
y = df_eval_clean["Diagnosis"]

# Defines strictly modifiable pannels that patients can actually change:
modifiable_features = [
    'BMI', 'Smoking', 'AlcoholConsumption', 'PhysicalActivity',
    'DietQuality', 'SleepQuality', 'SystolicBP', 'DiastolicBP',
    'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides'
]

# 2. Metrics Eval. (100 Iterations)
# Runs 100 random 80/20 train-test-splits to ensure the accuracy isn't just a "lucky" single run; guages the most accurate metrics scores:
accuracies = []
precisions =[]
recalls = []
f1_s = []
for i in range(100):
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X, y, test_size=0.2, random_state=i)
    scaler_f = StandardScaler()
    X_train_sc = scaler_f.fit_transform(X_train_f)
    X_test_sc = scaler_f.transform(X_test_f)

    # Using Decision Tree with Entropy for high interpretability (BEST MODEL FOR ACCURACY):
    temp_model = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=26, random_state=i)
    temp_model.fit(X_train_sc, y_train_f)
    y_pred = temp_model.predict(X_test_sc)

    accuracies.append(accuracy_score(y_test_f, y_pred))
    precisions.append(precision_score(y_test_f, y_pred))
    recalls.append(recall_score(y_test_f, y_pred))
    f1_s.append(f1_score(y_test_f, y_pred))

print("===== CDSS PERFORMANCE (DT Diagnostic Engine - 100 Iterations) ====")
print(f"Average Accuracy : {np.mean(accuracies):.4f}")
print(f"Average Precision: {np.mean(precisions):.4f}")
print(f"Average Recall   : {np.mean(recalls):.4f}")
print(f"Average F1 Score : {np.mean(f1_s):.4f}")

# 3. Final Model and SHAP-Based Explainer:
#Create new train test split
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X, y, test_size=0.2, random_state=42)
# Scales and trains on the new split to demonstrate SHAP and clinical guidrails in action
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)
final_dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=26, random_state=42)
final_dt_model.fit(X_train_scaled, y_train_final)

# Initializes SHAP explainer to interpret why the model makes specific decisions
explainer = shap.TreeExplainer(final_dt_model)

# Lines 72-84, utilized AI to come up with idea/code
# 4. Recommendation Logic
def get_hybrid_recommendations(patient_row, patient_scaled):
    # Calculates SHAP values for the specific patient
    shap_values = explainer.shap_values(patient_scaled)

    # Identifies which class (Diagnosis 0 [meaning "No Alzheimer's Detected"] or 1 [meaning "Alzheimer's Detected"]) the SHAP values are referring to
    idx_class_1 = 1 if isinstance(shap_values, list) else (0 if len(shap_values.shape) == 2 else 1)
    patient_shap = shap_values[idx_class_1][0] if isinstance(shap_values, list) else shap_values[0, :, 1] if len(shap_values.shape)==3 else shap_values[0]

    feature_impacts = []
    for i, col in enumerate(X.columns): # NOTE: enumerate is a function that allows for the looping through a list, tuple, string, etc. while keeping track of the index of each item
        # ACTIONABILITY FILTER: Only looks at modifiable features (Excludes things like memory loss, family history of Alzheimer's, etc.)
        if col in modifiable_features:
            feature_impacts.append({'feature': col, 'shap_val': patient_shap[i], 'value': patient_row[col]})

    # Sorts features by the absolute strength of their influence on the diagnosis
    top_drivers = sorted(feature_impacts, key=lambda x: abs(x['shap_val']), reverse=True)[:5]

    hybrid_results = []
    for item in top_drivers:
        f = item['feature']
        val = item['value']

        # Clinical Guardrail Logic Check/Engine: Manually checks against medical thresholds, preventing the model from giving medically erroneous information
        is_risk = False
        if f == 'BMI' and (val > 30 or val < 23): is_risk = True
        elif f == 'Smoking' and val == 1: is_risk = True
        elif f == 'AlcoholConsumption' and val > 14: is_risk = True
        elif f == 'PhysicalActivity' and val < 2.5: is_risk = True
        elif f == 'DietQuality' and val < 4: is_risk = True
        elif f == 'SleepQuality' and val < 7: is_risk = True
        elif f == 'SystolicBP' and (val > 130 or val < 60): is_risk = True
        elif f == 'DiastolicBP' and (val > 80 or val < 20): is_risk = True
        elif f == 'CholesterolTotal' and val > 200: is_risk = True
        elif f == 'CholesterolLDL' and val > 100: is_risk = True
        elif f == 'CholesterolHDL' and (val > 60 or val < 40): is_risk = True
        elif f == 'CholesterolTriglycerides' and val < 150: is_risk = True

        # Applies standardized labels based on the Guardrail check
        label = "Contributes to Risk (Address)" if is_risk else "Protective Factor (Keep it up!)"
        hybrid_results.append((f, f"{val:.2f}" if isinstance(val, float) else val, label))

    return hybrid_results
#AI used to create UI function
# 5. UI Function:
def run_hybrid_ui():
    print("\n" + "="*50)
    print("   ALZHEIMER'S CLINICAL DECISION SUPPORT SYSTEM")
    print("="*50)
    pid = input("Enter Patient ID: ").strip()
# Lines 125-129 utilized AI to help create failsafe
    # Searches for Patient ID in the main file/library (not the one with the removed last row)
    match = df_full[df_full['PatientID'].astype(str) == pid]

    if match.empty: #Makes sure patient ID is valid
        print(f"Patient {pid} not found.")
    else: #If patient ID is not in the new 20% test split, comment on possible bias
        if match.index[0] not in X_test_final.index: #Checks whether the patient ID used is in the test data
            print("Note: This patient was used for training. Results may be biased.")

        row = match.iloc[0]
        patient_data = match.drop(columns=["PatientID", "DoctorInCharge", "Diagnosis"])

        # Scales data using the same scaler fitted to training data
        scaled = scaler.transform(patient_data)

        # Generates model prediction
        pred = final_dt_model.predict(scaled)[0]
        status = "Alzheimer's Detected" if pred == 1 else "No Alzheimer's Detected"

        # Generates hybrid recommendations (SHAP + Guardrails)
        recs = get_hybrid_recommendations(row, scaled)

        print(f"\n>>> REPORT FOR PATIENT ID: {pid}")
        print(f"CDSS Diagnosis: {status}")
        print("-" * 30)
        print("Top Actionable Pannels (Modifiable Factors Only):")
        for f, v, l in recs:
            print(f"  - {f}: {v} ({l})")
        print("-" * 30)

# Launches the UI
run_hybrid_ui()

# AI-Generated Underwriting Reports

Model: XGBoost mortality prediction + SHAP + qwen2.5:7b (LangGraph agent)


---

## Applicant 118 (predicted risk 99.4%, actual: Deceased)

1. Underwriting Decision (category + action)
   - Category: Decline
   - Action: Decline or postpone; refer for medical review

2. Key Risk Factors:
   - Age: 80 years, significantly increases risk (+2.232 log-odds impact).
   - MCQ160B_1.0 and MCQ160F_1.0 (unknown specific codes): Both increase risk by +0.418 and +0.229 log-odds, respectively.
   - Low HDL Cholesterol: 3.3 mg/dL, categorized as 'low (higher risk)' (-0.211 log-odds impact).
   - Elevated Serum Creatinine: 1.1 mg/dL, within the normal range but slightly elevated for renal function (+0.278 log-odds impact).

3. Protective Factors:
   - None identified in this case.

4. Summary statement:
The applicant's overall risk is extremely high, primarily driven by advanced age and certain lifestyle or health factors indicated by the SHAP analysis. The low HDL cholesterol and slightly elevated serum creatinine levels further contribute to this high-risk profile, necessitating a decline in underwriting with consideration for medical review.

---

## Applicant 12662 (predicted risk 0.2%, actual: Alive)

1. Underwriting Decision (category + action)
   - Category: Preferred
   - Action: Approve at preferred (lowest) premium rates.

2. Key Risk Factors (most important first)
   - Age: 27 years, which significantly decreases the risk due to a younger age.
   - Serum Albumin: 4.8 g/dL, within the normal range for general clinical laboratory reference.
   - Urine Albumin: 1.82 mg/L, also within the normal range.

3. Protective Factors
   - Glycated Hemoglobin (HbA1c): 3.64%, which is below the diabetic range and decreases risk.
   - Waist Circumference: 79.2 cm, a value that does not indicate obesity or increased risk.
   - Serum Sialic Acid: 2.6 mg/dL, within normal ranges and thus does not increase risk.

4. Summary statement
The applicant's overall mortality risk is extremely low at 0.2%, as indicated by the SHAP analysis which highlights protective factors such as a young age, normal serum albumin levels, and no signs of microalbuminuria or obesity. These factors contribute to an underwriting decision of Preferred, allowing for approval at the lowest premium rates.

---

## Applicant 6339 (predicted risk 10.8%, actual: Alive)

1. Underwriting Decision (category + action)
   - Category: Standard
   - Action: Approve at standard premium rates

2. Key Risk Factors:
   - Urinary Albumin-to-Creatinine Ratio (URXUCR): SHAP contribution = -0.214, value = 5.22. This ratio is within the normal range, contributing to a lower risk.
   - Glycated Hemoglobin (HbA1c) level: SHAP contribution = +0.200, value = 4.09%. The HbA1c level of 4.09% is in the non-diabetic range, but still slightly increases the risk.

3. Protective Factors:
   - Age: SHAP contribution = -1.754, value = 42.0 years. A younger age significantly decreases the risk.
   - Smoking Status (SMQ020_1.0): SHAP contribution = -0.204, value = 0.0. The applicant is a non-smoker, which reduces the risk.

4. Summary statement:
The overall mortality risk for this applicant is relatively low at 10.8%, placing them in the 'Standard' category. Key factors such as age and smoking status contribute to a lower risk profile, while slightly elevated HbA1c levels and urinary albumin-to-creatinine ratio indicate some minor risks. Based on these factors, approval at standard premium rates is recommended.

---

## Applicant 8673 (predicted risk 10.8%, actual: Alive)

1. Underwriting Decision (category + action)
   - Category: Standard
   - Action: Approve at standard premium rates

2. Key Risk Factors:
   - Age: 47 years old, which decreases the risk significantly as it is younger than typical high-risk age groups.
   - Glycated Hemoglobin (HbA1c): 3.47%, falling outside the clinical reference range for diabetes, indicating normal glucose control and decreasing the risk.
   - Waist Circumference: 73.9 cm, which does not fall within a clinically significant range that would increase risk; however, it is on the higher end of what might be considered healthy.

3. Protective Factors:
   - Smoking Status (SMQ020_1.0): Non-smoker, decreasing the risk.
   - Systolic Blood Pressure: 2.64 mmHg, slightly elevated but within a range that does not significantly increase mortality risk.
   - Body Mass Index (BMXWT): 4.04 kg/m², which is within a healthy weight range and decreases the risk.

4. Summary statement:
The applicant's overall risk profile is favorable with several protective factors such as younger age, normal HbA1c levels, non-smoking status, and a healthy body mass index. These factors outweigh the minor increases in risk from slightly elevated systolic blood pressure and other less significant contributors. Based on these findings, approval at standard premium rates is recommended.

---

## Applicant 10891 (predicted risk 10.9%, actual: Alive)

1. Underwriting Decision (category + action)
   - Category: Standard
   - Recommended Action: Approve at standard premium rates.

2. Key Risk Factors:
   - Age: 57 years, which falls into the middle-aged category and slightly increases risk.
   - Urine Albumin (URXUMA): 1.33 mg/L, within the normal range, indicating no significant kidney issues.

3. Protective Factors:
   - Poverty Income Ratio: 5.0, which decreases risk by suggesting a stable economic situation.
   - Marital Status: Single (MARITAL_STATUS_1.0), also contributing to decreased risk.
   - Never Smoker (SMQ020_1.0): This protective factor further reduces the overall mortality risk.

4. Summary Statement:
The applicant's predicted mortality risk of 10.9% places them in a standard category, and their age is the primary factor slightly increasing this risk. However, several protective factors such as normal urine albumin levels, stable economic status, and being a non-smoker contribute to an overall favorable underwriting profile, leading to approval at standard premium rates.

---

## Applicant 17300 (predicted risk 10.9%, actual: Alive)

1. **Underwriting Decision**: Category 'Standard', Recommended Action: Approve at standard premium rates.

2. **Key Risk Factors**:
   - **Hemoglobin A1c (HbA1c)**: 1.76% is within the normal range, but slightly elevated, potentially indicating early signs of glucose metabolism issues.
   - **Platelets**: 158 cells/µL is in a category that suggests higher risk, which may indicate underlying inflammatory or hematological conditions.

3. **Protective Factors**:
   - **Age**: 43 years old, which decreases the overall mortality risk significantly.
   - **Albumin**: 4.9 g/dL, within normal range and likely protective against certain health risks.
   - **Creatinine**: 0.36 mg/dL, indicating good kidney function.

4. **Summary Statement**:
The applicant's overall risk is categorized as standard with a predicted mortality probability of approximately 10.9%. The key factors contributing to this assessment include slightly elevated HbA1c levels and high platelet counts, which may indicate potential health concerns. However, the protective factors such as age and normal albumin and creatinine levels help mitigate these risks. Given the overall standard risk category, approval is recommended at standard premium rates.

---

## Applicant 1176 (predicted risk 10.9%, actual: Alive)

1. **Underwriting Decision (category + action)**:
   - Category: Standard
   - Action: Approve at standard premium rates

2. **Key Risk Factors**:
   - Age 55 years: Decreases risk by a significant margin, as older age is generally associated with higher mortality but this applicant's age does not pose an excessive risk.
   - Waist Circumference of 81.4 cm: While no specific clinical reference range was found for waist circumference, the decrease in risk attributed to this factor suggests it is within a healthy range and does not indicate obesity or other related health issues that could increase mortality risk.

3. **Protective Factors**:
   - Urine Albumin of 1.48 mg/L: Within the normal range (0-30 mg/L), indicating no signs of kidney damage or proteinuria, which is a protective factor against increased mortality.
   - Glycated Hemoglobin (HbA1c) of 2.4%: Also within the normal range (<5.7%), suggesting good blood glucose control and reducing the risk of diabetes-related complications.

4. **Summary Statement**:
The overall underwriting decision is to approve this applicant at standard premium rates, given their low predicted mortality risk of 10.9%, which falls into the 'Standard' category. Key factors contributing to this favorable outcome include a normal waist circumference and urine albumin levels, as well as an age that does not significantly increase the risk. These protective factors outweigh other minor contributors such as marital status, leading to a standard rating.

---

## Applicant 16988 (predicted risk 10.9%, actual: Alive)

1. **Underwriting Decision (category + action)**: The predicted mortality risk of 10.9% categorizes the applicant as 'Standard', and the recommended action is to approve at standard premium rates.

2. **Key Risk Factors**:
   - **Age**: Age 30 decreases the risk significantly, contributing a SHAP value of -2.104.
   - **Height (BMXHT)**: Height of 185.5 cm does not have a specific clinical reference range available but is within normal adult height ranges and does not contribute to increased risk.
   - **Systolic Blood Pressure (LBXSASSI)**: Systolic blood pressure of 4.23 mmHg increases the risk, contributing a SHAP value of +0.349.
   - **Weight (BMXWT)**: Weight of 5.07 kg increases the risk, contributing a SHAP value of +0.231.

3. **Protective Factors**:
   - **Cycle Data from 2011-2012 (CYCLE_2011-2012)**: This factor decreases the risk by a SHAP value of -0.258.
   - **Urinary Uric Acid (URXUCR)**: Urinary uric acid level of 5.14 mg/dL decreases the risk, contributing a SHAP value of -0.215.

4. **Summary Statement**: The applicant has been categorized as 'Standard' with an overall mortality risk of 10.9%, which is within acceptable limits for standard underwriting. Key factors such as systolic blood pressure and weight slightly increase the risk, but are offset by protective factors like cycle data from 2011-2012 and urinary uric acid levels. The applicant will be approved at standard premium rates.
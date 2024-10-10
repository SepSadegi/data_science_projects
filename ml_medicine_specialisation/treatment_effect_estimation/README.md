# Treatment Effect Estimation Using Machine Learning

This project is about machine-learning techniques for predicting the effect of treatments on individual patients. It consists of: 

### Pandas Medical Dataset Practice

This script demonstrates basic Pandas operations using a medical dataset.

- **Code:** pandas_medical_data_analysis.py
- **Data:** 
    - `dummy-data.csv`
    
### Model Training/Tuning Basics with Sklearn

This script demonstrates basic Pandas operations using a medical dataset.

- **Code:** pandas_medical_data_analysis.py
- **Data:** 
    - `dummy-data.csv`
___

Here are the key points:

### Randomized Controlled Trials (RCTs):

- **Objective:** To understand the causal effect of a treatment on a population.

- **Design:**
    - **Treatment Arm:** Group receiving the new treatment.
    - **Control Arm:** Group not receiving the new treatment (could receive a placebo or standard of care).

- **Measuring Outcomes:** Look at the percentage of patients who have a heart attack after one year.

- **Calculating Absolute Risk:**
    - **Treatment Group:** Absolute risk = 2% (0.02).
    - **Control Group:** Absolute risk = 5% (0.05).

- **Randomization:** Randomly assigns people to groups to make sure they are similar in terms of characteristics (like age or health), reducing bias.

- **Group Comparison:**
    - **Check:** Compare mean and standard deviation of relevant characteristics (e.g., age, blood pressure) between groups.
    - **Goal:** Make sure differences in results are due to the treatment, not pre-existing differences.
    
***

### Absolute Risk Reduction (ARR):

- **Purpose:** Shows how much better the treatment is compared to no treatment.

- **How to Calculate:**
    - **Formula:** ARR = Absolute Risk in Control Group - Absolute Risk in Treatment Group.
    - **Example Calculation:** 0.05 (control) - 0.02 (treatment) = 0.03 (ARR). ARR of 0.03 means a 3% reduction in risk due to the treatment.

---

###  Number Needed to Treat (NNT):

- **Purpose:** Indicates how many people need to be treated to prevent one bad outcome.

- **How to Calculate:**
    - **Formula:** NNT = 1 / ARR.
    - **Example:** An ARR of 0.03 gives an NNT of 33.3, meaning 33.3 people need to be treated to prevent one heart attack.
    
___

###  Statistical Significance:

- **P-Value:**
    - **Definition:** Probability of observing the measured effect (or more extreme) if the treatment actually has no effect.
    - **Interpretation:** A p-value < 0.001 indicates a less than 0.1% chance that the observed effect is due to random chance, suggesting statistical significance.

- **Effect of Sample Size:**
    -**Small Sample Size:** Higher p-value, less statistical significance.
    -**Large Sample Size:** Lower p-value, higher statistical significance.
    
___

###  Causal Inference:

- **Purpose:** Determine the effect of a treatment on an individual patient.

**Possible Treatment Outcomes:**

1. **Benefit:** Treatment prevents the adverse outcome (e.g., heart attack) while no treatment would have led to it.
2. **No Effect:** Treatment or no treatment leads to the same outcome (e.g., heart attack happens or doesn’t happen regardless of treatment).
3. **Harm:** Treatment causes the adverse outcome while no treatment would have prevented it.

___

#### Neyman-Rubin Causal Model

- **Y_i(1):** Outcome for patient with treatment.
- **Y_i(0):** Outcome for patient without treatment.

##### Unit Level Treatment Effect

- **Benefit (-1):** Treatment is beneficial.
- **No Effect (0):** Treatment has no effect.
- **Harm (1):** Treatment is harmful.

___

###  Average Treatment Effect (ATE)

- **Fundamental Problem:** We cannot observe both potential outcomes (with and without treatment) for the same patient.
  - **Observed Outcome:** What actually happens to the patient.
  - **Counterfactual Outcome:** What would have happened if the patient had received the treatment.
  
- **Estimating ATE:**
  - **Data from Randomized Controlled Trials (RCTs):** Allows estimation of ATE by comparing groups.
  - **Grouping:** Separate patients into treated and control groups, then calculate the mean outcome for each group.

- **Calculation:**
  - **Observed Outcomes:** For treated group, mean outcome might be 0.32. For control group, mean outcome might be 0.51.
  - **Formula:** ATE = Mean(Y_i | W_i = 1) - Mean(Y_i | W_i = 0).
  - **Example:** ATE = 0.32 - 0.51 = -0.19.

- **Relation to Absolute Risk Reduction (ARR):**
  - **ARR:** The absolute difference in risk between treatment and control.
  - **Formula:** ARR = -ATE.
  - **Example:** If ATE is -0.19, ARR is 0.19, indicating a 19% reduction in risk due to treatment.

- **Importance:**
  - ATE provides a measure of the average impact of the treatment on a population.
  - A negative ATE indicates a reduction in risk due to the treatment.

- **Individualized Estimates:** While RCT data helps estimate ATE for a population, more personalized methods are needed for individual predictions.


___

###  Conditional Average Treatment Effect (CATE)

- **Objective:** Estimate how treatment effects vary based on specific patient characteristics (e.g., age).
  - **Example:** Determine if a treatment works better for patients who are 56 years old.

- **Conditional Average Treatment Effect (CATE):** The expected difference in outcomes given specific patient features (e.g., age = 56).

#### Estimation in Randomized Control Trials (RCTs)

1. **Estimate Mean Outcomes:**
   - **Treatment Group:** Calculate the average outcome for patients with the specific feature (e.g., age = 56).
   - **Control Group:** Calculate the average outcome for the same feature.

2. **Calculate CATE:**
   - **Formula:** CATE = Mean outcome with treatment - Mean outcome without treatment for the specified feature.

3. **Challenges:**
   - **Insufficient Data:** Small sample sizes for specific features (e.g., few patients aged 56 in both groups).
   - **Multiple Features:** Difficulty in estimating effects with multiple patient characteristics (e.g., age and blood pressure).

#### Solution

- **Use of Models:** Learn relationships between patient features and outcomes using machine learning models.
  - **Features (X):** Variables like age and blood pressure.
  - **Feature Values (x):** Specific values for these features (e.g., age = 56, blood pressure = 130).

- **Estimating CATE:**
  - **Treatment Response Function (μ̂1):** Estimates expected outcome with treatment given features.
  - **Control Response Function (μ̂0):** Estimates expected outcome without treatment given features.
  - **Formula:** CATE = μ̂1(x) - μ̂0(x).

By using these functions, we can estimate the CATE even when direct data is sparse, providing a more individualized understanding of treatment effects.

___

### T-Learner

- **Objective:** Estimate the Conditional Average Treatment Effect (CATE) using two separate models.

- **Base Learners:** Models used to predict outcomes based on patient features.
  - **Examples:** Decision trees, linear models.
  - **Function:** Each model learns from different parts of the data:
    - **μ̂1:** Model for patients in the treatment group (input: age, blood pressure, output: outcome).
    - **μ̂0:** Model for patients in the control group (input: age, blood pressure, output: outcome).

- **Training Process:**
  - **Data Splitting:** Divide dataset into training and validation sets.
  - **Prognostic Models:**
    - **μ̂1:** Estimates risk of adverse event with treatment.
    - **μ̂0:** Estimates risk of adverse event without treatment.

- **Example Using Decision Trees:**
  - **Model for Treatment:** Predicts risk based on age and blood pressure for treated patients.
  - **Model for Control:** Predicts risk based on age and blood pressure for control patients.

- **Estimating CATE:**
  - **Procedure:**
    - **Input Features:** For a patient with age 56 and blood pressure 130.
    - **Risk Score with Treatment (μ̂1):** E.g., 0.21.
    - **Risk Score without Treatment (μ̂0):** E.g., 0.45.
    - **CATE Calculation:** Subtract the two risk scores: 0.21 - 0.45 = -0.24.

- **Method Name:** The approach of using two separate models and taking their difference to estimate CATE is known as the **T-Learner**.

___

### S-Learner

- **Objective:** Estimate the Conditional Average Treatment Effect (CATE) using a single model.

- **Single Model Approach:**
  - **Model Function:** A single model (μ̂) is trained to predict outcomes.
  - **Treatment as a Feature:** The treatment indicator (0 or 1) is included as a feature along with other patient characteristics (e.g., age, blood pressure).

- **Estimating Outcomes:**
  - **With Treatment (W=1):** Estimate the outcome when treatment is given.
  - **Without Treatment (W=0):** Estimate the outcome when treatment is not given.
  - **CATE Calculation:** Difference between these two estimates.

- **Training Process:**
  - **Data Use:** Learn from data of both treated and control patients.
  - **Data Splitting:** Divide into training and validation sets to build and test the model.
  - **Model Type:** Typically a single decision tree.

- **Example Calculation:**
  - **For a Patient (Age 56, Blood Pressure 130):**
    - **Expected Outcome with Treatment (μ̂1):** E.g., 0.4.
    - **Expected Outcome without Treatment (μ̂0):** E.g., 0.5.
    - **CATE Estimate:** 0.4 - 0.5 = -0.1.

- **Challenges:**
  - **Feature Usage:** The model may not always use the treatment feature effectively, potentially leading to a treatment effect estimate of zero if the treatment feature is not utilized.
  - **Comparison with T-Learner:** Unlike the S-Learner, the T-Learner uses two separate models (one for treatment and one for control), which reduces the risk of the model ignoring the treatment effect.

- **Usefulness:** Both S-Learner and T-Learner methods are simple approaches for estimating individualized treatment effects, useful for personalized treatment decisions.

___

### Evaluate Individualized Treatment Effect

- **Objective:** Evaluate the accuracy of individualized treatment effect (ITE) estimates.

- **Challenge:** We cannot observe both treatment and control outcomes for the same patient. To assess ITE estimates, we need to approximate the counterfactual outcome (Y(0)) for treated patients.

- **Evaluation Methods:**
  1. **Matching by Features:**
     - **Process:** Find a control patient with similar features (age, blood pressure) to the treated patient.
     - **Example:** Match based on age and blood pressure.
  2. **Matching by Estimated Treatment Effect:**
     - **Process:** Find a control patient with a similar estimated treatment effect.
     - **Example:** Match based on estimated ITE rather than direct features.

- **Evaluation Procedure:**
  - **Select a Match Pair:** Choose one treated patient and one control patient from the matched groups.
  - **Compare Outcomes:**
    - **Factual Outcomes:** For each patient, observe the outcome under their assigned condition.
    - **Estimate Comparison:** Calculate the difference between the outcomes (Y(1) - Y(0)) for the matched pair.
  - **Example Results:**
    - **Pair 1:** Difference = -1 (Observed benefit).
    - **Pair 2:** Difference = 1 (Observed harm).
    - **Pair 3:** Difference = 0 (Observed no effect).

- **Assessment:** Evaluate whether higher predicted benefits correspond to observed benefits. The goal is to determine if the estimated ITE aligns with the actual observed outcomes.

___

### C-for-Benefit

- **Objective:** Assess whether higher predicted benefits correspond to higher observed benefits using the c-for-benefit metric.

- **Concept:**
  - Similar to the C-Index but adapted for three possible outcomes: 1 (harm), -1 (benefit), and 0 (no effect).
  - **Negative Outcome:** Corresponds to observed benefit (similar to negative treatment effect prediction).
  - **Positive Outcome:** Corresponds to observed harm (similar to positive treatment effect prediction).

- **Types of Pairs:**
  1. **Concordant Pair:**
     - **Definition:** A pair where the prediction of higher benefit aligns with a better observed outcome.
     - **Example:** If one pair predicts greater benefit and this pair actually shows a better outcome, it’s concordant.
  2. **Not Concordant Pair:**
     - **Definition:** A pair where the prediction of higher benefit does not match with the worse observed outcome.
     - **Example:** If a higher predicted benefit pair actually shows a worse outcome, it’s not concordant.
  3. **Risk Tie:**
     - **Definition:** A pair where the predicted treatment effect is the same, but outcomes differ.
     - **Issue:** Cannot determine which pair should have a higher score due to identical predictions but differing outcomes.

- **Evaluation:**
  - **Permissible Pairs:** Only compare pairs with different observed outcomes.
  - **C-for-Benefit calculated as:**
  
$$
\text{C-index} = \frac{\text{Number of Concordant Pairs} + 0.5 \times \text{Number of Risk Ties}}{\text{Number of Permissible Pairs}}
$$

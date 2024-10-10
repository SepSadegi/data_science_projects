# Model Evaluation Metrics Summary

## 1. True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
- **TP (True Positives)**: The number of correctly predicted positive cases.
- **TN (True Negatives)**: The number of correctly predicted negative cases.
- **FP (False Positives)**: The number of cases incorrectly predicted as positive.
- **FN (False Negatives)**: The number of cases incorrectly predicted as negative.

## 2. Accuracy
- **Accuracy**: The proportion of correctly classified cases (both positive and negative) out of the total number of cases.

  $`
  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
  `$

- **Equation for Accuracy**:

  $`
  \text{Accuracy} = (\text{Sensitivity} \times \text{Prevalence}) + (\text{Specificity} \times (1 - \text{Prevalence}))
  `$

## 3. Prevalence
- **Prevalence**: The proportion of actual positive cases in the population.

  $`
  \text{Prevalence} = \frac{TP + FN}{Total \ Population}
  `$

## 4. Sensitivity (True Positive Rate)
- **Sensitivity**: The ability of the model to correctly identify positive cases.

  $`
  \text{Sensitivity} = \frac{TP}{TP + FN}
  `$

- Sensitivity tells us how good the model is at correctly identifying patients who actually have the disease and labels them as having the disease.
- A sensitivity of 1 means the model identifies all diseased patients as having the disease, and does not miss any cases.

## 5. Specificity (True Negative Rate)
- **Specificity**: The ability of the model to correctly identify negative cases.

  $`
  \text{Specificity} = \frac{TN}{TN + FP}
  `$

- Specificity tells us how good the model is at correctly identifying the healthy patients as not having the disease.

## 6. Positive Predictive Value (PPV)
- **PPV (Precision)**: The proportion of positive predictions that are true positives.

  $`
  \text{PPV} = \frac{TP}{TP + FP}
  `$

- **PPV Equation**:

  $`
  \text{PPV} = \frac{(\text{Sensitivity} \times \text{Prevalence})}{(\text{Sensitivity} \times \text{Prevalence}) + ((1 - \text{Specificity}) \times (1 - \text{Prevalence}))}
  `$

## 7. Negative Predictive Value (NPV)
- **NPV**: The proportion of negative predictions that are true negatives.

  $`
  \text{NPV} = \frac{TN}{TN + FN}
  `$

## 8. ROC Curve (Receiver Operating Characteristic)
- **ROC Curve**: A graphical plot that shows the trade-off between sensitivity (True Positive Rate) and 1-specificity (False Positive Rate) for different threshold settings of a model.

## 9. AUC-ROC (Area Under the ROC Curve)
- **AUC-ROC (c-statistic)**: A single value summarizing the performance of the model across all thresholds, representing the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance.

  - **AUC = 1**: Perfect model.
  - **AUC = 0.5**: No discrimination (random guessing).

## 10. Confidence Intervals
- **Confidence Intervals**: A range of values derived from a sample that is likely to contain the true population parameter with a certain level of confidence (e.g., 95% confidence interval).

- The width of the confidence interval depends on the variance of the normal distribution. The variance of each sample is identical, but the variance of the average is divided by \(n\) (the sample size).

  - **Larger Sample Size**: The variance of the average decreases as the sample size increases, making the confidence interval **tighter** (narrower).

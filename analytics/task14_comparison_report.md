# Task 14 – Classification vs Regression

## Classification Metrics

| model               |   accuracy |   precision |   recall |   f1_score |   roc_auc |
|:--------------------|-----------:|------------:|---------:|-----------:|----------:|
| Logistic Regression |     0.8090 |      0.7833 |   0.6912 |     0.7344 |    0.8610 |
| Decision Tree       |     0.7640 |      0.7600 |   0.5588 |     0.6441 |    0.8374 |
| Random Forest       |     0.8090 |      0.7656 |   0.7206 |     0.7424 |    0.8196 |

Classification metrics measure the performance of a binary prediction problem. Accuracy measures overall correct predictions, while precision, recall and F1 focus on the positive class. ROC-AUC measures the model's ability to rank positive examples above negative examples across classification thresholds.

## Regression Metrics

| model             |     mae |    rmse |     r2 |   adjusted_r2 |
|:------------------|--------:|--------:|-------:|--------------:|
| Linear Regression | 21.0986 | 41.7021 | 0.3482 |        0.3091 |

Regression metrics evaluate prediction error for a continuous target. MAE gives the average absolute prediction error, RMSE penalizes larger errors more strongly, and R² measures the proportion of target variance explained by the model. Adjusted R² also accounts for the number of predictors in the model.

## Final Recommendation

The classification task is best represented by Random Forest, which achieved an F1 score of 0.7424 and a ROC-AUC of 0.8196. Its recall of 0.7206 indicates how effectively the model identifies passengers who survived. For the fare regression task, Linear Regression achieved an R² of 0.3482, with an RMSE of 41.7021 and MAE of 21.0986. Therefore, the classification and regression tasks should be evaluated using their respective metrics rather than comparing raw score values across the two task types.

### Best Classification Model

- Model: Random Forest
- F1 Score: 0.7424
- ROC-AUC: 0.8196
- Recall: 0.7206

### Best Regression Model

- Model: Linear Regression
- R²: 0.3482
- RMSE: 41.7021
- MAE: 21.0986

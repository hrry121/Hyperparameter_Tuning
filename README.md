# 🔍 Hyperparameter Tuning with Cross-Validation & MLflow Tracking

This project demonstrates hyperparameter tuning on a regression model using **Scikit-Learn's** `GridSearchCV`, evaluated with **Mean Squared Error (MSE)**, and tracked using **MLflow** for experiment management.

---

## 📌 Project Highlights

- **Model Optimization** via Grid Search over a defined hyperparameter space  
- **Cross-Validation** with `cv=3` to ensure model generalization  
- **Metric Used**: Mean Squared Error (MSE) for evaluation  
- **Experiment Tracking** with MLflow (parameters, metrics, artifacts, and models)  
- **Best Model Selection** based on the lowest average MSE across folds

---

## 📂 Project Structure

```bash
├── mlruns/              # MLflow logs (auto-generated)
├── house_prediction.py  # Script to run tuning + tracking
├── requirements.txt     # Required packages
└── README.md            # You're here!

---

## 🛠️ Tech Stack

- Python
- Scikit-Learn  
- MLflow  
- Pandas  

---

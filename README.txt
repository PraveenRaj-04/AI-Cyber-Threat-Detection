🛡️ AI-Based Cyber Threat Detection System

An AI-driven system that detects cyber attacks from network traffic using machine learning techniques.
This project uses the NSL-KDD dataset to train, evaluate, and compare multiple ML models for intrusion detection.

📌 Project Overview

Cyber attacks such as DoS, Probe, R2L, and U2R pose serious threats to modern networks.
This project applies machine learning algorithms to automatically identify malicious network activity and help improve cybersecurity monitoring.

🎯 Objectives

Detect cyber attacks from network traffic data

Train and evaluate machine learning models for intrusion detection

Compare multiple ML algorithms

Analyze important network features used in attack detection

Provide a reusable trained model for future detection
📊 Dataset Information

Dataset Name: NSL-KDD

Type: Network intrusion detection dataset

Classes:

DoS (Denial of Service)

Probe

R2L (Remote to Local)

U2R (User to Root)

Normal

Dataset Location:

data/raw/nsl_kdd_dataset.csv
⚙️ Technologies Used

Programming Language: Python

Libraries:

Pandas

NumPy

Scikit-learn

Joblib

Matplotlib

Algorithms:

Random Forest

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Logistic Regression
🚀 How to Run the Project (Step-by-Step)
🔹 Step 1: Open Terminal

Open terminal or command prompt inside the project folder:

ai-cyber-threat-detection

🔹 Step 2: Activate Virtual Environment

Windows

venv\Scripts\activate

🔹 Step 3: Go to Source Folder
cd src

🔹 Step 4: Preprocess the Dataset
python preprocess.py


Output:

data/processed/train_processed.csv
data/processed/test_processed.csv

🔹 Step 5: Train Random Forest Intrusion Detection Model
python train_intrusion_rf.py


Output:

models/random_forest_model.joblib


Console output includes:

Accuracy

Classification report

🔹 Step 6: Train and Compare Multiple Models
python train_multi_models.py


Output:

reports/model_comparison.csv

🔹 Step 7: Generate Feature Importance
python generate_feature_importance.py


Output:

reports/feature_importance.csv
reports/feature_importance.png

 FINAL SHORT PATH:
cd src
python preprocess.py
python train_intrusion_rf.py
python train_multi_models.py
python generate_feature_importance.py

.

🧠 Model Training Process (Step-by-Step)

This section explains how the machine learning model is trained in the AI-Based Cyber Threat Detection System.

🔹 Step 1: Load the Raw Dataset

The NSL-KDD dataset is loaded from the data/raw/ directory.

This dataset contains network traffic records labeled as normal or different types of cyber attacks.

Input file:

data/raw/nsl_kdd_dataset.csv

🔹 Step 2: Data Preprocessing

Before training, the raw data must be cleaned and converted into a machine-learning-ready format.

Command:

python preprocess.py


Preprocessing includes:

Handling missing values

Encoding categorical features into numerical values

Ensuring the label column is correctly identified

Splitting the dataset into training and testing sets

Output files generated:

data/processed/train_processed.csv
data/processed/test_processed.csv

🔹 Step 3: Load Processed Training Data

The processed training dataset is loaded from train_processed.csv.

Features (X) and labels (y) are separated for model training.

🔹 Step 4: Train the Random Forest Model

A Random Forest classifier is used as the primary intrusion detection model.

Command:

python train_intrusion_rf.py


During training:

The model learns patterns from network traffic features

It builds multiple decision trees to classify traffic into attack categories

The trained model is evaluated using test data

🔹 Step 5: Evaluate Model Performance

After training, the model is evaluated using:

Accuracy

Precision

Recall

F1-score

A classification report is printed in the terminal to show how well the model detects different types of attacks.

🔹 Step 6: Save the Trained Model

The trained Random Forest model is saved for future use.

This allows the model to be reused without retraining.

Saved model file:

models/random_forest_model.joblib

🔹 Step 7: Train and Compare Multiple Models (Optional)

To improve analysis, multiple machine learning models are trained and compared.

Command:

python train_multi_models.py


Models compared:

Random Forest

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Logistic Regression

Output:

reports/model_comparison.csv

🔹 Step 8: Feature Importance Analysis

This step identifies which features contribute most to cyber threat detection.

Command:

python generate_feature_importance.py


Outputs:

reports/feature_importance.csv
reports/feature_importance.png

📌 Summary of Training Workflow
Raw Dataset
   ↓
Data Preprocessing
   ↓
Train/Test Split
   ↓
Random Forest Model Training
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Model Comparison & Feature Analysis

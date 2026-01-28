STEP 0: Open Terminal / Command Prompt

Open CMD inside the project folder:

ai-cyber-threat-detection
STEP 1: Activate Virtual Environment

Windows

venv\Scripts\activate
STEP 2: Go to Source Folder
cd src

🔹 STEP 3: Run Data Preprocessing
python preprocess.py


✔ Creates train_processed.csv & test_processed.csv

🔹 STEP 4: Train Random Forest Intrusion Model
python train_intrusion_rf.py


✔ Trains model
✔ Saves model in models/

🔹 STEP 5: Train & Compare Multiple Models
python train_multi_models.py


✔ Compares RF, SVM, KNN, Logistic Regression
✔ Saves comparison in reports/

🔹 STEP 6: Generate Feature Importance
python generate_feature_importance.py


✔ Generates feature importance CSV & graph

🔹 STEP 7: Project Finished 🎉

Your outputs are saved in:

data/processed/
models/
reports/
 FINAL SHORT PATH:
cd src
python preprocess.py
python train_intrusion_rf.py
python train_multi_models.py
python generate_feature_importance.py

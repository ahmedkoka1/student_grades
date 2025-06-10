📚 Student Grade Prediction (G3) using Linear Regression

This project is a full pipeline for predicting the final grade (G3) of students based on various academic, personal, and social factors. It includes data preprocessing, feature engineering, model training using Linear Regression, and a Streamlit-powered web application for real-time predictions.

🚀 Features

🔍 Exploratory Data Analysis and Cleaning

📈 Feature Engineering (custom derived features)

🧠 Model Training using Scikit-learn's Linear Regression

🧪 Feature Selection with Recursive Feature Elimination (RFE)

🌐 Web Interface built with Streamlit

💾 Model serialization using Joblib

📁 Project Structure

project_linearregression_full/
├── student_performance_clean.csv   # Cleaned student data
├── Untitled15.ipynb                # Notebook with EDA and model training
├── student_model.pkl               # Trained model pipeline
├── app.py                          # Streamlit web application
├── README.md                       # Project documentation

🧠 Model Details

Algorithm: Linear Regression

Feature Selection: RFE (Top 10 features)

Evaluation Metric: R² score

Achieved R²: ~0.91 on test data

📊 Input Features (via Streamlit app)

Includes demographic, academic, and lifestyle inputs such as:

School, Sex, Age

Parental Education, Support

Study Time, Absences, Failures

G1 and G2 grades (required for G3 prediction)

Custom features: g1_absences, g1_health, Medu_fedu, final_degree

💡 How to Use

1. Install Requirements

pip install -r requirements.txt

If you don’t have a requirements.txt, make sure you have at least:

pip install streamlit scikit-learn pandas numpy

2. Run the App

streamlit run app.py

3. Upload/Ensure model file

Make sure student_model.pkl is in the same directory.

📝 Model Training (Notebook)

Load and clean the dataset

Remove outliers using Z-score filtering

Apply log transformation on absences

Engineer additional features to enhance model performance

Encode categorical variables with OneHotEncoding

Normalize numerical data

Fit a Linear Regression model and select top features with RFE

📈 Results

R² Score on Test Set: 91%

Strong correlation observed between G1, G2 and final G3 grade

📬 Contact

For any questions, suggestions, or improvements:
Developer: [Your Name]Email: your.email@example.com

📜 License

This project is open-source and available under the MIT License.

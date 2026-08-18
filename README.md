# PriceGuard-AI
AI-powered product price analysis and anomaly detection application built with Python and Streamlit.
# Overview
PriceGuard AI is a machine-learning-based application designed to analyze product prices, identify unusual price patterns, and provide insights to support smarter purchasing decisions.
# Features
📊 Product price analysis
📈 Price prediction using machine learning
🏷️ Price category classification
🚨 Anomaly detection for unusual prices
📉 Data processing and visualization
💾 Model saving and loading using Joblib
🖥️ Interactive Streamlit interface
# Machine Learning Models
The project includes:
Regression — for price prediction
Classification — for categorizing product prices
Anomaly Detection — for identifying unusual price values
# Technologies Used
Python
Streamlit
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Joblib
# Project Structure
PriceGuard-AI/
├── README.md
├── app.py
├── Dataset.csv
├── Prices.csv
├── requirements.txt
├── .gitignore
└── src/
    ├── model.py
    ├── config.py
    ├── data_processing.py
    ├── models.py
    └── utils.py
# How to Run
pip install -r requirements.txt
streamlit run app.py
# Future Improvements
Real-time product price tracking
Integration with e-commerce APIs
Price history visualization
Improved recommendation system
Automated price alerts

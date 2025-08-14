# E-Commerce Customer Segmentation & LTV Prediction

This project analyzes the Olist E-commerce dataset to segment customers using RFM analysis, predict their 12-month Customer Lifetime Value (LTV), and provide a tool for data-driven marketing decisions.

* **Source**: [Kaggle - Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

💡 Problem Statement
The core business problem is to move beyond generic marketing and understand customers on a deeper level. By segmenting customers based on their purchasing behavior, we can tailor marketing campaigns, improve customer retention, and optimize marketing spend by focusing on high-value segments.

🚀 Solution
This project implements an end-to-end data science pipeline:

Data Preparation: A hybrid SQL-Python approach is designed. A SQL script (sql/pre_aggregate.sql) demonstrates how to efficiently pre-process and aggregate raw data within a database. The Python notebook then performs the equivalent steps for a self-contained analysis.

RFM Analysis: Customers are scored based on their Recency, Frequency, and Monetary value.

Customer Segmentation: K-Means clustering is used to group customers into four distinct, actionable segments: Champions, Potential Loyalists, New Customers, and At-Risk.

LTV Prediction: A BG/NBD and Gamma-Gamma model from the lifetimes library is trained to predict the 12-month LTV for each customer.

Interactive Dashboard: A Streamlit application provides an interactive interface to explore customer segments and their value, demonstrating the project's business impact.

🛠️ Tech Stack
Data Analysis: Python, Pandas, NumPy

Database: SQL (MySQL dialect)

Machine Learning: Scikit-learn (K-Means), Lifetimes (BG/NBD, Gamma-Gamma)

Data Visualization: Matplotlib, Seaborn

Web App: Streamlit

📊 Key Insights
The "Champions" segment, while being a smaller group, holds the highest predicted LTV, making them the most valuable target for loyalty programs.

The "At-Risk" segment has a high average recency. Targeted re-engagement campaigns for this group could yield a high ROI.

LTV prediction allows the business to look beyond past revenue and identify customers with high future potential.

🏃‍♀️ How to Run
Setup Environment:

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate for Windows

# Install dependencies
pip install -r requirements.txt

Run Analysis Notebook:

Open and run the cells in notebooks/1_Model_Training_and_Analysis.ipynb. This will perform the analysis and generate the final_customer_data.csv and model files in the models/ directory.

Launch Dashboard:

streamlit run app/dashboard.py

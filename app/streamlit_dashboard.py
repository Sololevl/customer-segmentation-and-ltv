import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Customer Segmentation & LTV Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# LOAD DATA AND MODELS
# =============================================================================
@st.cache_data
def load_data():
    """Loads the final customer data."""
    df = pd.read_csv('final_customer_data.csv')
    return df

@st.cache_resource
def load_models():
    """Loads the saved clustering model and scaler."""
    kmeans_model = joblib.load('models/kmeans_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    return kmeans_model, scaler

df = load_data()
kmeans_model, scaler = load_models()

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Customer Segment Prediction"])

st.sidebar.title("About")
st.sidebar.info(
    "This dashboard provides insights into customer segmentation based on RFM analysis "
    "and predicts 12-month Customer Lifetime Value (LTV)."
)

# =============================================================================
# PAGE 1: DASHBOARD OVERVIEW
# =============================================================================
if page == "Dashboard Overview":
    st.title("📊 Customer Segmentation Dashboard")
    st.markdown("An overview of customer segments and their predicted value.")

    # --- Key Metrics ---
    st.header("Key Metrics")
    total_customers = df['customer_id'].nunique()
    total_revenue = df['monetary'].sum()
    avg_ltv = df['predicted_ltv'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Total Past Revenue", f"${total_revenue:,.2f}")
    col3.metric("Average Predicted 12-Month LTV", f"${avg_ltv:,.2f}")

    # --- LTV by Segment Chart ---
    st.header("Average Predicted LTV by Segment")
    ltv_by_segment = df.groupby('segment')['predicted_ltv'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=ltv_by_segment.index, y=ltv_by_segment.values, ax=ax, palette='viridis')
    ax.set_ylabel('Average Predicted LTV ($)')
    ax.set_xlabel('Customer Segment')
    ax.set_title('Average 12-Month LTV by Customer Segment', fontsize=14)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # --- Customer Data Explorer ---
    st.header("Customer Data Explorer")
    segment_filter = st.selectbox("Filter by Segment:", ["All"] + list(df['segment'].unique()))
    
    if segment_filter == "All":
        display_df = df
    else:
        display_df = df[df['segment'] == segment_filter]
    
    st.dataframe(display_df.sort_values(by='predicted_ltv', ascending=False).head(20))

# =============================================================================
# PAGE 2: CUSTOMER SEGMENT PREDICTION
# =============================================================================
elif page == "Customer Segment Prediction":
    st.title("🔮 Predict a Customer's Segment")
    st.markdown("Enter a customer's RFM values to predict which segment they belong to.")

    # --- User Input Form ---
    with st.form("prediction_form"):
        recency = st.number_input("Recency (days since last purchase)", min_value=0, value=50)
        frequency = st.number_input("Frequency (total number of purchases)", min_value=1, value=5)
        monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0, format="%.2f")
        
        submitted = st.form_submit_button("Predict Segment")

    if submitted:
        # --- Prediction Logic ---
        # Create a DataFrame from the user's input
        input_data = pd.DataFrame({
            'recency': [recency],
            'frequency': [frequency],
            'monetary': [monetary]
        })
        
        # Apply the same transformations as in the notebook
        input_log = np.log1p(input_data)
        input_scaled = scaler.transform(input_log)
        
        # Make the prediction
        prediction = kmeans_model.predict(input_scaled)
        
        # Map the prediction to the segment name
        # We need the segment map from the notebook. For simplicity, we hardcode it here.
        # A more advanced approach would save this map as well.
        segment_map_from_notebook = {
            0: 'Champions', 1: 'Potential Loyalists', 2: 'New Customers', 3: 'At-Risk'
        }
        # Note: This mapping might need adjustment if your notebook's cluster numbers change.
        # It's better to dynamically determine or save this map.
        predicted_segment = segment_map_from_notebook.get(prediction[0], "Unknown Segment")

        st.success(f"### This customer belongs to the **{predicted_segment}** segment.")


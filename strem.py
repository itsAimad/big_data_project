import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(page_title="Patient Monitoring System",
                   layout="wide",
                   page_icon="🏥")

# Application title and description
st.title("🏥 Real-time Patient Monitoring System")
st.markdown("""
This dashboard provides real-time monitoring of patient vital signs with risk prediction.
""")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Patient Monitor", "Analytics", "Predictions"])

# ========== PATIENT MONITOR TAB ==========
with tab1:
    # Patient selection
    st.sidebar.header("Patient Selection")
    patient_id = st.sidebar.selectbox("Select Patient ID",
                                      options=[f"PT{i:04d}" for i in range(1, 21)])

    # Layout in columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Vital Signs Monitor")

        # Create metrics display
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric(label="Heart Rate", value="78 bpm", delta="+2")
        with metric_cols[1]:
            st.metric(label="Oxygen Saturation", value="97%", delta="-0.5")
        with metric_cols[2]:
            st.metric(label="Temperature", value="36.8°C", delta="+0.2")
        with metric_cols[3]:
            st.metric(label="BP", value="124/82 mmHg", delta="-3/+1")

        # Create real-time charts
        fig = make_subplots(rows=3, cols=1,
                           subplot_titles=("Heart Rate & Respiratory Rate",
                                          "Blood Pressure",
                                          "Temperature & O2 Saturation"),
                           shared_xaxes=True,
                           vertical_spacing=0.1,
                           row_heights=[0.33, 0.33, 0.33])

        # Sample data for demonstration
        time_points = [datetime.now() - timedelta(minutes=i) for i in range(60, 0, -1)]
        hr_data = np.random.normal(78, 3, 60)
        rr_data = np.random.normal(16, 1, 60)
        sys_bp = np.random.normal(124, 4, 60)
        dia_bp = np.random.normal(82, 3, 60)
        temp_data = np.random.normal(36.8, 0.1, 60)
        o2_data = np.random.normal(97, 0.5, 60)

        # Add traces for each vital sign
        fig.add_trace(go.Scatter(x=time_points, y=hr_data, name="Heart Rate",
                               line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=time_points, y=rr_data, name="Respiratory Rate",
                               line=dict(color='blue')), row=1, col=1)

        fig.add_trace(go.Scatter(x=time_points, y=sys_bp, name="Systolic BP",
                               line=dict(color='darkred')), row=2, col=1)
        fig.add_trace(go.Scatter(x=time_points, y=dia_bp, name="Diastolic BP",
                               line=dict(color='lightblue')), row=2, col=1)

        fig.add_trace(go.Scatter(x=time_points, y=temp_data, name="Temperature",
                               line=dict(color='orange')), row=3, col=1)
        fig.add_trace(go.Scatter(x=time_points, y=o2_data, name="O2 Saturation",
                               line=dict(color='purple')), row=3, col=1)

        # Update layout
        fig.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Patient info card
        st.subheader("Patient Information")
        st.markdown(f"""
        **Patient ID:** {patient_id}
        **Name:** John Doe
        **Age:** 45
        **Gender:** Male
        **Height:** 175 cm
        **Weight:** 80.5 kg
        **BMI:** 26.3
        """)

        # Risk prediction card
        st.subheader("Risk Assessment")
        risk_score = 0.32

        # Show risk gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score", 'font': {'size': 24}},
            delta = {'reference': 0.5, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [0, 1], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "green"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "red"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.8}}))

        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Risk factors
        st.subheader("Contributing Factors")

        contributing_factors = {
            "Elevated Heart Rate": 40,
            "Blood Pressure": 30,
            "Age": 15,
            "BMI": 15
        }

        fig = go.Figure(go.Bar(
            x=list(contributing_factors.values()),
            y=list(contributing_factors.keys()),
            orientation='h',
            marker_color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        ))

        fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ========== ANALYTICS TAB ==========
with tab2:
    st.subheader("Patient Population Analytics")

    # Show filters
    filter_cols = st.columns(4)
    with filter_cols[0]:
        age_range = st.slider("Age Range", 18, 90, (30, 60))
    with filter_cols[1]:
        gender = st.multiselect("Gender", ["Male", "Female"], ["Male", "Female"])
    with filter_cols[2]:
        risk_status = st.multiselect("Risk Status", ["High Risk", "Low Risk"], ["High Risk", "Low Risk"])
    with filter_cols[3]:
        time_range = st.selectbox("Time Period", ["Last 24 Hours", "Last Week", "Last Month", "All Time"])

    # Create example charts
    chart_cols = st.columns(2)

    with chart_cols[0]:
        # Distribution by age and risk
        st.markdown("**Age Distribution by Risk Category**")

        # Sample data
        ages = np.random.normal(50, 15, 100).astype(int)
        ages = [max(18, min(a, 90)) for a in ages]  # Clamp to dataset range
        risk = np.random.choice(["High Risk", "Low Risk"], 100, p=[0.3, 0.7])

        df = pd.DataFrame({
            "Age": ages,
            "Risk Category": risk
        })

        fig = px.histogram(df, x="Age", color="Risk Category",
                          barmode="overlay",
                          color_discrete_map={"High Risk": "red", "Low Risk": "green"})
        st.plotly_chart(fig, use_container_width=True)

    with chart_cols[1]:
        # Vital signs correlation
        st.markdown("**Vital Signs Correlation Matrix**")

        # Create sample correlation matrix
        vital_signs = ["Heart Rate", "Respiratory Rate", "Systolic BP",
                       "Diastolic BP", "Temperature", "O2 Saturation"]

        # Sample correlation matrix
        corr_matrix = np.array([
            [1.0, 0.7, 0.5, 0.3, 0.4, -0.3],
            [0.7, 1.0, 0.4, 0.2, 0.5, -0.4],
            [0.5, 0.4, 1.0, 0.8, 0.1, -0.2],
            [0.3, 0.2, 0.8, 1.0, 0.0, -0.1],
            [0.4, 0.5, 0.1, 0.0, 1.0, -0.3],
            [-0.3, -0.4, -0.2, -0.1, -0.3, 1.0]
        ])

        fig = px.imshow(corr_matrix,
                      x=vital_signs,
                      y=vital_signs,
                      color_continuous_scale='RdBu_r',
                      color_continuous_midpoint=0)
        st.plotly_chart(fig, use_container_width=True)

    # Second row of charts
    chart_cols2 = st.columns(2)

    with chart_cols2[0]:
        st.markdown("**Risk Distribution by Gender and Age Group**")

        # Create sample data
        age_groups = ["18-30", "31-45", "46-60", "61-75", "76+"]
        males_high = [5, 12, 18, 25, 30]
        males_low = [25, 22, 15, 10, 5]
        females_high = [4, 10, 15, 22, 28]
        females_low = [26, 24, 18, 12, 7]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Males (High Risk)',
            x=age_groups,
            y=males_high,
            marker_color='darkred'
        ))

        fig.add_trace(go.Bar(
            name='Males (Low Risk)',
            x=age_groups,
            y=males_low,
            marker_color='lightgreen'
        ))

        fig.add_trace(go.Bar(
            name='Females (High Risk)',
            x=age_groups,
            y=females_high,
            marker_color='red'
        ))

        fig.add_trace(go.Bar(
            name='Females (Low Risk)',
            x=age_groups,
            y=females_low,
            marker_color='green'
        ))

        fig.update_layout(barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    with chart_cols2[1]:
        st.markdown("**Vital Signs Outside Normal Range**")

        # Sample data
        vitals = ["Heart Rate", "Respiratory Rate", "Temperature",
                 "O2 Saturation", "Systolic BP", "Diastolic BP"]

        pct_abnormal = [12, 8, 5, 15, 20, 18]

        colors = ['red' if x > 15 else 'orange' if x > 10 else 'yellow' for x in pct_abnormal]

        fig = go.Figure([go.Bar(x=vitals, y=pct_abnormal, marker_color=colors)])
        fig.update_layout(yaxis_title="% Outside Normal Range")

        st.plotly_chart(fig, use_container_width=True)

# ========== PREDICTIONS TAB ==========
with tab3:
    st.subheader("Predictive Analytics")

    # Patient selection for prediction
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Patient Selection")
        predict_patient = st.selectbox("Select Patient for Prediction",
                                      options=[f"PT{i:04d}" for i in range(1, 21)],
                                      key="predict_patient")

        st.markdown("### Timeline Selection")
        prediction_hours = st.slider("Prediction Window (hours)", 1, 24, 6)

        st.markdown("### Current Status")
        status_cols = st.columns(2)

        with status_cols[0]:
            st.metric(label="Current Risk Score", value="0.32", delta="")
        with status_cols[1]:
            st.metric(label="Predicted Risk Score", value="0.61", delta="+0.29")

        # Action recommendations
        st.markdown("### Recommended Actions")
        st.info("✅ Increase monitoring frequency")
        st.info("✅ Check electrolyte levels")
        st.info("✅ Review medication interactions")
        st.warning("⚠️ Potential deterioration in next 4-6 hours")

    with col2:
        # Risk trend prediction
        st.markdown("### Predicted Risk Trend")

        # Create sample prediction data
        hours = list(range(0, prediction_hours + 1))
        current_risk = 0.32

        # Create an increasing risk curve
        predicted_risk = [current_risk]
        for _ in range(prediction_hours):
            # Add a random increase to the risk, but cap it at 1.0
            increase = np.random.uniform(0.02, 0.1)  # Random increase between 0.02 and 0.1
            new_risk = min(predicted_risk[-1] + increase, 1.0)  # Ensure risk doesn't exceed 1.0
            predicted_risk.append(new_risk)

        # Create the plot
        fig = go.Figure(data=[
            go.Scatter(x=hours, y=predicted_risk, mode='lines+markers',
                       line=dict(color='red'), marker=dict(size=8))
        ])

        # Customize the plot layout
        fig.update_layout(
            xaxis_title="Hours from Now",
            yaxis_title="Predicted Risk Score",
            yaxis_range=[0, 1.05],  # Set y-axis range
            title="Risk Trend Prediction",
            title_x=0.5, # Center the title
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            paper_bgcolor='rgba(0,0,0,0)' # Transparent background
        )

        st.plotly_chart(fig, use_container_width=True)

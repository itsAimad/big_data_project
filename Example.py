import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from kafka import KafkaConsumer
import json
from datetime import datetime
import pickle
import requests
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Monitoring System",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF69B4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .tab-subheader {
        font-size: 1.8rem;
        color: #FF69B4;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 5px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        border-radius: 5px;
        padding: 1rem;
        text-align: center;
        background-color: #f8f9fa;
        border-left: 5px solid #FF69B4;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.8rem;
    }
    .prediction-positive {
        color: #FF0000;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .prediction-negative {
        color: #00CC66;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .sidebar {
        background-color: #FFF;
        padding: 1rem;
    }
    .interpretation {
        padding: 1rem;
        background-color: #f0f7ff;
        border-radius: 5px;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def load_sample_data():
    """Load sample breast cancer data for demo purposes"""
    # This is normally where you'd connect to your Kafka stream
    # For demo purposes, we'll use Wisconsin Breast Cancer Dataset format
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    
    column_names = ['id', 'diagnosis'] + [
        f"{feature}_{stat}" for feature in 
        ['radius', 'texture', 'perimeter', 'area', 'smoothness', 
         'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension'] 
        for stat in ['mean', 'se', 'worst']
    ]
    
    try:
        df = pd.read_csv(url, header=None, names=column_names)
        # Convert diagnosis to binary (M=1, B=0)
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        return df
    except:
        # Fallback: generate synthetic data matching your schema
        np.random.seed(42)
        n_samples = 100
        
        # Create synthetic data that roughly mimics breast cancer features
        data = {}
        for feature in ['radius', 'texture', 'perimeter', 'area', 'smoothness', 
                       'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']:
            # Mean values
            if feature == 'radius':
                data[f'{feature}_mean'] = np.random.normal(15, 3, n_samples)
            elif feature == 'texture':
                data[f'{feature}_mean'] = np.random.normal(20, 4, n_samples)
            elif feature == 'perimeter':
                data[f'{feature}_mean'] = np.random.normal(90, 20, n_samples)
            elif feature == 'area':
                data[f'{feature}_mean'] = np.random.normal(650, 150, n_samples)
            elif feature == 'smoothness':
                data[f'{feature}_mean'] = np.random.normal(0.1, 0.02, n_samples)
            else:
                data[f'{feature}_mean'] = np.random.normal(0.1, 0.05, n_samples)
            
            # SE values (smaller than mean)
            data[f'{feature}_se'] = data[f'{feature}_mean'] * np.random.normal(0.1, 0.02, n_samples)
            
            # Worst values (larger than mean)
            data[f'{feature}_worst'] = data[f'{feature}_mean'] * np.random.normal(1.2, 0.1, n_samples)
        
        # Create synthetic diagnoses (30% malignant, 70% benign)
        data['diagnosis'] = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        
        # For malignant cases, make the features more extreme
        malignant_indices = np.where(data['diagnosis'] == 1)[0]
        for feature in ['radius', 'perimeter', 'area', 'concavity', 'concave_points']:
            data[f'{feature}_mean'][malignant_indices] *= 1.5
            data[f'{feature}_worst'][malignant_indices] *= 1.7
        
        df = pd.DataFrame(data)
        df['id'] = np.arange(1, n_samples+1)
        
        return df

def mock_kafka_consumer():
    """Mock Kafka consumer for demo purposes"""
    df = load_sample_data()
    selected_rows = df.sample(1)
    
    # Create a message similar to what would come from Kafka
    message = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_id": f"P{np.random.randint(1000, 9999)}",
        "features": selected_rows.iloc[0].drop(['id', 'diagnosis']).to_dict(),
        "prediction": int(selected_rows['diagnosis'].values[0]),
        "prediction_probability": np.random.uniform(0.7, 0.99) if selected_rows['diagnosis'].values[0] == 1 else np.random.uniform(0.01, 0.3)
    }
    
    return message

def get_feature_importance():
    """Return mock feature importance for the model"""
    # This would normally come from your trained model
    features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 
               'concave_points_mean', 'radius_worst', 'perimeter_worst', 'area_worst', 'concave_points_worst']
    importance = [0.25, 0.08, 0.15, 0.18, 0.07, 0.06, 0.12, 0.04, 0.03, 0.02]
    return pd.DataFrame({'feature': features, 'importance': importance}).sort_values('importance', ascending=False)

def create_3d_scatter():
    """Create 3D scatter plot with the three most important features"""
    df = load_sample_data()
    
    fig = px.scatter_3d(
        df, 
        x='radius_mean', 
        y='concave_points_mean', 
        z='area_mean',
        color='diagnosis',
        color_continuous_scale=['#00CC66', '#FF0000'],
        opacity=0.7,
        title="3D Visualization of Key Features"
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Radius Mean',
            yaxis_title='Concave Points Mean',
            zaxis_title='Area Mean'
        ),
        coloraxis_colorbar=dict(
            title="Diagnosis",
            tickvals=[0, 1],
            ticktext=["Benign", "Malignant"],
        )
    )
    
    return fig

def create_correlation_heatmap():
    """Create correlation heatmap for mean features"""
    df = load_sample_data()
    mean_cols = [col for col in df.columns if 'mean' in col]
    
    corr = df[mean_cols].corr()
    
    fig = px.imshow(
        corr,
        title="Correlation Between Mean Features",
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
    )
    
    fig.update_layout(height=600)
    
    return fig

def feature_distribution_plot(feature):
    """Create distribution plot for a given feature by diagnosis"""
    df = load_sample_data()
    
    fig = px.histogram(
        df, x=feature, color='diagnosis',
        marginal='box',
        color_discrete_map={0: '#00CC66', 1: '#FF0000'},
        labels={'diagnosis': 'Diagnosis'},
        barmode='overlay',
        opacity=0.7,
        title=f"Distribution of {feature} by Diagnosis"
    )
    
    fig.update_layout(
        xaxis_title=feature,
        yaxis_title="Count",
        legend_title="Diagnosis",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            itemsizing='constant'
        )
    )
    
    # Update legend labels
    fig.data[0].name = "Benign"
    fig.data[1].name = "Malignant"
    
    return fig

def get_feature_explanation(feature):
    """Provide explanation for a feature"""
    explanations = {
        'radius': "Average distance from center to points on the perimeter",
        'texture': "Standard deviation of gray-scale values",
        'perimeter': "Size of the core tumor",
        'area': "Area of the core tumor",
        'smoothness': "Local variation in radius lengths",
        'compactness': "Perimeter² / area - 1.0",
        'concavity': "Severity of concave portions of the contour",
        'concave_points': "Number of concave portions of the contour",
        'symmetry': "Symmetry of the cell nuclei",
        'fractal_dimension': "Coastline approximation - 1"
    }
    
    base_feature = feature.split('_')[0]
    stat_type = feature.split('_')[1] if len(feature.split('_')) > 1 else ""
    
    stat_explanations = {
        'mean': "Average value for the feature across the cell nuclei",
        'se': "Standard error (standard deviation / sqrt(number of nuclei))",
        'worst': "Mean of the three largest values (indicating the most extreme cases)"
    }
    
    base_explanation = explanations.get(base_feature, "No explanation available")
    stat_explanation = stat_explanations.get(stat_type, "")
    
    return f"{base_explanation}. {stat_explanation}"

def simulate_real_time_predictions():
    """Simulate real-time predictions from Kafka stream"""
    # In a real implementation, you would establish a Kafka consumer here
    # For demo purposes, we'll simulate data
    message = mock_kafka_consumer()
    return message

# Main application
def main():
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/YourUsername/your-repo/main/logo.png", width=100)
        st.title("Breast Cancer Monitoring System")
        st.markdown("#### A real-time prediction system using machine learning")
        
        st.divider()
        
        if st.button("📊 Generate New Prediction", key="generate_new"):
            st.session_state.last_prediction_time = datetime.now().strftime("%H:%M:%S")
            st.session_state.last_prediction = simulate_real_time_predictions()
        
        if 'last_prediction_time' in st.session_state:
            st.success(f"Last prediction at: {st.session_state.last_prediction_time}")
        
        st.divider()
        
        # Show current active predictions count
        total_predictions = st.session_state.get('total_predictions', 0)
        malignant_count = st.session_state.get('malignant_count', 0)
        benign_count = st.session_state.get('benign_count', 0)
        
        col1, col2 = st.columns([3,1])
        with col1:
            st.metric("Malignant", malignant_count, delta=None)
        with col2:
            st.metric("Benign", benign_count, delta=None)
            
    # Initialize session state
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
        st.session_state.total_predictions = 0
        st.session_state.malignant_count = 0
        st.session_state.benign_count = 0
    
    # Update predictions if there's a new one
    if 'last_prediction' in st.session_state and (len(st.session_state.predictions) == 0 or 
                                                st.session_state.last_prediction['timestamp'] != st.session_state.predictions[-1]['timestamp']):
        st.session_state.predictions.append(st.session_state.last_prediction)
        st.session_state.total_predictions += 1
        
        if st.session_state.last_prediction['prediction'] == 1:
            st.session_state.malignant_count += 1
        else:
            st.session_state.benign_count += 1
        
        # Keep only the last 50 predictions
        if len(st.session_state.predictions) > 50:
            st.session_state.predictions.pop(0)
            
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Data Visualization", "Real-time Predictions", "Data Insights", "Test a Case"])

    # Tab 1: Overview
    with tab1:
        st.markdown("<h3 class='tab-subheader'>Breast Cancer Monitoring System Overview</h3>", unsafe_allow_html=True)
        
        # Display system architecture
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### System Architecture")
            st.image("https://miro.medium.com/max/1400/1*SZfUXYn3h3DH6csfRSqXhg.png", 
                     caption="Data pipeline architecture: Kafka → Spark Streaming → ML Model → Streamlit")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### About Breast Cancer Detection")
            st.write("""
            This system uses machine learning to analyze cellular features extracted from fine needle aspirates (FNA) 
            of breast masses. The system processes these features in real-time to predict whether a mass is benign or malignant.
            
            The features measured describe characteristics of the cell nuclei present in the digitized image of an FNA, 
            including radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, 
            and fractal dimension.
            """)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Key Metrics")
            
            # Display metrics
            st.metric("Model Accuracy", "94.2%", delta="1.2%")
            st.metric("Sensitivity", "96.7%", delta="0.5%")
            st.metric("Specificity", "91.8%", delta="-0.3%")
            st.metric("F1 Score", "0.932", delta="0.02")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Breast Cancer Types")
            
            # Create a pie chart for cancer types
            labels = ['Invasive Ductal Carcinoma', 'Ductal Carcinoma In Situ', 'Invasive Lobular Carcinoma', 'Others']
            values = [70, 15, 10, 5]
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Add educational video placeholder
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 3D Animation of Breast Cancer Development")
        cols = st.columns([1, 2, 1])
        with cols[1]:
            # In a real app, you'd embed a video here. For this example, we'll use a placeholder
            st.markdown("""
            <div style="position: relative; padding-bottom: 56.25%; height: 0;">
                <iframe src="https://www.youtube.com/embed/QmLVqqmMV4g" 
                        style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" 
                        frameborder="0" allowfullscreen>
                </iframe>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 2: Data Visualization
    with tab2:
        st.markdown("<h3 class='tab-subheader'>Data Visualization</h3>", unsafe_allow_html=True)
        
        # Feature distribution plots
        col1, col2 = st.columns([2,1])
        with col1:
            st.plotly_chart(feature_distribution_plot('radius_mean'), use_container_width=True)
        with col2:
            st.plotly_chart(feature_distribution_plot('texture_mean'), use_container_width=True)
        
        # Correlation heatmap
        st.plotly_chart(create_correlation_heatmap(), use_container_width=True)
        
        # 3D scatter plot
        st.plotly_chart(create_3d_scatter(), use_container_width=True)
        
    # Tab 3: Real-time Predictions
    with tab3:
        st.markdown("<h3 class='tab-subheader'>Real-time Predictions</h3>", unsafe_allow_html=True)
        
        if 'last_prediction' in st.session_state:
                st.session_state.predictions.append(st.session_state.last_prediction)
    st.session_state.total_predictions += 1
    
    if st.session_state.last_prediction['prediction'] == 1:
        st.session_state.malignant_count += 1
    else:
        st.session_state.benign_count += 1
    
    # Keep only the last 50 predictions
    if len(st.session_state.predictions) > 50:
        st.session_state.predictions.pop(0)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Data Visualization", "Real-time Predictions", "Data Insights", "Test a Case"])

# Tab 1: Overview
with tab1:
    st.markdown("<h3 class='tab-subheader'>Breast Cancer Monitoring System Overview</h3>", unsafe_allow_html=True)
    
    # Display system architecture
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### System Architecture")
        st.image("https://miro.medium.com/max/1400/1*SZfUXYn3h3DH6csfRSqXhg.png", 
                 caption="Data pipeline architecture: Kafka → Spark Streaming → ML Model → Streamlit")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### About Breast Cancer Detection")
        st.write("""
        This system uses machine learning to analyze cellular features extracted from fine needle aspirates (FNA) 
        of breast masses. The system processes these features in real-time to predict whether a mass is benign or malignant.
        
        The features measured describe characteristics of the cell nuclei present in the digitized image of an FNA, 
        including radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, 
        and fractal dimension.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Key Metrics")
        
        # Display metrics
        st.metric("Model Accuracy", "94.2%", delta="1.2%")
        st.metric("Sensitivity", "96.7%", delta="0.5%")
        st.metric("Specificity", "91.8%", delta="-0.3%")
        st.metric("F1 Score", "0.932", delta="0.02")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Breast Cancer Types")
        
        # Create a pie chart for cancer types
        labels = ['Invasive Ductal Carcinoma', 'Ductal Carcinoma In Situ', 'Invasive Lobular Carcinoma', 'Others']
        values =[70, 15, 10, 5]
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Add educational video placeholder
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 3D Animation of Breast Cancer Development")
    cols = st.columns([1,1])
    with cols[1]:
        # In a real app, you'd embed a video here. For this example, we'll use a placeholder
        st.markdown("""
        <div style="position: relative; padding-bottom: 56.25%; height: 0;">
            <iframe src="https://www.youtube.com/embed/QmLVqqmMV4g" 
                    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" 
                    frameborder="0" allowfullscreen>
            </iframe>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: Data Visualization
with tab2:
    st.markdown("<h3 class='tab-subheader'>Data Visualization</h3>", unsafe_allow_html=True)
    
    # Feature distribution plots
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(feature_distribution_plot('radius_mean'), use_container_width=True)
    with col2:
        st.plotly_chart(feature_distribution_plot('texture_mean'), use_container_width=True)
    
    # Correlation heatmap
    st.plotly_chart(create_correlation_heatmap(), use_container_width=True)
    
    # 3D scatter plot
    st.plotly_chart(create_3d_scatter(), use_container_width=True)
    
# Tab 3: Real-time Predictions
with tab3:
    st.markdown("<h3 class='tab-subheader'>Real-time Predictions</h3>", unsafe_allow_html=True)
    
    if 'last_prediction' in st.session_state:
        st.write(f"Patient ID: {st.session_state.last_prediction['patient_id']}")
        st.write(f"Timestamp: {st.session_state.last_prediction['timestamp']}")
        
        # Use styled text for prediction
        if st.session_state.last_prediction['prediction'] == 1:
            st.markdown(f"<p class='prediction-positive'>Prediction: Malignant (Confidence: {st.session_state.last_prediction['prediction_probability']:.2f})</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='prediction-negative'>Prediction: Benign (Confidence: {st.session_state.last_prediction['prediction_probability']:.2f})</p>", unsafe_allow_html=True)
    
    if len(st.session_state.predictions) > 0:
        st.markdown("<h4 class='tab-subheader'>Recent Predictions</h4>", unsafe_allow_html=True)
        
        recent_predictions_df = pd.DataFrame(st.session_state.predictions).sort_values(by='timestamp', ascending=False).head(10)
        st.dataframe(recent_predictions_df[['timestamp', 'patient_id', 'prediction', 'prediction_probability']])
    
# Tab 4: Data Insights
with tab4:
    st.markdown("<h3 class='tab-subheader'>Data Insights</h3>", unsafe_allow_html=True)
    
    # Feature Importance Analysis
    feature_importance_df = get_feature_importance()
    
    fig = px.bar(feature_importance_df, x='feature', y='importance',
                 title='Feature Importance')
    st.plotly_chart(fig, use_container_width=True)
    
    selected_feature = st.selectbox("Select a feature to view its explanation", feature_importance_df['feature'].unique())
    explanation = get_feature_explanation(selected_feature)
    
    st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
    st.write(f"#### Explanation for {selected_feature}:")
    st.write(explanation)
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 5: Test a Case
with tab5:
    st.markdown("<h3 class='tab-subheader'>Test a Case</h3>", unsafe_allow_html=True)
    
    feature_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                    'smoothness_mean', 'compactness_mean', 'concavity_mean',
                    'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
                    'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
                    'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
                    'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
    
    # Create input form
    with st.form(key='test_case_form'):
        input_values = {}
        for name in feature_names:
            input_values[name] = st.number_input(label=name)
        
        submit_button = st.form_submit_button(label='Predict Diagnosis')
    
    if submit_button:
        # Mock model prediction based on inputs
        prediction = 1 if sum(input_values.values()) > 10 else 0  # Example: Replace with your model
        confidence = np.random.uniform(0.7, 0.99) if prediction == 1 else np.random.uniform(0.01, 0.3)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if prediction == 1:
            st.markdown(f"<p class='prediction-positive'>Prediction: Malignant (Confidence: {confidence:.2f})</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='prediction-negative'>Prediction: Benign (Confidence: {confidence:.2f})</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<p class='footer'>© 2025 Breast Cancer Monitoring System</p>", unsafe_allow_html=True)


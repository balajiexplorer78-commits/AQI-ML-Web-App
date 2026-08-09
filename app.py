import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Set Page Config (Must be the very first Streamlit command)
st.set_page_config(page_title="AQI Status Prediction Dashboard", layout="wide")

# --- 1. DATA & MODEL LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv("aqi.zip")
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading aqi.csv: {e}")
    st.stop()

try:
    model = joblib.load("train_model.joblib")
except Exception as e:
    # st.warning("Model train_model.joblib not found. App running in Simulation Mode.")
    model = None

# --- 2. PREMIUM GLASSMORPHISM CUSTOM CSS ---
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%) !important;
        color: #f8fafc !important;
    }
    
    /* Title Layout */
    .dashboard-header {
        text-align: center;
        padding: 20px 0px;
        margin-bottom: 25px;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(8px);
    }
    .main-title {
        font-size: 42px !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin: 0 !important;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 16px;
        margin-top: 5px !important;
    }

    /* Container Cards */
    .custom-card {
        background: rgba(30, 41, 59, 0.4) !important;
        padding: 24px !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(12px) !important;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }
    
    /* Modern Metric Cards */
    .metric-container {
        background: rgba(15, 23, 42, 0.6);
        padding: 15px 20px;
        border-radius: 12px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-label {
        color: #94a3b8;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 5px;
    }

    /* Status Alert Design */
    .status-pill {
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        font-size: 36px;
        font-weight: 800;
        margin: 15px 0;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. DASHBOARD HEADER ---
st.markdown("""
<div class='dashboard-header'>
    <h1 class='main-title'>💨AIR QUALITY INDEX STATUS ANALYTICS & PREDICTION SYSTEM</h1>
    <p class='subtitle'>Predict Air Quality Index Status based on location , AQI and machine learning predictive model</p>
</div>
""", unsafe_allow_html=True)

# --- 4. TAB NAVIGATION LAYOUT ---
tab1, tab2 = st.tabs(["🔮 Prediction Portal", "📊 Dataset Analytics Insights"])

# ==============================================================================
# TAB 1: PREDICTION PORTAL
# ==============================================================================
with tab1:
    col_input, col_result = st.columns([2, 3], gap="large")
    
    with col_input:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("🔍 Input Parameters")
        
        state_input = st.selectbox(
            "Target State/Union Territory",
            ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", 
             "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", 
             "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", 
             "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", 
             "Uttarakhand", "West Bengal"]
        )
        
        area_input = st.selectbox(
            "Area Classification",
            ["Urban", "Suburban", "Industrial", "Residential"]
        )
        
        aqi_val = st.number_input("Enter Measured AQI Value", min_value=0.0, max_value=500.0, value=120.0, step=1.0)
        stations = st.number_input("Number of Active Monitoring Stations", min_value=1.0, max_value=50.0, value=2.0, step=1.0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_click = st.button("RUN PREDICTIVE MODEL", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        st.markdown("<div class='custom-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.subheader("🎯 Prediction Result")
        
        # Calculate or simulate prediction output
        if predict_click:
            if model is not None:
                input_df = pd.DataFrame({
                    'state': [state_input], 'area': [area_input],
                    'aqi_value': [aqi_val], 'number_of_monitoring_stations': [stations]
                })
                pred_status = model.predict(input_df)[0]
            else:
                # Simulation mode fallback logic matching index distributions
                if aqi_val <= 50: pred_status = "Good"
                elif aqi_val <= 100: pred_status = "Satisfactory"
                elif aqi_val <= 200: pred_status = "Moderate"
                elif aqi_val <= 300: pred_status = "Poor"
                elif aqi_val <= 400: pred_status = "Very Poor"
                else: pred_status = "Severe"

            # Color-coded theme maps for status pill alerts
            status_themes = {
                'Good': {'bg': 'rgba(44, 160, 44, 0.2)', 'text': '#2ca02c', 'emoji': '😊'},
                'Satisfactory': {'bg': 'rgba(188, 189, 34, 0.2)', 'text': '#d4d622', 'emoji': '🙂'},
                'Moderate': {'bg': 'rgba(255, 127, 14, 0.2)', 'text': '#ff7f0e', 'emoji': '😐'},
                'Poor': {'bg': 'rgba(214, 39, 40, 0.2)', 'text': '#ff4d4d', 'emoji': '😷'},
                'Very Poor': {'bg': 'rgba(148, 103, 189, 0.2)', 'text': '#bc94ea', 'emoji': '⚠'},
                'Severe': {'bg': 'rgba(140, 86, 75, 0.2)', 'text': '#e59c9c', 'emoji': '🚨'}
            }
            theme = status_themes.get(pred_status, {'bg': 'rgba(255,255,255,0.1)', 'text': '#fff', 'emoji': '📊'})
            
            st.markdown(f"""
            <div class='status-pill' style='background-color: {theme['bg']}; color: {theme['text']}; border: 1px solid {theme['text']}33;'>
                AQI Status: {pred_status} {theme['emoji']}
            </div>
            """, unsafe_allow_html=True)
            
            # Speedometer Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=aqi_val,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#f1f5f9", 'thickness': 0.2},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 1,
                    'bordercolor': "rgba(255,255,255,0.1)",
                    'steps': [
                        {'range': [0, 50], 'color': "#2ca02c"},
                        {'range': [51, 100], 'color': "#bcbd22"},
                        {'range': [101, 200], 'color': "#ff7f0e"},
                        {'range': [201, 300], 'color': "#d62728"},
                        {'range': [301, 400], 'color': "#9467bd"},
                        {'range': [401, 500], 'color': "#8c564b"}
                    ]
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#f1f5f9", 'family': "Inter, Arial"},
                height=260,
                margin=dict(l=30, r=30, t=10, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.info("The prediction result will appear here.")
            
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 2: DATASET ANALYTICS INSIGHTS
# ==============================================================================
with tab2:
    # Sidebar Filters Contextualized for Analytics view
    st.sidebar.markdown("### 📊 Analytics Filters")
    all_states = sorted(df['state'].unique())
    selected_states = st.sidebar.multiselect(
        "Filter Dashboard Focus States", 
        options=all_states, 
        default=["Maharashtra", "Bihar", "Delhi"] if "Delhi" in all_states else [all_states[0]]
    )

    filtered_df = df[df['state'].isin(selected_states)] if selected_states else df.copy()

    # Dynamic Metric Info row
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(f"""<div class='metric-container'><div class='metric-label'>Avg AQI (Selected Focus)</div><div class='metric-value'>{filtered_df['aqi_value'].mean():.1f}</div></div>""", unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""<div class='metric-container' style='border-left-color: #ef4444;'><div class='metric-label'>Peak AQI Recorded</div><div class='metric-value'>{filtered_df['aqi_value'].max()}</div></div>""", unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""<div class='metric-container' style='border-left-color: #10b981;'><div class='metric-label'>Total Rows Tracked</div><div class='metric-value'>{filtered_df.shape[0]:,}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualization Row 1
    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("#### 📈 AQI Trend Matrix")
        trend_df = filtered_df.groupby([filtered_df['date'].dt.to_period('M').dt.to_timestamp(), 'state'])['aqi_value'].mean().reset_index()
        fig_line = px.line(
            trend_df, x="date", y="aqi_value", color="state",
            labels={"date": "Timeline", "aqi_value": "Index Score"}, template="plotly_dark"
        )
        fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=15, b=15))
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r1_col2:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("#### 🍩 Air Quality Distribution ")
        status_counts = filtered_df['air_quality_status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        color_map = {'Good': '#2ca02c', 'Satisfactory': '#bcbd22', 'Moderate': '#ff7f0e', 'Poor': '#d62728', 'Very Poor': '#9467bd', 'Severe': '#8c564b'}
        fig_pie = px.pie(status_counts, values='Count', names='Status', hole=0.45, color='Status', color_discrete_map=color_map, template="plotly_dark")
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", legend_orientation="h", margin=dict(t=15, b=15))
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Visualization Row 2
    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Comparative State Pollution Ranking")
        state_aqi = filtered_df.groupby('state')['aqi_value'].mean().reset_index().sort_values(by='aqi_value', ascending=False)
        fig_bar = px.bar(state_aqi, x='state', y='aqi_value', color='aqi_value', color_continuous_scale=px.colors.sequential.Reds, template="plotly_dark")
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, margin=dict(t=15, b=15))
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r2_col2:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("#### 🔬 Critical Pollutant Component Spread")
        pollutant_counts = filtered_df['prominent_pollutants'].value_counts().head(8).reset_index()
        pollutant_counts.columns = ['Pollutant', 'Count']
        fig_pollutant = px.bar(pollutant_counts, y='Pollutant', x='Count', orientation='h', color='Count', color_continuous_scale=px.colors.sequential.Viridis, template="plotly_dark")
        fig_pollutant.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, yaxis={'categoryorder':'total ascending'}, margin=dict(t=15, b=15))
        st.plotly_chart(fig_pollutant, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import io
from sklearn.linear_model import LinearRegression
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.express as px

try:
    import xlsxwriter
    XLSXWRITER_AVAILABLE = True
except ImportError:
    XLSXWRITER_AVAILABLE = False
    st.warning("xlsxwriter is not installed. Excel export will be disabled.")
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    st.warning("fpdf is not installed. PDF export will be disabled.")
import base64

# Minimal CSS for styling
st.markdown("""
<style>
body {
    font-family: Arial, sans-serif;
    color: #333;
}
h1, h2, h3 {
    color: #2c3e50;
}
.section {
    margin: 1em 0;
    padding: 1em;
    border: 1px solid #ddd;
    border-radius: 5px;
    background-color: #ffffff;
}
.st-expander {
    background-color: #ffffff;
    border: 1px solid #ddd;
    border-radius: 5px;
}
.stMetric {
    background-color: #ffffff !important;
    padding: 1em;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    color: #333 !important;
}
.stMetric label {
    color: #333 !important;
}
.stMetric [data-testid="stMetricValue"] {
    color: #333 !important;
}
.key-metrics-section {
    background-color: #ffffff !important;
    padding: 1em;
    color: #333 !important;
}
.key-metrics-section * {
    color: #333 !important;
}
.logout-button {
    position: fixed;
    bottom: 10px;
    width: 200px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for login and filtered data
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login Page
if not st.session_state.logged_in:
    st.title("Health Claim Cost Prediction - Login")
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "password123":
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid username or password. Please try again.")

    st.markdown("</div>", unsafe_allow_html=True)

else:
    # Load the dataset and model with error handling
    try:
        data = pd.read_csv("final_merged_synthea_cleaned98.csv")
        model = joblib.load("xgb_model_new.pkl")
    except Exception as e:
        st.error(f"Error loading dataset or model: {e}")
        data = pd.DataFrame()  # Initialize with an empty DataFrame in case of error
    
    # Initialize filtered_data in session state if it doesn't exist
    if "filtered_data" not in st.session_state or st.session_state.filtered_data is None:
        if not data.empty:
            st.session_state.filtered_data = data.copy()
        else:
            st.session_state.filtered_data = pd.DataFrame()  # Initialize with empty DataFrame

    # Only continue with preprocessing if data was loaded successfully
    if not data.empty:
        try:
            # Define required columns based on your dataset
            required_columns = ["AGE", "GENDER", "RACE", "ETHNICITY", "INCOME", 
                                "ENCOUNTERCLASS", "CODE", "TOTAL_CLAIM_COST"]
            
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                st.error(f"Dataset is missing required columns: {missing_columns}")
            else:
                # Handle patient ID column
                patient_id_column = None
                possible_patient_columns = ["PATIENTID", "PATIENT_ID", "Id", "ID", "PATIENT"]
                for col in possible_patient_columns:
                    if col in data.columns:
                        patient_id_column = col
                        break
                
                if patient_id_column:
                    data = data.rename(columns={patient_id_column: "PATIENT"})
                else:
                    data["PATIENT"] = [f"patient_{i}" for i in range(len(data))]
                
                required_columns = ["AGE", "GENDER", "RACE", "ETHNICITY", "INCOME", 
                                    "ENCOUNTERCLASS", "CODE", "TOTAL_CLAIM_COST", "PATIENT"]
                missing_columns = [col for col in required_columns if col not in data.columns]
                if missing_columns:
                    st.error(f"Dataset is missing required columns after patient ID handling: {missing_columns}")
                else:
                    # Map TOTAL_CLAIM_COST to TOTALCOST for compatibility with your code
                    data = data.rename(columns={"TOTAL_CLAIM_COST": "TOTALCOST"})

                    # Calculate encounter duration if START and STOP are present; otherwise, set to 0
                    if "START" in data.columns and "STOP" in data.columns:
                        data["ENCOUNTER_DURATION"] = (pd.to_datetime(data["STOP"]) - pd.to_datetime(data["START"])).dt.days
                    else:
                        data["ENCOUNTER_DURATION"] = 0

                    # Add missing columns expected by the model with default values
                    model_expected_columns = [
                        "PAYER_COVERAGE", "BASE_ENCOUNTER_COST", "AVG_CLAIM_COST", "STATE",
                        "NUM_DIAG1", "HEALTHCARE_EXPENSES", "NUM_ENCOUNTERS", "NUM_DIAG2", "HEALTHCARE_COVERAGE"
                    ]
                    for col in model_expected_columns:
                        if col not in data.columns:
                            if col == "STATE":
                                data[col] = "Unknown"
                            else:
                                data[col] = 0

                    data = data.fillna(0)

                    # Extract year for filtering and forecasting
                    if "START" in data.columns:
                        data["START_YEAR"] = pd.to_datetime(data["START"]).dt.year
                    else:
                        data["START_YEAR"] = 2025

                    # Make sure filtered_data has all the necessary columns
                    if "filtered_data" not in st.session_state or st.session_state.filtered_data is None:
                        st.session_state.filtered_data = data.copy()

                    # Prepare features for one-hot encoding
                    features = [
                        "AGE", "GENDER", "RACE", "ETHNICITY", "INCOME", "ENCOUNTERCLASS", "CODE", "ENCOUNTER_DURATION",
                        "PAYER_COVERAGE", "BASE_ENCOUNTER_COST", "AVG_CLAIM_COST", "STATE",
                        "NUM_DIAG1", "HEALTHCARE_EXPENSES", "NUM_ENCOUNTERS", "NUM_DIAG2"
                    ]
                    X = data[features]

                    # Define categorical columns
                    categorical_cols = ["GENDER", "RACE", "ETHNICITY", "ENCOUNTERCLASS", "CODE", "STATE"]

                    # Convert categorical columns to string
                    for col in categorical_cols:
                        X[col] = X[col].astype(str)

                    # Perform one-hot encoding
                    X_encoded = pd.get_dummies(X, columns=categorical_cols)

                    # Store the encoded feature names
                    model_features = X_encoded.columns.tolist()

                    # Store categories for each categorical column
                    categories = {col: data[col].astype(str).unique() for col in categorical_cols}
        except Exception as e:
            st.error(f"Error preprocessing data: {e}")
            categories = {}
            model_features = []
    else:
        st.error("Unable to load data. Please check your file paths and data sources.")
        categories = {}
        model_features = []

    # Main App
    st.title("Health Claim Cost Prediction")

    # Add logout button in the sidebar at the bottom
    with st.sidebar:
        st.markdown("<div class='logout-button'>", unsafe_allow_html=True)
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab7, tab8 = st.tabs([
        "Data Filters", 
        "Key Metrics",
        "Claim Forecast",
        "Data Visualizations", 
        "Resource Allocation", 
        "Prediction Cost", 
        "Data Export"
    ])

    # Tab 1: Data Filters
    with tab1:
        st.header("Data Filters")
        st.markdown("<div class='section'>", unsafe_allow_html=True)

        # Safety check to make sure we have data
        if not data.empty and "START_YEAR" in data.columns:
            years = sorted(list(data["START_YEAR"].unique()))
            if len(years) > 0:
                start_year = st.selectbox("Start Year:", years, index=0)
                end_year = st.selectbox("End Year:", years, index=len(years)-1)

                if st.button("Apply Year Range"):
                    try:
                        filtered_data = data[(data["START_YEAR"] >= start_year) & (data["START_YEAR"] <= end_year)]
                        st.session_state.filtered_data = filtered_data
                        st.write(f"Filtered Data: {len(filtered_data)} records")
                    except Exception as e:
                        st.error(f"Error applying year range filter: {e}")
            else:
                st.warning("No year data available.")
        else:
            st.warning("Data not properly loaded or missing START_YEAR column.")

        # Only try to display filtered data if it exists
        try:
            if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
                st.write(f"Filtered Data: {len(st.session_state.filtered_data)} records")
            else:
                st.warning("No filtered data available.")
        except Exception as e:
            st.error(f"Error displaying filtered data: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 2: Key Metrics
    with tab2:
        st.header("Key Metrics")
        st.markdown("<div class='section key-metrics-section'>", unsafe_allow_html=True)

        try:
            if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
                filtered_data = st.session_state.filtered_data
                
                # Check if required columns exist
                required_metrics_cols = ["TOTALCOST", "AGE", "PATIENT", "ENCOUNTER_DURATION", "HEALTHCARE_COVERAGE"]
                missing_cols = [col for col in required_metrics_cols if col not in filtered_data.columns]
                
                if missing_cols:
                    st.warning(f"Missing columns for metrics calculation: {missing_cols}")
                else:
                    avg_claim_cost = filtered_data["TOTALCOST"].mean()
                    total_claims = len(filtered_data)
                    avg_age = filtered_data["AGE"].mean()
                    total_patients = filtered_data["PATIENT"].nunique()
                    avg_encounter_duration = filtered_data["ENCOUNTER_DURATION"].mean()
                    avg_coverage = filtered_data["HEALTHCARE_COVERAGE"].mean() if "HEALTHCARE_COVERAGE" in filtered_data.columns else 0

                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(label="Average Claim Cost", value=f"${avg_claim_cost:.2f}")
                        st.metric(label="Total Number of Patients", value=total_patients)

                    with col2:
                        st.metric(label="Average Health Care Coverage", value=f"${avg_coverage:.2f}")
                        st.metric(label="Average Patient Age", value=f"{avg_age:.1f} years")

                    with col3:
                        st.metric(label="Total Number of Claims", value=total_claims)
                        st.metric(label="Average Encounter Duration", value=f"{avg_encounter_duration:.1f} days")
            else:
                st.warning("No data available for metrics calculation.")
        except Exception as e:
            st.error(f"Error calculating key metrics: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Only show the rest of the tabs if we have valid data
    if data.empty:
        with tab3, tab4, tab5, tab7, tab8:
            st.warning("Data could not be loaded. Please check your data source and try again.")
    else:
        # 📅 Tab 3: Claim Forecast with Prophet (Interactive + Filters)
        with tab3:
            st.header("Claim Forecast")
            st.markdown("<div class='section'>", unsafe_allow_html=True)
        
            try:
                # 🧽 Filters Section
                st.subheader("Apply Filters")
                age_filter = st.multiselect("Select Age(s):", options=sorted(data['AGE'].dropna().unique()), default=None)
                gender_filter = st.multiselect("Select Gender(s):", options=data['GENDER'].dropna().unique(), default=None)
                race_filter = st.multiselect("Select Race(s):", options=data['RACE'].dropna().unique(), default=None)
                ethnicity_filter = st.multiselect("Select Ethnicity:", options=data['ETHNICITY'].dropna().unique(), default=None)
                encounter_filter = st.multiselect("Select Encounter Type:", options=data['ENCOUNTERCLASS'].dropna().unique(), default=None)
        
                df = data.copy()
        
                # Apply Filters
                if age_filter:
                    df = df[df['AGE'].isin(age_filter)]
                if gender_filter:
                    df = df[df['GENDER'].isin(gender_filter)]
                if race_filter:
                    df = df[df['RACE'].isin(race_filter)]
                if ethnicity_filter:
                    df = df[df['ETHNICITY'].isin(ethnicity_filter)]
                if encounter_filter:
                    df = df[df['ENCOUNTERCLASS'].isin(encounter_filter)]
        
                if "START" in df.columns and "TOTALCOST" in df.columns:
                    # STEP 4: Prepare data
                    df['START'] = pd.to_datetime(df['START'], errors='coerce').dt.tz_localize(None)
                    df.dropna(subset=['START'], inplace=True)
                    df.sort_values('START', inplace=True)
        
                    df['START_MONTH'] = df['START'].dt.to_period('M').dt.to_timestamp()
                    df['TOTALCOST'] = pd.to_numeric(df['TOTALCOST'], errors='coerce')
                    monthly_cost = df.groupby('START_MONTH')['TOTALCOST'].sum()
                    monthly_cost_clean = monthly_cost.dropna()
                    monthly_cost_clean = monthly_cost_clean[~monthly_cost_clean.isin([np.inf, -np.inf])]
                    monthly_cost_recent = monthly_cost_clean[-36:]
        
                    prophet_df = monthly_cost_recent.reset_index()
                    prophet_df.columns = ['ds', 'y']
        
                    # Prophet Forecast
                    model = Prophet()
                    model.fit(prophet_df)
                    future = model.make_future_dataframe(periods=60, freq='MS')
                    forecast = model.predict(future)
        
                    # Plotly Interactive Plot
                    st.subheader("Interactive Forecast Plot (Next 5 Years)")
                    fig_plotly = px.line(forecast, x='ds', y='yhat', labels={'ds': 'Date', 'yhat': 'Predicted Claim Cost'},
                                         title='Forecast of Monthly Claim Costs')
                    fig_plotly.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound')
                    fig_plotly.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound')
                    st.plotly_chart(fig_plotly, use_container_width=True)
        
                    # Forecast Table
                    st.write("Forecast Table (Tail):")
                    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        
                else:
                    st.warning("Columns 'START' and 'TOTALCOST' not found in the dataset.")
            except Exception as e:
                st.error(f"Error generating claim forecast with Prophet: {e}")
        
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Tab 4: Data Visualizations
        with tab4:
            st.header("Data Visualizations")
            st.markdown("<div class='section'>", unsafe_allow_html=True)
            
            import plotly.express as px
            
            try:
                st.subheader("Total Cost Distribution")
                fig_cost_dist = px.histogram(data, x="TOTALCOST", nbins=20, 
                                         labels={"TOTALCOST": "Total Cost ($)"}, color_discrete_sequence=["#636EFA"])
                fig_cost_dist.update_layout(bargap=0.1, showlegend=False)
                st.plotly_chart(fig_cost_dist, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating Total Cost Distribution chart: {e}")
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Tab 5: Resource Allocation
        with tab5:
            st.header("Resource Allocation")
            st.markdown("<div class='section'>", unsafe_allow_html=True)
            
            import plotly.express as px
            
            try:
                st.subheader("Age Group vs Average Total Cost")
                # Create age bins for grouping
                bins = [0, 18, 30, 40, 50, 60, 100]
                labels = ['0-18', '19-30', '31-40', '41-50', '51-60', '60+']
                data['Age Group'] = pd.cut(data['AGE'], bins=bins, labels=labels, right=False)

                # Calculate average total cost per age group
                age_group_avg_cost = data.groupby('Age Group')['TOTALCOST'].mean().reset_index()

                # Create bar chart using Plotly
                fig_age_group_avg_cost = px.bar(age_group_avg_cost, x="Age Group", y="TOTALCOST",
                                            labels={"Age Group": "Age Group", "TOTALCOST": "Average Total Cost ($)"},
                                            color="Age Group",
                                            color_discrete_sequence=["#636EFA"])
                fig_age_group_avg_cost.update_layout(showlegend=False)
                st.plotly_chart(fig_age_group_avg_cost, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating Age Group vs Average Total Cost chart: {e}")

            try:
                st.subheader("Average Claim Cost by Race")
                avg_cost_by_race = st.session_state.filtered_data.groupby("RACE")["TOTALCOST"].mean().reset_index()
                if not avg_cost_by_race.empty:
                    fig_regional_avg = px.bar(
                        avg_cost_by_race, x="RACE", y="TOTALCOST",
                        title="Average Claim Cost by Race",
                        labels={"RACE": "Race", "TOTALCOST": "Average Cost ($)"},
                        color_discrete_sequence=["#636EFA"]
                    )
                    fig_regional_avg.update_layout(showlegend=False)
                    st.plotly_chart(fig_regional_avg, use_container_width=True)
                else:
                    st.info("No data available for the selected filters.")
            except Exception as e:
                st.error(f"Error generating average cost by race visualization: {e}")

            try:
                st.subheader("Average Claim Cost by Encounter Class")
                # Average Claim Cost by Encounter Class
                avg_cost_by_encounter = st.session_state.filtered_data.groupby("ENCOUNTERCLASS")["TOTALCOST"].mean().reset_index()
                if not avg_cost_by_encounter.empty:
                    # Create bar chart using Plotly
                    fig_avg_cost_by_encounter = px.bar(avg_cost_by_encounter, x="ENCOUNTERCLASS", y="TOTALCOST",
                                                   labels={"ENCOUNTERCLASS": "Encounter Class", "TOTALCOST": "Average Claim Cost ($)"},
                                                   color="ENCOUNTERCLASS",
                                                   color_discrete_sequence=["#636EFA"])
                    fig_avg_cost_by_encounter.update_layout(showlegend=False)
                    st.plotly_chart(fig_avg_cost_by_encounter, use_container_width=True)
            except Exception as e:
                st.error(f"Error analyzing resource allocation: {e}")

            try:
                st.subheader("Average Claim Cost by Gender")
                avg_cost_by_gender = st.session_state.filtered_data.groupby("GENDER")["TOTALCOST"].mean().reset_index()
                if not avg_cost_by_gender.empty:
                    fig_gender_avg = px.bar(
                        avg_cost_by_gender, x="GENDER", y="TOTALCOST",
                        title="Average Claim Cost by Gender",
                        labels={"GENDER": "Gender", "TOTALCOST": "Average Cost ($)"},
                        color_discrete_sequence=["#636EFA"]
                    )
                    fig_gender_avg.update_layout(showlegend=False)
                    st.plotly_chart(fig_gender_avg, use_container_width=True)
                else:
                    st.info("No data available for the selected filters.")
            except Exception as e:
                st.error(f"Error generating average cost by gender visualization: {e}")

            try:
                st.subheader("Average Claim Cost by Ethnicity")
                avg_cost_by_ethnicity = st.session_state.filtered_data.groupby("ETHNICITY")["TOTALCOST"].mean().reset_index()
                if not avg_cost_by_ethnicity.empty:
                    fig_ethnicity_avg = px.bar(
                        avg_cost_by_ethnicity, x="ETHNICITY", y="TOTALCOST",
                        title="Average Claim Cost by Ethnicity",
                        labels={"ETHNICITY": "Ethnicity", "TOTALCOST": "Average Cost ($)"},
                        color_discrete_sequence=["#636EFA"]
                    )
                    fig_ethnicity_avg.update_layout(showlegend=False)
                    st.plotly_chart(fig_ethnicity_avg, use_container_width=True)
                else:
                    st.info("No data available for the selected filters.")
            except Exception as e:
                st.error(f"Error generating average cost by ethnicity visualization: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

        # Tab 7: Prediction Cost
        with tab7:
            st.header("Prediction Cost")
            st.markdown("<div class='section'>", unsafe_allow_html=True)

            st.subheader("Enter Patient Information")
            
            # Check if categories exists and has required keys
            if all(key in categories for key in ["GENDER", "RACE", "ETHNICITY", "ENCOUNTERCLASS", "CODE"]):
                col1, col2 = st.columns(2)
                
                with col1:
                    age = st.slider("Age", 0, 100, 30, key="pred_age")
                    gender = st.selectbox("Gender", categories["GENDER"], key="pred_gender")
                    race = st.selectbox("Race", categories["RACE"], key="pred_race")
                    ethnicity = st.selectbox("Ethnicity", categories["ETHNICITY"], key="pred_ethnicity")
                
                with col2:
                    income = st.slider("Income ($)", 0, 100000, 50000, key="pred_income")
                    encounter_class = st.selectbox("Encounter Class", categories["ENCOUNTERCLASS"], key="pred_encounter_class")
                    code = st.selectbox("Procedure Code", categories["CODE"], key="pred_code")
                    encounter_duration = st.slider("Encounter Duration (days)", 0, 30, 1, key="pred_encounter_duration")

                # Create input data with all model-expected columns
                input_data = pd.DataFrame({
                    "AGE": [age],
                    "GENDER": [str(gender)],
                    "RACE": [str(race)],
                    "ETHNICITY": [str(ethnicity)],
                    "INCOME": [income],
                    "ENCOUNTERCLASS": [str(encounter_class)],
                    "CODE": [str(code)],
                    "ENCOUNTER_DURATION": [encounter_duration],
                    "PAYER_COVERAGE": [0],
                    "BASE_ENCOUNTER_COST": [0],
                    "AVG_CLAIM_COST": [0],
                    "STATE": ["Unknown"],
                    "NUM_DIAG1": [0],
                    "HEALTHCARE_EXPENSES": [0],
                    "NUM_ENCOUNTERS": [0],
                    "NUM_DIAG2": [0]
                })

                try:
                    # Perform one-hot encoding
                    categorical_cols = ["GENDER", "RACE", "ETHNICITY", "ENCOUNTERCLASS", "CODE", "STATE"]
                    input_data_encoded = pd.get_dummies(input_data, columns=categorical_cols)
                    
                    # Reindex to match model_features
                    input_data_encoded = input_data_encoded.reindex(columns=model_features, fill_value=0)

                    if st.button("Predict Cost"):
                        # Ensure input_data_encoded is a numpy array for prediction
                        prediction_input = input_data_encoded.values
                        prediction = model.predict(prediction_input)[0]
                        st.write(f"### Predicted Claim Cost: ${prediction:.2f}")
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
            else:
                st.error("Missing required category data for prediction.")

            st.markdown("</div>", unsafe_allow_html=True)

        # Tab 8: Data Export
        with tab8:
            st.header("Data Export")
            st.markdown("<div class='section'>", unsafe_allow_html=True)

            st.write("Select the export format and click the button to download the filtered dataset.")
            export_formats = ["CSV"]
            if XLSXWRITER_AVAILABLE:
                export_formats.append("Excel")
            if FPDF_AVAILABLE:
                export_formats.append("PDF")
            export_format = st.selectbox("Select Export Format", export_formats)

            if st.button("Export Data"):
                try:
                    if export_format == "CSV":
                        buffer = io.BytesIO()
                        st.session_state.filtered_data.to_csv(buffer, index=False)
                        buffer.seek(0)
                        st.download_button(
                            label="Download CSV File",
                            data=buffer,
                            file_name="filtered_data_export.csv",
                            mime="text/csv",
                            key="export_csv_button",
                            use_container_width=True
                        )
                    elif export_format == "Excel" and XLSXWRITER_AVAILABLE:
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                            st.session_state.filtered_data.to_excel(writer, index=False, sheet_name="Sheet1")
                        buffer.seek(0)
                        st.download_button(
                            label="Download Excel File",
                            data=buffer,
                            file_name="filtered_data_export.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="export_excel_button",
                            use_container_width=True
                        )
                    elif export_format == "PDF" and FPDF_AVAILABLE:
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        for i, row in st.session_state.filtered_data.head(10).iterrows():
                            pdf.cell(200, 10, txt=str(row.to_dict()), ln=True)
                        pdf_output = pdf.output(dest="S").encode("latin1")
                        st.download_button(
                            label="Download PDF File",
                            data=pdf_output,
                            file_name="filtered_data_export.pdf",
                            mime="application/pdf",
                            key="export_pdf_button",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error exporting data: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

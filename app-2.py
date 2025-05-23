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

import streamlit as st
import pandas as pd
import joblib

# Initialize session state for login and filtered data
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "filtered_data" not in st.session_state:
    st.session_state.filtered_data = pd.DataFrame()

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

        if not data.empty:
            st.session_state.filtered_data = data.copy()

    except Exception as e:
        st.error(f"Error loading dataset or model: {e}")
        data = pd.DataFrame()  # Safe fallback

    # You can now safely use `data` and `st.session_state.filtered_data` below
    if data.empty:
        st.warning("Dataset is empty or failed to load.")
    else:
        st.success("Dataset loaded successfully!")
        # Proceed with app logic here using `data` or `st.session_state.filtered_data`

    
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
                    # Convert AGE into age groups
                    def categorize_age(age):
                        if age < 18:
                            return "0-17"
                        elif 18 <= age <= 24:
                            return "18-24"
                        elif 25 <= age <= 34:
                            return "25-34"
                        elif 35 <= age <= 44:
                            return "35-44"
                        elif 45 <= age <= 54:
                            return "45-54"
                        elif 55 <= age <= 64:
                            return "55-64"
                        else:
                            return "65+"
                    
                    data["AGE_GROUP"] = data["AGE"].apply(categorize_age)


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
                       "AGE_GROUP", "GENDER", "RACE", "ETHNICITY", "INCOME", "ENCOUNTERCLASS", "CODE", "ENCOUNTER_DURATION",
                        "PAYER_COVERAGE", "BASE_ENCOUNTER_COST", "AVG_CLAIM_COST", "STATE",
                        "NUM_DIAG1", "HEALTHCARE_EXPENSES", "NUM_ENCOUNTERS", "NUM_DIAG2"
                    ]
                    X = data[features]

                    # Define categorical columns
                    categorical_cols = ["AGE_GROUP","GENDER", "RACE", "ETHNICITY", "ENCOUNTERCLASS", "CODE", "STATE"]

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

   # Sidebar Filters
st.sidebar.header("Filter Data")

if not data.empty and "START_YEAR" in data.columns:
    years = sorted(list(data["START_YEAR"].unique()))
    start_year = st.sidebar.selectbox("Start Year", years, index=0)
    end_year = st.sidebar.selectbox("End Year", years, index=len(years)-1)

    # Helper to auto-select all if checkbox is checked
    def multiselect_with_select_all(label, options):
        select_all = st.sidebar.checkbox(f"Select All {label}", value=True, key=label)
        selected = st.sidebar.multiselect(label, options, default=options if select_all else [])
        return selected

    # Multi-select filter options
    age_group_options = sorted(data["AGE_GROUP"].dropna().unique())
    gender_options = sorted(data["GENDER"].dropna().unique())
    race_options = sorted(data["RACE"].dropna().unique())
    ethnicity_options = sorted(data["ETHNICITY"].dropna().unique())
    encounter_class_options = sorted(data["ENCOUNTERCLASS"].dropna().unique())

    selected_age_groups = multiselect_with_select_all("Age Group", age_group_options)
    selected_genders = multiselect_with_select_all("Gender", gender_options)
    selected_races = multiselect_with_select_all("Race", race_options)
    selected_ethnicities = multiselect_with_select_all("Ethnicity", ethnicity_options)
    selected_encounters = multiselect_with_select_all("Encounter Class", encounter_class_options)

    # Only filter if at least one option is selected per field
    if all([selected_age_groups, selected_genders, selected_races, selected_ethnicities, selected_encounters]):
        filtered_data = data[
            (data["START_YEAR"] >= start_year) &
            (data["START_YEAR"] <= end_year) &
            (data["AGE_GROUP"].isin(selected_age_groups)) &
            (data["GENDER"].isin(selected_genders)) &
            (data["RACE"].isin(selected_races)) &
            (data["ETHNICITY"].isin(selected_ethnicities)) &
            (data["ENCOUNTERCLASS"].isin(selected_encounters))
        ]
    else:
        filtered_data = pd.DataFrame()  # Empty result if any filter is not selected

    # Save to session state
    st.session_state.filtered_data = filtered_data

    # Optional: Show warning if nothing matches
    if filtered_data.empty:
        st.warning("⚠️ No data matches the selected filters. Try adjusting them.")
else:
    st.sidebar.warning("Data not loaded or missing required columns.")

     # Create tabs
tab2, tab3, tab4, tab5, tab7, tab8 = st.tabs([
        
        "Key Metrics",
        "Claim Forecast",
        "Data Visualizations", 
        "Resource Allocation", 
        "Prediction Cost", 
        "Data Export"
    ])


    # Tab 2: Key Metrics
with tab2:
    st.header("Key Metrics")
    st.markdown("<div class='section key-metrics-section'>", unsafe_allow_html=True)

    try:
        if "filtered_data" in st.session_state and not st.session_state.filtered_data.empty:
            filtered_data = st.session_state.filtered_data

            # Check for required columns before computing metrics
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
                avg_coverage = filtered_data["HEALTHCARE_COVERAGE"].mean()

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(label="Average Claim Cost", value=f"${avg_claim_cost:,.2f}")
                    st.metric(label="Total Patients", value=f"{total_patients:,}")

                with col2:
                    st.metric(label="Average Coverage", value=f"${avg_coverage:,.2f}")
                    st.metric(label="Average Age", value=f"{avg_age:.1f} years")

                with col3:
                    st.metric(label="Total Claims", value=f"{total_claims:,}")
                    st.metric(label="Avg Encounter Duration", value=f"{avg_encounter_duration:.1f} days")
        else:
            st.warning("No filtered data available. Please adjust filters in the sidebar.")
    except Exception as e:
        st.error(f"Error calculating key metrics: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


  
# 📅 Tab 3: Claim Forecast with Prophet (Interactive + Regressors)
with tab3:
    st.header("Claim Forecast")
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    if "filtered_data" in st.session_state and not st.session_state.filtered_data.empty:
        df = st.session_state.filtered_data.copy()

        try:
            import warnings
            warnings.filterwarnings("ignore")
            from prophet import Prophet

            # Clean and prepare the data
            df["START"] = pd.to_datetime(df["START"], errors="coerce").dt.tz_localize(None)
            df.dropna(subset=["START"], inplace=True)
            df = df[df["TOTALCOST"] >= 0]
            df["TOTALCOST"] = pd.to_numeric(df["TOTALCOST"], errors="coerce")
            df = df.sort_values("START")

            df["START_MONTH"] = df["START"].dt.to_period("M").dt.to_timestamp()
            regressors = ['AGE', 'NUM_STATUS1', 'NUM_DIAG1', 'NUM_DIAG2', 
                          'INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']
            cat_vars = ['RACE', 'ETHNICITY', 'GENDER','ENCOUNTERCLASS']

            # Handle numeric and categorical
            df[regressors] = df[regressors].apply(pd.to_numeric, errors='coerce')
            df_encoded = pd.get_dummies(df[cat_vars], drop_first=True)
            df_final = pd.concat([df[['START_MONTH', 'TOTALCOST']], df[regressors], df_encoded], axis=1)

            grouped = df_final.groupby('START_MONTH').agg({
                'TOTALCOST': 'sum',
                **{col: 'mean' for col in regressors + list(df_encoded.columns)}
            }).reset_index()

            grouped.rename(columns={'START_MONTH': 'ds', 'TOTALCOST': 'y'}, inplace=True)
            grouped['ds'] = pd.to_datetime(grouped['ds'], errors='coerce')
            grouped = grouped[grouped['ds'].dt.year >= 2000].drop_duplicates(subset='ds').sort_values('ds')

            grouped['lag_1'] = grouped['y'].shift(1)
            grouped['rolling_mean_3'] = grouped['y'].shift(1).rolling(window=3).mean()
            grouped = grouped.dropna().reset_index(drop=True)

            # Fit Prophet with regressors
            model = Prophet(interval_width=0.95, mcmc_samples=300)
            for col in grouped.columns:
                if col not in ['ds', 'y']:
                    model.add_regressor(col)
            model.fit(grouped)

            # Forecast until 2030
            last_date = grouped['ds'].max()
            future_end = pd.to_datetime('2030-12-01')
            months_to_forecast = (future_end.year - last_date.year) * 12 + (future_end.month - last_date.month)
            future = model.make_future_dataframe(periods=months_to_forecast, freq='MS')

            for col in grouped.columns:
                if col not in ['ds', 'y']:
                    future[col] = grouped[col].iloc[-1]  # placeholder: last known value

            forecast = model.predict(future)

            # Plot interactive forecast
            st.subheader("Forecast of Monthly Claim Costs (till 2030)")
            fig = px.line(forecast, x='ds', y='yhat',
                          labels={'ds': 'Date', 'yhat': 'Predicted Total Claim Cost'},
                          title='Projected Claim Costs using Prophet')
            fig.add_scatter(x=grouped['ds'], y=grouped['y'], mode='markers', name='Actual')
            st.plotly_chart(fig, use_container_width=True)

           
        except Exception as e:
            st.error(f"Error generating advanced forecast: {e}")

    else:
        st.warning("No filtered data available. Please adjust filters in the sidebar.")

    st.markdown("</div>", unsafe_allow_html=True)

     # Tab 4: Data Visualizations
with tab4:
    st.header("Data Visualizations")
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    import plotly.express as px

    # ✅ Use filtered data if available
    if "filtered_data" in st.session_state and not st.session_state.filtered_data.empty:
        df = st.session_state.filtered_data.copy()

        try:
            # ✅ Check if TOTALCOST column is present
            if "TOTALCOST" in df.columns:
                st.subheader("Total Cost Distribution")
                fig_cost_dist = px.histogram(
                    df, x="TOTALCOST", nbins=20,
                    labels={"TOTALCOST": "Total Cost ($)"},
                    color_discrete_sequence=["#636EFA"]
                )
                fig_cost_dist.update_layout(bargap=0.1, showlegend=False)
                st.plotly_chart(fig_cost_dist, use_container_width=True)
            else:
                st.warning("Missing column: 'TOTALCOST'. Please check your dataset.")
        except Exception as e:
            st.error(f"Error creating Total Cost Distribution chart: {e}")
    else:
        st.warning("No filtered data available. Please apply filters in the sidebar.")

    st.markdown("</div>", unsafe_allow_html=True)

            
            # Tab 5: Resource Allocation
with tab5:
    st.header("Resource Allocation")
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    import plotly.express as px
    if "filtered_data" in st.session_state and not st.session_state.filtered_data.empty:
        df = st.session_state.filtered_data.copy()

        try:
            st.subheader("Age Group vs Average Total Cost")
            if 'AGE_GROUP' in df.columns:
                age_group_avg_cost = df.groupby('AGE_GROUP')['TOTALCOST'].mean().reset_index()
                fig_age_group_avg_cost = px.bar(
                    age_group_avg_cost, x="AGE_GROUP", y="TOTALCOST",
                    labels={"AGE_GROUP": "Age Group", "TOTALCOST": "Average Total Cost ($)"},
                    color="AGE_GROUP",
                    color_discrete_sequence=["#636EFA"]
                )
                fig_age_group_avg_cost.update_layout(showlegend=False)
                st.plotly_chart(fig_age_group_avg_cost, use_container_width=True)
            else:
                st.warning("Age Group column is missing in the data.")
        except Exception as e:
            st.error(f"Error creating Age Group vs Average Total Cost chart: {e}")

        try:
            st.subheader("Average Claim Cost by Race")
            avg_cost_by_race = df.groupby("RACE")["TOTALCOST"].mean().reset_index()
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
            avg_cost_by_encounter = df.groupby("ENCOUNTERCLASS")["TOTALCOST"].mean().reset_index()
            if not avg_cost_by_encounter.empty:
                fig_avg_cost_by_encounter = px.bar(
                    avg_cost_by_encounter, x="ENCOUNTERCLASS", y="TOTALCOST",
                    labels={"ENCOUNTERCLASS": "Encounter Class", "TOTALCOST": "Average Claim Cost ($)"},
                    color="ENCOUNTERCLASS",
                    color_discrete_sequence=["#636EFA"]
                )
                fig_avg_cost_by_encounter.update_layout(showlegend=False)
                st.plotly_chart(fig_avg_cost_by_encounter, use_container_width=True)
        except Exception as e:
            st.error(f"Error analyzing resource allocation: {e}")

        try:
            st.subheader("Average Claim Cost by Gender")
            avg_cost_by_gender = df.groupby("GENDER")["TOTALCOST"].mean().reset_index()
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
            avg_cost_by_ethnicity = df.groupby("ETHNICITY")["TOTALCOST"].mean().reset_index()
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

    else:
        st.warning("Filtered data is not available.")

    st.markdown("</div>", unsafe_allow_html=True)

     
# Tab 7: Prediction Cost
with tab7:
    st.header("Prediction Cost")
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    st.subheader("Enter Patient Information")

    # Safely define categories dictionary from the data DataFrame
    try:
        categories = {
            "GENDER": sorted(data["GENDER"].dropna().unique()),
            "RACE": sorted(data["RACE"].dropna().unique()),
            "ETHNICITY": sorted(data["ETHNICITY"].dropna().unique()),
            "ENCOUNTERCLASS": sorted(data["ENCOUNTERCLASS"].dropna().unique()),
            "CODE": sorted(data["CODE"].dropna().unique()),
        }
    except Exception as e:
        categories = {}
        st.error(f"Error loading category values: {e}")

    # Check if all required keys are present in the categories dictionary
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
            # Perform one-hot encoding for categorical features
            categorical_cols = ["GENDER", "RACE", "ETHNICITY", "ENCOUNTERCLASS", "CODE", "STATE"]
            input_data_encoded = pd.get_dummies(input_data, columns=categorical_cols)

            # Reindex to match the model's expected feature set
            input_data_encoded = input_data_encoded.reindex(columns=model_features, fill_value=0)

            if st.button("Predict Cost"):
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

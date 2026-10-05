import streamlit as st
import joblib
import pandas as pd


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load("employee_attrition_model.pkl")
threshold = joblib.load("attrition_threshold.pkl")


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📊 Employee Attrition Predictor")

st.write(
    "Use employee information to estimate the probability "
    "of employee attrition using a trained machine learning model."
)

st.info(
    "This application uses a Logistic Regression model trained "
    "on the IBM HR Analytics Employee Attrition dataset."
)


# --------------------------------------------------
# EMPLOYEE INFORMATION
# --------------------------------------------------

st.header("👤 Employee Information")


# --------------------------------------------------
# BASIC INFORMATION
# --------------------------------------------------

st.subheader("Basic Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=30
    )

with col2:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col3:
    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )


# --------------------------------------------------
# JOB INFORMATION
# --------------------------------------------------

st.subheader("💼 Job Information")

col1, col2, col3 = st.columns(3)

with col1:
    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

with col2:
    department = st.selectbox(
        "Department",
        [
            "Sales",
            "Research & Development",
            "Human Resources"
        ]
    )

with col3:
    job_level = st.selectbox(
        "Job Level",
        [1, 2, 3, 4, 5],
        index=1
    )


# --------------------------------------------------
# WORK CONDITIONS
# --------------------------------------------------

st.subheader("🏢 Work Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    overtime = st.selectbox(
        "Overtime",
        ["No", "Yes"]
    )

with col2:
    business_travel = st.selectbox(
        "Business Travel",
        [
            "Non-Travel",
            "Travel_Rarely",
            "Travel_Frequently"
        ]
    )

with col3:
    distance_from_home = st.number_input(
        "Distance From Home",
        min_value=1,
        max_value=30,
        value=5
    )


# --------------------------------------------------
# SATISFACTION & WORK-LIFE
# --------------------------------------------------

st.subheader("😊 Satisfaction & Work-Life")

col1, col2, col3 = st.columns(3)

with col1:
    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        [1, 2, 3, 4],
        index=1
    )

with col2:
    environment_satisfaction = st.selectbox(
        "Environment Satisfaction",
        [1, 2, 3, 4],
        index=2
    )

with col3:
    work_life_balance = st.selectbox(
        "Work-Life Balance",
        [1, 2, 3, 4],
        index=1
    )


# --------------------------------------------------
# CAREER INFORMATION
# --------------------------------------------------

st.subheader("📈 Career Information")

col1, col2, col3 = st.columns(3)

with col1:
    total_working_years = st.number_input(
        "Total Working Years",
        min_value=0,
        max_value=40,
        value=8
    )

with col2:
    years_at_company = st.number_input(
        "Years At Company",
        min_value=0,
        max_value=40,
        value=3
    )

with col3:
    years_in_current_role = st.number_input(
        "Years In Current Role",
        min_value=0,
        max_value=20,
        value=2
    )


col1, col2, col3 = st.columns(3)

with col1:
    years_since_last_promotion = st.number_input(
        "Years Since Last Promotion",
        min_value=0,
        max_value=15,
        value=1
    )

with col2:
    years_with_curr_manager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        max_value=20,
        value=2
    )

with col3:
    num_companies_worked = st.number_input(
        "Number of Companies Worked",
        min_value=0,
        max_value=10,
        value=2
    )


# --------------------------------------------------
# COMPENSATION & PERFORMANCE
# --------------------------------------------------

st.subheader("💰 Compensation & Performance")

col1, col2, col3 = st.columns(3)

with col1:
    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=20000,
        value=4000
    )

with col2:
    monthly_rate = st.number_input(
        "Monthly Rate",
        min_value=2000,
        max_value=27000,
        value=15000
    )

with col3:
    daily_rate = st.number_input(
        "Daily Rate",
        min_value=100,
        max_value=1500,
        value=1000
    )


col1, col2, col3 = st.columns(3)

with col1:
    hourly_rate = st.number_input(
        "Hourly Rate",
        min_value=30,
        max_value=100,
        value=70
    )

with col2:
    percent_salary_hike = st.number_input(
        "Percent Salary Hike",
        min_value=10,
        max_value=30,
        value=12
    )

with col3:
    performance_rating = st.selectbox(
        "Performance Rating",
        [1, 2, 3, 4],
        index=2
    )


# --------------------------------------------------
# ADDITIONAL INFORMATION
# --------------------------------------------------

st.subheader("📚 Additional Information")

col1, col2, col3 = st.columns(3)

with col1:
    education = st.selectbox(
        "Education",
        [1, 2, 3, 4, 5],
        index=2
    )

with col2:
    education_field = st.selectbox(
        "Education Field",
        [
            "Life Sciences",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Human Resources",
            "Other"
        ]
    )

with col3:
    training_times_last_year = st.number_input(
        "Training Times Last Year",
        min_value=0,
        max_value=10,
        value=3
    )


col1, col2, col3 = st.columns(3)

with col1:
    job_involvement = st.selectbox(
        "Job Involvement",
        [1, 2, 3, 4],
        index=2
    )

with col2:
    relationship_satisfaction = st.selectbox(
        "Relationship Satisfaction",
        [1, 2, 3, 4],
        index=2
    )

with col3:
    stock_option_level = st.selectbox(
        "Stock Option Level",
        [0, 1, 2, 3],
        index=0
    )


# --------------------------------------------------
# CREATE EMPLOYEE DATAFRAME
# --------------------------------------------------

employee_data = pd.DataFrame([{
    "Age": age,
    "DailyRate": daily_rate,
    "DistanceFromHome": distance_from_home,
    "Education": education,
    "EnvironmentSatisfaction": environment_satisfaction,
    "HourlyRate": hourly_rate,
    "JobInvolvement": job_involvement,
    "JobLevel": job_level,
    "JobSatisfaction": job_satisfaction,
    "MonthlyIncome": monthly_income,
    "MonthlyRate": monthly_rate,
    "NumCompaniesWorked": num_companies_worked,
    "PercentSalaryHike": percent_salary_hike,
    "PerformanceRating": performance_rating,
    "RelationshipSatisfaction": relationship_satisfaction,
    "StockOptionLevel": stock_option_level,
    "TotalWorkingYears": total_working_years,
    "TrainingTimesLastYear": training_times_last_year,
    "WorkLifeBalance": work_life_balance,
    "YearsAtCompany": years_at_company,
    "YearsInCurrentRole": years_in_current_role,
    "YearsSinceLastPromotion": years_since_last_promotion,
    "YearsWithCurrManager": years_with_curr_manager,
    "BusinessTravel": business_travel,
    "Department": department,
    "EducationField": education_field,
    "Gender": gender,
    "JobRole": job_role,
    "MaritalStatus": marital_status,
    "OverTime": overtime
}])


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.divider()

st.header("🔮 Attrition Prediction")

st.write(
    "Click the button below to evaluate the employee profile."
)


if st.button(
    "Predict Attrition",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------
    # PREDICT PROBABILITY
    # --------------------------------------------------

    probabilities = model.predict_proba(employee_data)

    yes_index = list(model.classes_).index("Yes")

    attrition_probability = probabilities[0, yes_index]

    prediction = (
        "Yes"
        if attrition_probability >= threshold
        else "No"
    )


    # --------------------------------------------------
    # DISPLAY PROBABILITY
    # --------------------------------------------------

    st.metric(
        label="Attrition Probability",
        value=f"{attrition_probability * 100:.2f}%"
    )

    st.progress(
        min(attrition_probability, 1.0)
    )

    st.write(
        f"Model decision threshold: "
        f"{threshold * 100:.0f}%"
    )


    # --------------------------------------------------
    # DISPLAY PREDICTION
    # --------------------------------------------------

    if prediction == "Yes":

        st.error(
            "⚠️ High Attrition Risk — "
            "Employee is predicted to leave."
        )

    else:

        st.success(
            "✅ Low Attrition Risk — "
            "Employee is predicted to stay."
        )


    # --------------------------------------------------
    # PROBABILITY EXPLANATION
    # --------------------------------------------------

    st.info(
        "The probability represents the model's estimated "
        "probability of the employee belonging to the "
        "'Attrition = Yes' class. The final prediction is "
        "determined using the selected decision threshold."
    )


    # --------------------------------------------------
    # PERSONALIZED MODEL EXPLANATION
    # --------------------------------------------------

    st.subheader(
        "🔍 Why did the model make this prediction?"
    )

    st.write(
        "The factors below show which employee characteristics "
        "contributed most strongly to the model's prediction. "
        "Positive contributions push the model toward higher "
        "attrition risk, while negative contributions push it "
        "toward lower attrition risk."
    )


    # Get preprocessing component
    preprocessor = model.named_steps["preprocessor"]

    # Get Logistic Regression model
    logistic_model = model.named_steps["model"]


    # --------------------------------------------------
    # TRANSFORM EMPLOYEE DATA
    # --------------------------------------------------

    transformed_employee = (
        preprocessor.transform(employee_data)
    )


    # Convert sparse matrix to dense array if necessary
    if hasattr(transformed_employee, "toarray"):

        transformed_employee = (
            transformed_employee.toarray()
        )


    # --------------------------------------------------
    # GET FEATURE NAMES AND COEFFICIENTS
    # --------------------------------------------------

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    coefficients = (
        logistic_model.coef_[0]
    )


    # --------------------------------------------------
    # CALCULATE FEATURE CONTRIBUTIONS
    # --------------------------------------------------

    contributions = (
        transformed_employee[0] * coefficients
    )


    # --------------------------------------------------
    # CREATE CONTRIBUTION DATAFRAME
    # --------------------------------------------------

    contribution_data = pd.DataFrame({

        "Feature": feature_names,

        "Contribution": contributions

    })


    # Remove preprocessing prefixes

    contribution_data["Feature"] = (
        contribution_data["Feature"]
        .str.replace(
            "num__",
            "",
            regex=False
        )
        .str.replace(
            "cat__",
            "",
            regex=False
        )
    )


    # --------------------------------------------------
    # POSITIVE CONTRIBUTIONS
    # --------------------------------------------------

    positive_contributions = (
        contribution_data[
            contribution_data["Contribution"] > 0
        ]
        .sort_values(
            "Contribution",
            ascending=False
        )
        .head(5)
    )


    # --------------------------------------------------
    # NEGATIVE CONTRIBUTIONS
    # --------------------------------------------------

    negative_contributions = (
        contribution_data[
            contribution_data["Contribution"] < 0
        ]
        .sort_values(
            "Contribution",
            ascending=True
        )
        .head(5)
    )


    # --------------------------------------------------
    # DISPLAY CONTRIBUTIONS
    # --------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            "### 🔴 Factors increasing risk"
        )

        if len(positive_contributions) > 0:

            st.dataframe(
                positive_contributions[
                    ["Feature", "Contribution"]
                ],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.write(
                "No strong risk-increasing factors."
            )


    with col2:

        st.markdown(
            "### 🟢 Factors decreasing risk"
        )

        if len(negative_contributions) > 0:

            st.dataframe(
                negative_contributions[
                    ["Feature", "Contribution"]
                ],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.write(
                "No strong risk-decreasing factors."
            )


    # --------------------------------------------------
    # INTERPRETATION NOTE
    # --------------------------------------------------

    st.caption(
        "These contributions describe how the Logistic "
        "Regression model used the employee's characteristics "
        "to arrive at its prediction. They represent model "
        "associations, not causal relationships."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "This prediction is generated by a machine learning "
    "model and should be treated as a decision-support "
    "tool rather than a definitive prediction of employee "
    "behavior."
)





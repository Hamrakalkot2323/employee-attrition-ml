Employee Attrition Prediction Using Machine Learning

This is an end-to-end ML project that predicts employee attrition using data of employee demographic, job, satisfaction, career, and compensation information.

The project focuses on handling class imbalance, evaluating models using appropriate metrics, selecting a decision threshold using out-of-fold predictions, and providing interpretable predictions through an interactive Streamlit application.

----------------------------------------------

Project Objective

As employee attrition can have significant consequences for organizations, including recruitment costs, productivity loss, and disruption to teams the goal of this project is to develop a ML system that can identify employees who may have a higher risk of attrition, also rather than relying only on accuracy, this project focuses on identifying employees who leave by emphasizing metrics such as:-

- Recall
- F1-score
- F2-score
- Precision-Recall AUC (PR-AUC)
- ROC-AUC

This project also includes model interpretability so that the predictions can be examined in terms of the employee characteristics that influenced them.

------------------------------------------------------------

Dataset

The project uses the IBM HR Analytics Employee Attrition dataset.

The dataset contains:

- 1,470 employees
- 35 original columns

- Demographic information
- Job characteristics
- Satisfaction measures
- Compensation information
- Career history
- Attrition status

The target variable is: Attrition, with two classes: Yes/ No

The dataset is imbalanced, with substantially more employees who stayed than employees who left.



Exploratory Data Analysis

Several exploratory analyses were performed to understand patterns associated with employee attrition.
The analysis included:

Overall attrition distribution
Attrition by overtime
Attrition by job satisfaction
Attrition by age
Attrition by job role
Monthly income vs. attrition
Correlation analysis
Feature relationships with attrition
Combined overtime and job satisfaction analysis

The EDA helped identify potentially important patterns before model development.

--------------------------------------------------------------

Data Preprocessing

Four columns containing constant or unnecessary information were removed:

EmployeeCount
Over18
StandardHours
EmployeeNumber

The cleaned dataset contained:

1,470 rows
31 columns

The data was divided using an 80/20 stratified train-test split.

------------------------------------------------------

Feature preprocessing

Numerical features were standardized using: StandardScaler

Categorical features were encoded using: OneHotEncoder

with: handle_unknown="ignore"

All preprocessing was incorporated into a Scikit-learn Pipeline to ensure that the same transformations were applied consistently during training and prediction.

-----------------------------------------------------------------------------

Models Evaluated

Three machine learning models were evaluated:

Logistic Regression
Random Forest
XGBoost

Because the dataset is imbalanced, model comparison focused primarily on PR-AUC rather than accuracy alone.

Cross-validation

Models were evaluated using:

RepeatedStratifiedKFold

with:

5 folds
3 repeats
Random state = 42

Model comparison

| Model               | Mean PR-AUC |
| ------------------- | ----------: |
| Logistic Regression |      0.6457 |
| Random Forest       |      0.5457 |
| XGBoost             |      0.5402 |

Logistic Regression achieved the strongest mean PR-AUC and was therefore selected for further tuning.

------------------------------------------------------------------

Hyperparameter Tuning

The regularization parameter C of Logistic Regression was tuned using cross-validation.

Values tested:

0.01
0.1
1
10
100

The best value was: C = 0.1
with a mean cross-validation PR-AUC of approximately: 0.6472

--------------------------------------------------------------------------------

Model Evaluation

The final Logistic Regression model was evaluated on an untouched test set.

At the default threshold of 0.50:

| Metric    |  Score |
| --------- | -----: |
| PR-AUC    | 0.6038 |
| ROC-AUC   | 0.8114 |
| Precision |  78.9% |
| Recall    |  31.9% |
| F1-score  |  45.5% |

Although precision was relatively high, the model missed many employees who actually experienced attrition. Since identifying potential leavers was an important goal, a lower decision threshold was investigated.

-------------------------------------------------------------------------------------------

Decision Threshold Optimization

Instead of selecting the threshold directly from the test set, out-of-fold predictions were generated from the training data. This allowed threshold selection without using the final test set for tuning.
The project prioritized F2-score, which places greater emphasis on recall than precision.
The selected threshold was: 0.20

The final test set was evaluated only after the threshold had been selected.
Final test results at threshold = 0.20

| Metric    |     Score |
| --------- | --------: |
| Precision |   42.5%   |
| Recall    |   66.0%   |
| F1-score  |   51.7%   |
| F2-score  |   59.4%   |

Confusion matrix:

                Predicted
                No    Yes

Actual No      205    42
Actual Yes      16    31

The selected threshold increased recall from 31.9% to 66.0%, allowing the model to identify substantially more employees who actually left.
This came with a reduction in precision, representing the expected trade-off between catching more potential leavers and generating more false positives.

---------------------------------------------------------------------------------------

Model Interpretability

Logistic Regression coefficients were analyzed to understand which features were associated with higher or lower model scores.
Some of the strongest positive associations included:

Overtime = Yes
Frequent business travel
Laboratory Technician job role
Number of companies worked
Years since last promotion
Single marital status
Distance from home

Some stronger negative associations included:

Environment satisfaction
Age
Job satisfaction
Job involvement
Total working years
Years with current manager

These relationships represent model associations rather than causal relationships.

-------------------------------------------------------------------------------------

Interactive Streamlit Application

The trained model was integrated into an interactive Streamlit application.

The application allows users to:

Enter employee information
Generate an attrition probability
Apply the selected decision threshold
View the predicted attrition class
View factors contributing to the individual prediction

The application also provides personalized model contributions to help explain why a particular prediction was produced.

---------------------------------------------------------------------------

Model Behavior Experiments

Several controlled experiments were performed using the Streamlit application.

For example, a low-risk employee profile produced:

Attrition probability: 0.17%

A deliberately high-risk profile produced:

Attrition probability: 97.74%

Keeping the high-risk profile fixed and changing only overtime from Yes to No changed the prediction probability from:

97.74% → 91.29%

Changing only job satisfaction from 1 to 4 resulted in:

97.74% → 94.49%

These experiments demonstrated how the model combines multiple employee characteristics when generating predictions.

----------------------------------------------------------------

Technologies Used:-

Python
Pandas
NumPy
Scikit-learn
XGBoost
Matplotlib
Seaborn
Joblib
Streamlit
Git / GitHub

-------------------------------------------------------------


Project Structure

employee-attrition-ml/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── .venv/
│
├── explore_data.py
├── app.py
│
├── employee_attrition_model.pkl
├── attrition_threshold.pkl
│
├── README.md
└── requirements.txt

-----------------------------------------------------------------

How to Run

1. Clone the repository
git clone <your-repository-url>
2. Navigate into the project
cd employee-attrition-ml
3. Create and activate a virtual environment
python -m venv .venv

On Windows PowerShell:

.venv\Scripts\Activate.ps1
4. Install dependencies
pip install -r requirements.txt
5. Run the Streamlit application
streamlit run app.py

The application will open in your browser.

-------------------------------------------------------------------

⚠️ Disclaimer

This project is intended for educational and research purposes.

Employee attrition predictions should be treated as decision-support information rather than definitive predictions of individual employee behavior.

The relationships identified by the model should not be interpreted as causal relationships.

-----------------------------------------------------------------

Author

Hamra
Computer Science
BITS Pilani, Dubai Campus



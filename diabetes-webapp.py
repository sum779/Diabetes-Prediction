import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

df = load_data()

# Custom CSS for styling
st.markdown(
    """
    <style>
        .sidebar .sidebar-content {
        background-color: #f0f2f6;
        color: #333333;
    }
    .sidebar .sidebar-content .block-container {
        padding: 1rem;
    }
    .sidebar .stSlider{
        color: #024993!important; /* Tomato color for the slider */
    }
    body {
        background-color: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    
    h1 {
        color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Adding a button with a link to your website
st.markdown(
    """
    <a href="https://www.sumitaryal.com" target="_blank">
        <button style="background-color: #007bff; color: white; padding: 20px 20px; border: 1px solid; border-radius: 10px; cursor: pointer;">
            Home
        </button>
    </a>
        <a href="https://www.linkedin.com/in/thesumitaryal" target="_blank">
        <button style="background-color: #007bff; color: white; padding: 20px 20px; border: 1px solid; border-radius: 10px; cursor: pointer;">
            Linkedin Profile
        </button>
    </a>
    """,
    unsafe_allow_html=True
)
# HEADINGS
st.markdown("<h1 style='text-align: center;'>Diabetes Checkup App</h1>", unsafe_allow_html=True)
st.info("This app is developed using Machine Learning")
st.sidebar.header('Patient Data')
st.sidebar.info('Please see Patient Data Description for more detail')

# X AND Y DATA
feature = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
target = ['Outcome']
X = df[feature].values
Y = df[target].values

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Feature Engineering
fill_zero = SimpleImputer(missing_values=0, strategy="mean")
x_train = fill_zero.fit_transform(x_train)
x_test = fill_zero.fit_transform(x_test)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

@st.cache_data
def train_model(x_train, y_train):
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=0),
                               param_grid=param_grid,
                               cv=5,
                               n_jobs=-1,
                               verbose=0)
    grid_search.fit(x_train, y_train.ravel())
    return grid_search.best_estimator_

best_rf = train_model(x_train, y_train)

# FUNCTION
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose (mg/dL)', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure (mm Hg)', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness (mm)', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin (mu U/mL)', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# Predictions
user_result = best_rf.predict(user_data)

# VISUALISATIONS
st.markdown("<h2 style='text-align: center;'>Patient Report</h2>", unsafe_allow_html=True)

output = ''
if user_result[0] == 0:
    output = 'You dont have Diabetes'
    color = 'green'
else:
    output = 'You may have Diabetes. Please consult a Doctor'
    color = 'red'

# Visualisation color (example of adding a colored message)
st.markdown(f'<h1 style="color:{color}; text-align: center;">{output}</h1>', unsafe_allow_html=True)

st.subheader('Accuracy:')
st.write(str(accuracy_score(y_test, best_rf.predict(x_test)) * 100) + '%')

# Accordion for dataset information
st.error('This model is intended for educational and research purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. The accuracy indicates that there is still a chance of incorrect predictions. Always consult with a qualified healthcare provider for any health-related decisions and diagnoses.', icon="🚨")

with st.expander("How it is calculated?"):
    st.info("""
        This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases.
        The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.
        Several constraints were placed on the selection of these instances from a larger database.
        In particular, all patients here are females at least 21 years old of Pima Indian heritage.
        An advanced machine learning algorithm, Random Forest,was used in the dataset to predict diabetes.
    """)
with st.expander("Who can use it"):
    st.info("""
Researchers: To study the effectiveness of machine learning algorithms in medical diagnosis.
    """)
with st.expander("Patient Data Description"):
    st.write("""
    **Pregnancies**: Number of times pregnant

    **Glucose**: Plasma glucose concentration at 2 hours in an oral glucose tolerance test

    **BP**: Diastolic blood pressure (mm Hg)

    **SkinThickness**: Triceps skin fold thickness (mm)

    **Insulin**: 2-Hour serum insulin (mu U/ml)

    **BMI**: Body mass index (weight in kg / (height in m)^2)

    **DiabetesPedigreeFunction**: Diabetes pedigree function

    **Age**: Age in years
    """)


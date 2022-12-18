import streamlit as st
import pickle
from PIL import Image
import pandas as pd
import plotly.express as px

rf = pickle.load(open('../models/model','rb'))
encoder = pickle.load(open('../models/encoder','rb'))
cm = Image.open('../img/cm.png') 
cm_sample = Image.open('../img/Confusion-Matrix-1-635x358.jpeg') 



def onehotencoding(data_frame):

    # Convert categorical column data type to categorical.

    cols = ['migrant_worker']

    for col in cols:
        data_frame[col] = data_frame[col].astype(object)

    df_num_train = data_frame.select_dtypes(exclude=[object])
    df_cat_train = data_frame.select_dtypes(include=[object])
    transformed = encoder.transform(df_cat_train)
    df_encoded = pd.DataFrame(transformed, columns=encoder.get_feature_names_out())
    df_train_processed = pd.merge(df_num_train, df_encoded ,left_index=True,right_index=True, how='inner')
    return df_train_processed



st.title("Credit Card Default Detection System")
st.caption('Made by Raymond Lim')
st.caption(" LinkedIn: [raymondlimht](https://www.linkedin.com/in/raymondlimht/), Github repo: [link](https://github.com/raymondlimht).")
st.subheader("Introduction")
st.markdown("This is a credit card default detection system uses machine learning algorithm called a Random Forest Classifier to detect which customers are at risk of defaulting on their credit card payments. This can help credit card companies to identify customers who may need additional support or who may be at higher risk of defaulting, so that they can take appropriate action to mitigate the risk.")
st.subheader("Getting Started")
st.markdown('''Here are detailed instructions for using a credit card default detection system web application: \n
1. On the side bar, you should see a form to input the client profile. Enter the required information, such as the client's age, gender, credit score, and yearly income. \n
2. Once you have entered all of the required information, click the button labeled "Click to predict credit card default!" to proceed. \n
3. The web application will process the inputted information and predict credit card default. The model will determine whether the credit card holder is likely to default on their payments and will issue a warning if necessary. A higher default probability indicates a higher likelihood of credit card default.''')



with st.sidebar:
    st.header('Client profile input')
    age = st.slider('Age:', min_value=18, max_value=99, step=1, value= 18)
    gender = st.selectbox('Gender:', ('Male', 'Female'))
    owns_car = st.radio('Owns car', ['Yes', 'No'], index=1)
    owns_house = st.radio('Owns house', ['Yes', 'No'], index=1)
    no_of_children = st.slider('Number of children:', min_value=0, max_value=10, step=1, value= 0) 
    net_yearly_income = st.number_input('Input Annual Income', min_value=0.00, max_value=300000.00, step=5000.00, value= 100000.00)
    no_of_days_employed = st.number_input('No of days employed', min_value=0, max_value=10000,step=10, value=365)
    occupation_type = st.selectbox('Occupation type', ('Unknown', 'Core staff', 'Accountants',
       'High skill tech staff', 'Sales staff', 'Managers', 'Drivers',
       'Medicine staff', 'Cleaning staff', 'HR staff', 'Security staff',
       'Cooking staff', 'Waiters/barmen staff', 'Low-skill Laborers', 'Laborers',
       'Private service staff', 'Secretaries', 'Realty agents',
       'IT staff'))
    total_family_members = st.number_input('Total family members', min_value=0, max_value=20, value=0)
    migrant_worker = st.radio('Migrant worker?', ['Yes', 'No'], index=1)
    yearly_debt_payments = 0 
    credit_limit =st.number_input('Credit Limit', min_value=0.00, max_value=4000000.00, value=20000.00)
    credit_limit_used = st.slider('Credit limit used (%):', min_value=0, max_value=100, step=1, value= 50) 
    credit_score = st.number_input('Credit score', min_value=0, max_value= 1000, value= 500)
    prev_defaults = st.number_input('Previous defaults', min_value=0, max_value=10, value=0)
    default_in_last_6months = st.number_input('Defaults in last 6 months', min_value=0, max_value=10, value=0)

    run = st.button( 'Click to predict credit card default!')

    if gender == 'Male':
        gender = 'M'
    elif gender == 'Female':
        gender = 'F'

    if owns_car == 'Yes':
        owns_car = 'Y'
    elif owns_car == 'No':
        owns_car = 'N'

    if owns_house == 'Yes':
        owns_house = 'Y'
    elif owns_house == 'No':
        owns_house = 'N'

    if migrant_worker == 'Yes':
        migrant_worker = 1
    elif migrant_worker == 'No':
        migrant_worker = 0

st.markdown('---')  
st.subheader('Prediction Outcome:')


input = {
    'age': [age],
    'gender': [gender],
    'owns_car': [owns_car],
    'owns_house': [owns_house],
    'no_of_children': [no_of_children],
    'net_yearly_income': [net_yearly_income],
    'no_of_days_employed': [no_of_days_employed],
    'occupation_type': [occupation_type],
    'total_family_members': [total_family_members],
    'migrant_worker': [migrant_worker],
    'yearly_debt_payments': [yearly_debt_payments],
    'credit_limit': [credit_limit],
    'credit_limit_used(%)': [credit_limit_used],
    'credit_score': [credit_score],
    'prev_defaults': [prev_defaults], 
    'default_in_last_6months': [default_in_last_6months]
}

df = pd.DataFrame(input)

# New features 

df['cash_balance'] = (df.net_yearly_income - df.yearly_debt_payments)
df['employment_years'] = (df.no_of_days_employed / 365)
df['total_income'] = df.net_yearly_income * df.employment_years

## Profit category

def get_profit_category(value):
    if value < 0 :
        return "Loss"
    else:
        return "Profit"

df["in_profit"] = [get_profit_category(i) for i in df["cash_balance"].values]


df_processed = onehotencoding(df)
predictions = rf.predict(df_processed)
predict_proba = rf.predict_proba(df_processed)

rf_proba = pd.DataFrame(predict_proba, columns=['No Credit Card Default', 'Credit Card Default'])
rf_proba = rf_proba.T
rf_proba.reset_index(inplace=True)
rf_proba = rf_proba.rename(columns = {'index':'Credit Card Default'})


## Declare variables
pct_fraud = predict_proba[0,1] * 100
total_income = df['total_income'][0]
employment_years = df['employment_years'][0]
cash_balance = df['cash_balance'][0]
financial_performance = df['in_profit'][0]


if predictions[0] == 1:
    st.write(''' 🚨 ⚠️  **Warning!** The model predicts there will be credit card default on the account.  \n
There is a {:.2f}% probability that the credit card holder will default on their payments, according to the prediction model.'''.format(pct_fraud))
elif predictions[0] == 0 :
    st.write(''' 👍 ✅  **Safe!** The model predicts there will be no credit card default on the account. \n
There is a {:.2f}% probability that the credit card holder will default on their payments, according to the prediction model.'''.format(pct_fraud))


st.subheader('Additional information about the credit card holder:')

st.markdown(
''' - Cash Balance <sup> 1 </sup> = $ {:.2f} \n
- Employment years = {:.1f} years\n
- Total Income <sup> 2 </sup>  = $ {:.2f}\n 
- Financial Performance = {}\n
            '''.format(cash_balance, employment_years, total_income, financial_performance),unsafe_allow_html=True)       
st.caption('<small> 1. Cash Balance = Net Yearly Income - Yearly Debt Payments</small>',unsafe_allow_html=True)
st.caption('<small> 2. Total Income = Net Yearly Income x Employment Years</small>',unsafe_allow_html=True)

st.markdown('---')  


pie_chart = px.pie(rf_proba, values=0, names='Credit Card Default', title="Probability of Credit Card Default")
# pie_chart.show()
st.plotly_chart(pie_chart, use_container_width=True)    



st.header('Random Forest Classifier')
st.write('''We are using a prediction model called the Random Forest classifier to forecast whether a credit card holder is likely to default on their payments. 
    The Random Forest classifier is a popular machine learning algorithm that uses multiple decision trees to make predictions. It is known for its ability to handle large datasets and to generate accurate and robust predictions.
This model has a recall score of 0.98, which means that it is able to accurately identify a large proportion of credit card defaults in our dataset. 
''')
st.subheader('Feature Importance')
st.write('In a Random Forest Classifier, feature importance refers to the relative importance of each feature in predicting the target variable. The Random Forest algorithm assigns a score to each feature, indicating its importance in the model\'s prediction.  \
        Features with a higher score are considered more important than those with a lower score.')


feature_importance = pd.DataFrame(list(zip(list(df_processed.columns), list(rf.feature_importances_))), columns = ['features', 'feature_impt'])
feature_importance.sort_values(by=['feature_impt'], ascending=False,inplace=True)
feature_importance['feature_impt'] = feature_importance['feature_impt'].round(decimals = 4)

fig = px.bar(feature_importance.head(10).sort_values(by ='feature_impt', ascending = True),x= 'feature_impt' , y= 'features', text= "feature_impt", title="Random Forest Feature Importance")

st.plotly_chart(fig, use_container_width=True)
# fig.show()

st.subheader('Confusion Matrix')
st.write('A confusion matrix is a table that is used to evaluate the performance of a classification model. It is also often used to calculate evaluation metrics such as accuracy, precision, and recall, which can help you understand the performance of the model and identify areas where it may be performing poorly. ')
st.image(cm, caption='Confusion Matrix')

st.subheader('Interpret Confusion Matrix ')
st.image(cm_sample, caption='Source: https://rapidminer.com/glossary/confusion-matrix/')

st.subheader('Recall')
st.write('Recall is defined as the number of true positive predictions made by the model divided by the total number of positive cases in the dataset.')
st.latex('''(True Positive) / (True Positive + False Negative)''')
st.write('''In this case, the recall would be calculated as follows:

Recall =  7873 / (7873 + 29)
= 0.996

This means that the model has a recall of 99.6%, indicating that it was able to correctly identify almost all of the positive cases in the dataset.''')

st.subheader('Precision')
st.write('Precision is defined as the number of true positive predictions made by the model divided by the total number of positive predictions made by the model.')
st.latex('''Precision = (True Positive) / (True Positive + False Positive)''')
st.write('''In this case, the precision would be calculated as follows:

Precision = 7873 / (7873 + 152)
= 0.980

This means that the model has a precision of 98.0%, indicating that out of all the predictions it made for the positive class, 98.0% of them were correct.''')

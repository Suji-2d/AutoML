import streamlit as st
import pandas as pd
from io import StringIO
import os
import pickle

#profiling imports
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report 

#autoMl imports
import model.classifier_model as cm
import model.regression_model as rm
#from pycaret.regression import*


with st.sidebar:

    st.image("https://developer.apple.com/assets/elements/icons/create-ml/create-ml-96x96_2x.png")
    st.title('Auto ML')
    #st.write("""---""")
    nav_choice = st.radio(
        "Navigation",
       ['Uploading','Profiling','Mechine Learning','Forecasting']
    )

if nav_choice =="Uploading":
    st.write("""
    ### Upload you Data for Modeling!
    """)
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file,index_col=0)
        st.dataframe(df)
        df.to_csv('model_data.csv',index=False)

if os.path.exists("model_data.csv"):
    df=pd.read_csv('model_data.csv',index_col=0)

if nav_choice=="Profiling":
    st.title("Automated Exploratory Data Analysis")
    data_report = df.profile_report()
    st_profile_report(data_report)

model_type = "Classification"

if nav_choice == 'Mechine Learning':
    st.title("ML model selection")
    model_type = st.radio('Select model type',
    ('Classification','Regression'))
    target = st.selectbox('Select the target',df.columns)
    if st.button('Train Model'):
        st.write("""---""")
        if model_type == 'Classification':
            model_list = cm.get_model(target) #[ml experiment settings, model compare results, best model]
            st.info("This is the ML experiment settings")
            st.dataframe(model_list[0])
            st.info("Comparision table of ML models")
            st.dataframe(model_list[1])
        else:
            model_list = rm.get_model(target) #[ml experiment settings, model compare results, best model]
            st.info("This is the ML experiment settings")
            st.dataframe(model_list[0])
            st.info("Comparision table of ML models")
            st.dataframe(model_list[1])
            
        with open('best_model.pkl','rb') as f :
            st.download_button('Download Model',f,'best_model.pkl')

if nav_choice == 'Forecasting':
    st.title('Predict target with the model')
    try:
        with open('best_model.pkl','rb') as f :
            model = pickle.load(f)
            test_file = st.file_uploader("Choose a file")
            if test_file:
                test_df = pd.read_csv(test_file,index_col=0)
                if model_type == "Classification":
                    test_result = cm.predict_test(test_df)
                else:
                    test_result = rm.predict_test(test_df)
                test_result.to_csv('test_result.csv',index=False)
                if st.button('Predict'):
                    st.dataframe(test_result)
                    with ('test_result.csv','rb') as f :
                        st.download_button('Download Model',f,'test_result.csv')
    except Exception as e:
        st.write("Oops..! Something went worng, please check if you target and test data match")
        st.write(e)


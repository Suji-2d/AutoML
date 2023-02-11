import pandas as pd
from pycaret.classification import*
import os

# if os.path.exists("./model_data.csv"):
#     df=pd.read_csv('./model_data.csv',index_col=None)

def get_model(df,target):
    setup(df,target=target)
    setup_df = pull()
    best_model = compare_models()
    compare_df = pull()
    save_model(best_model,'best_model')
    return([setup_df,compare_df,best_model])

def predict_test(test_df):
    best_model = load_model('best_model') 
    return predict_model(best_model,data=test_df)

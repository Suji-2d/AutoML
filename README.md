# Auto-ML Project Documentation  
This project provides automation of machine learning (ML) regression and classification tasks using Python, Pandas, pyCaret, and Streamlit. The project consists of the following pages:

### 1. Uploading  
The "Uploading" page allows users to upload a CSV file for analysis.
It provides a function to check if the CSV file is small, and if not, it offers options to sample the data.
Additionally, the page includes features for splitting the data into train and test sets if needed.
### 2. Profiling (pandas_profiling)  
The "Profiling" page utilizes pandas_profiling to generate a comprehensive report on the uploaded dataset.
It provides insights and statistical analysis of the data, including summary statistics, data types, missing values, correlations, and more.
### 3. AutoML (Classification/Regression) [pyCaret]  
The "AutoML" page focuses on the ML modeling process.
Users can select the target variable and choose the ML type (classification or regression).
The page trains and evaluates various ML models using pyCaret, an automated machine learning library.
It provides performance metrics, such as accuracy, precision, recall, and F1-score, to help users assess model performance.
### 4. Prediction [pyCaret]  
The "Prediction" page enables users to upload a CSV file for prediction or testing purposes.
The best-performing model from the AutoML process is utilized to generate predictions for the provided dataset.

### Future enhansment:   
- Add "Clear Data" button to delete internally stored files (input_df, output_df and model_pkl)
- Add EDA page and use LLM for providing EDA for the dataset
- Add sampling and train and test splitting features in Uploading page


deployement platfrom: Streamlit.   
app-link: https://suji-2d-automl-automl-app-rwdf0z.streamlit.app/

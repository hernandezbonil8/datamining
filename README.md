⚽ Football Match Outcome Prediction

Welcome to our football match prediction project! This repo walks through analyzing team-level stats from the 2021–2022 season and using that data to build a model that predicts whether a team will win. We also included a Streamlit web app so you can easily interact with the model and try out predictions yourself.

 Authors

Emerson Hernandez, Toufeeq Sharieff, Mani Vutukuri and Michael Nguyen

Course: CS 484 – Data Mining  
George Mason University — Spring 2025

 Project Structure

File/Folder                 Purpose                                                                 

 `data/`                    Contains the CSV file used for training and EDA                        
 `eda.py`                   Exploratory Data Analysis: basic stats, missing values, plots          
 `model_training.py`        Trains the logistic regression model and saves it as a `.pkl` file     
 `match_prediction.py`      Allows you to test the model manually on team matchups                 
 `match_prediction_model.pkl`  Trained model file (used for predictions)                          
 `streamlit_app.py`         The Streamlit interface to run predictions in a web browser            
 `requirements.txt`         Python package dependencies                                             
 `final_report.tex`         (Optional) LaTeX file used to generate our final write-up/report       


 Setup Instructions

First, make sure you have Python 3.7+ and install the dependencies:


pip install -r requirements.txt


 1. Run EDA (`eda.py`)


python eda.py


This will:
- Load the dataset from `data/2021-2022 Football Team Stats.csv`
- Display summary statistics
- Check for missing values
- Generate a histogram of 'Goals For (GF)' if available

Output is printed in the terminal and the plot is shown using Seaborn/Matplotlib.

 2. Train the Model (`model_training.py`)

python model_training.py


This will:
- Train a logistic regression model using goal difference and other engineered features
- Save the trained model as `match_prediction_model.pkl`

 3. Make Manual Predictions (`match_prediction.py`)


python match_prediction.py --save-synthetic


This script:
- Loads the trained model
- Lets you manually enter two teams (e.g., Manchester City vs Liverpool)
- Prints out the predicted win probability

Sample output:


Predicted probability that Manchester City wins over Liverpool: 0.87
Result: Manchester City is predicted to win


4. Launch the Web App (`streamlit_app.py`)


streamlit run streamlit_app.py



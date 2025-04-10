# Head-to-Head Football Match Outcome Predictor

## Overview
This project uses three datasets:
- **2021-2022 Football Team Stats.csv**: Club-level statistics.
- **Soccer_Football Clubs Ranking in june.csv**: Additional club ranking data.
- **players_15.csv**: Player-level data with performance metrics (overall rating, potential).

These datasets are merged to form a unified club-level dataset, with player statistics aggregated by club. We then create synthetic match-level examples by pairing clubs and taking the difference in their features. A Logistic Regression model is trained on these examples to predict the outcome of a head-to-head match (i.e., which club is likely to win based on a higher “point score”).

## Files
- **data/**: Contains the CSV files.
- **match_prediction.py**: Merges the datasets, creates synthetic data, trains the model, and provides a function to predict match outcomes. Run this to generate and save `match_prediction_model.pkl`.
- **streamlit_app.py**: An interactive UI built with Streamlit for predicting head-to-head match outcomes.
- **requirements.txt**: Lists the required Python packages.
- **final_report.tex**: The LaTeX source for your final project report.

## Setup and Usage

1. **Organize Files:**
   - Place the CSV files inside a folder named `data` in the project directory.
   - Ensure the project folder contains all the files as listed above.

2. **Install Dependencies:**
   - Create a `requirements.txt` file with the following:
     ```
     pandas
     numpy
     matplotlib
     seaborn
     scikit-learn
     joblib
     streamlit
     ```
   - Install the dependencies by running:
     ```bash
     python -m pip install -r requirements.txt
     ```

3. **Train the Model:**
   - Run the model training script:
     ```bash
     python match_prediction.py
     ```
   - This will merge the data, create synthetic match-level data, train a Logistic Regression model, and save the model as `match_prediction_model.pkl`.

4. **Run the Interactive App:**
   - Launch the Streamlit app with:
     ```bash
     streamlit run streamlit_app.py
     ```
   - In the app, select two clubs from the dropdown menus and click “Predict Outcome” to see the prediction.

5. **Final Report:**
   - Update and compile the `final_report.tex` using your preferred LaTeX editor.

## Future Work
- Explore additional features (home/away records, recent form, injury data).
- Experiment with other classical models or ensemble techniques.
- Enhance the UI further for deeper insights.

Feel free to reach out if you have any questions!

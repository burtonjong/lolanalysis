# Project Proposal – Predicting League of Legends Match Outcomes Using First 10 Minutes of Game Data

## Proposed Client (Who)
The proposed client is a competitive esports analytics organization that works with professional League of Legends teams in the EMEA region. Their goal is to enhance match preparation and in game decision making through predictive analytics.

## Question being investigated (Why)
Can we predict the outcome of a League of Legends match in the LEC Winter 2025 season based solely on in-game statistics collected during the first 10 minutes of gameplay in order to help the team decide what areas of the game to focus on in the early game (first 10 minutes)..

## Plan of Attack (How)
The plan is to preprocess the dataset and isolate early-game features from the first 10 minutes of the game. We will engineer relevant features, apply multiple machine learning models to predict match outcomes, and evaluate their performance using standard classification metrics. Model interpretability will be supported using SHAP to identify the most critical early-game indicators.

## Dataset, Models, Frameworks, Components (What)
Dataset: _'League of Legends LEC Winter Season 2025 Stats'_ from Kaggle.
Link: [https://www.kaggle.com/datasets/smvjkk/league-of-legends-lec-winter-season-2025-stats](https://www.kaggle.com/datasets/smvjkk/league-of-legends-lec-winter-season-2025-stats)

## Models: 
- Logistic Regression (baseline linear model)
- Random Forest (non-linear)
- Gradient Boosting (non-linear)

## Tools and Frameworks:
- Python, Jupyter Notebook, VS code
- Scikit-learn, XGBoost (potentially)
- Pandas, NumPy, Matplotlib, Seaborn
- SHAP for interpretability

## Components:
- Filter early-game features (first 10 minutes)
- Data preprocessing and model training
- Feature importance visualization and SHAP analysis




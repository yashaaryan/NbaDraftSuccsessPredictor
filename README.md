# NBA Draft Prospect Success Prediction

## Project Overview
This project aims to predict the success of NBA draft prospects based on their college statistics, physical attributes, and historical draft data. It utilizes machine learning models, primarily XGBoost, to analyze various features and classify prospects as successful or not.

## Dataset
The dataset consists of player information such as:
- College performance metrics
- Physical attributes 
- Draft position
- Conference and team details
- Post Nba career stats

## Features Used
The model uses the following features for prediction:
- College stats (points, rebounds, assists, etc.)
- Physical attributes 
- Draft information (pick number, team, conference, etc.)

## Model Training & Evaluation
### Steps:
1. **Preprocessing**:
   - Drop non-feature columns (`Player`, `team`, `yr`, etc.)
   - Handle missing values
   - Normalize data if necessary

2. **Model Training**:
   - Train an `XGBoost` classifier with optimized hyperparameters
   - Use `train_test_split` with a 75% test size for evaluation

3. **Hyperparameter Tuning**:
   - Utilize `GridSearchCV` to optimize `max_depth`, `learning_rate`, `n_estimators`, etc.
   - Implement early stopping to prevent overfitting

4. **Evaluation**:
   - Measure accuracy using `accuracy_score`
   - Perform cross-validation to ensure robustness

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib (for visualization)

## Installation
```sh
pip install pandas numpy scikit-learn xgboost matplotlib
```

## Running the Model
Run the following script to train and evaluate the model:
```sh
python train_model.py
```

## Future Improvements
- Experiment with ensemble models (Random Forest, LightGBM)
- Improve feature engineering
- Tune hyperparameters further
- Deploy as a web API for real-time predictions


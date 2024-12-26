# Rickety Cricket
Rickety Cricket is a live cricket predictions app for T20 matches. The project includes:

* Model training code for creating the CatBoost (gradient boosted trees).
* An AWS Lambda function for periodically making predictions and updating DynamoDB results.
* A Flask web app for viewing live matches, generating predictions, and displaying model performance.

## Key Features
1. Live Match Prediction

* Fetches current men’s T20 matches via CricAPI.
* Predicts the probability of the chasing team winning using a CatBoost model.

2. Automated Predictions

* An AWS Lambda function regularly checks for new matches, runs predictions, and stores them in DynamoDB.

3. Model Performance Tracking

* A dedicated route in the Flask app (/track_model_performance) displays weekly and overall accuracy.

4. Model Training

* The model_training folder includes notebooks and scripts for data preprocessing and CatBoost model training.

## Data Source & License

This project uses historical cricket data from [Cricsheet](https://cricsheet.org/),
licensed under the [Open Data Commons Attribution License (ODC-By 1.0)](http://opendatacommons.org/licenses/by/1.0/).

Per the license requirements, we attribute the original data provider:
"Data provided by Cricsheet. Used under ODC-By 1.0."
# Rickety Cricket
Rickety Cricket is a live cricket predictions app that uses machine learning (CatBoost) to predict the probability of a chasing team winning a match in progress. The project includes:

* A Flask web app for viewing live matches, generating predictions, and displaying model performance.
* An AWS Lambda function for periodically making predictions and updating DynamoDB results.
* Model training code for creating and refining the CatBoost model using historical data.

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

## Setup & Installation
Clone the repository:

bash
Copy code
git clone https://github.com/DannyCapper/rickety-cricket.git
cd rickety-cricket
Create and activate a virtual environment:

bash
Copy code
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
Install dependencies:

For development and web usage:
bash
Copy code
pip install --upgrade pip
pip install -r requirements/dev_requirements.txt
pip install -r requirements/web_requirements.txt
If you need to run the Lambda code locally, also install:
bash
Copy code
pip install -r requirements/lambda_requirements.txt
Install the package in editable mode (optional but recommended):

bash
Copy code
pip install -e .
This ensures src/ is recognized as an importable module.

(Optional) Environment variables:

If you’re using AWS Secrets Manager for your CricAPI key, ensure your AWS credentials and region settings are properly configured (aws configure).
For local testing with a local or remote DynamoDB, set relevant environment variables if needed (e.g., AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY).
Usage
Run the Flask Web App Locally
From the project root (after activating your virtual environment):

bash
Copy code
# If installed in editable mode:
flask --app src/web/app.py run
OR:

bash
Copy code
cd src/web
python app.py
Then open http://127.0.0.1:5000/ in your browser.

Key routes:

Home (/): Displays live men’s T20 matches (if any) and a form to generate a prediction.
Predict (/predict): Endpoint to get a probability for the chasing team.
Model Performance (/track_model_performance): Shows weekly and overall prediction accuracy.
Lambda Function (Local Test)
If you want to invoke the Lambda code (src/lambda/) locally:

bash
Copy code
# Make sure lambda dependencies are installed
pip install -r requirements/lambda_requirements.txt

# For a quick local test (if your code is in src/lambda/lambda_function.py):
python src/lambda/lambda_function.py
(You may need AWS credentials if your code calls real AWS resources.)

Testing
Unit Tests & Coverage:

bash
Copy code
pytest --cov=src --cov=model_training --cov-report=term-missing
Manual Testing:

Ensure the web app runs, and navigate to / and /track_model_performance.
Check the predictions flow when selecting a live match (if available).
Inspect logs to ensure DynamoDB reads/writes and CricAPI calls succeed.
Docker (Optional)
If you have a Dockerfile in the project root, you can build and run the container:

bash
Copy code
docker build -t rickety-cricket .
docker run -p 5000:5000 rickety-cricket
Then open http://localhost:5000 to view the Flask app.

(Adjust the Dockerfile as necessary for the Lambda image if you have separate images.)

AWS Lambda Deployment
To deploy the Lambda function to AWS:

Package the Lambda code with its dependencies.
Upload (manually or via CI/CD) to AWS Lambda.
Configure environment variables (e.g., table name, region, etc.) in the Lambda console or via IaC.
Schedule the Lambda to run (e.g., using EventBridge) to periodically fetch match info, predict, and update results in DynamoDB.
Contributing
Fork this repo.
Create a new branch for your feature or bugfix.
Commit your changes with a clear message.
Open a Pull Request.
We welcome issues and PRs that fix bugs, improve code quality, or add new features.

License
This project is licensed under the MIT License. See the LICENSE file for details.
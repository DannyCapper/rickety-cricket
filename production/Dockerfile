# Use the official AWS Lambda Python 3.8 image
FROM public.ecr.aws/lambda/python:3.8

# Install OS dependencies (if needed)
RUN yum install -y gcc

# Copy requirements.txt and install dependencies
COPY requirements.txt ./
COPY predictions.py ./
RUN pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy your script and model file into the container
COPY predict_winner.py ${LAMBDA_TASK_ROOT}
COPY model.cbm ${LAMBDA_TASK_ROOT}

# Set the command to your handler (e.g., live_predictions.main)
CMD ["predict_winner.main"]

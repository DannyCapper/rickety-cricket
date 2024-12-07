# Use the official AWS Lambda Python 3.8 base image
FROM public.ecr.aws/lambda/python:3.8

# Install OS dependencies if needed
RUN yum install -y gcc

# Copy the requirements file and install dependencies into the Lambda task root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy project folders into the Lambda task root
COPY production/ ${LAMBDA_TASK_ROOT}/production/
COPY utils/ ${LAMBDA_TASK_ROOT}/utils/

# Set the working directory to the Lambda task root
WORKDIR ${LAMBDA_TASK_ROOT}

# Specify the handler function for AWS Lambda
CMD ["rickety_cricket.production.predict_winner.main"]
FROM public.ecr.aws/lambda/python:3.9

COPY requirements/lambda.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/lambda/ ${LAMBDA_TASK_ROOT}/src/lambda/
COPY src/utils/ ${LAMBDA_TASK_ROOT}/src/utils/
COPY src/model.cbm ${LAMBDA_TASK_ROOT}/src/model.cbm

CMD [ "src.lambda.lambda_function.main" ]
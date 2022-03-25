FROM public.ecr.aws/lambda/python:3.8

COPY app.py requirements.txt ./
COPY consts.py ./
COPY aws_s3.py ./
COPY InceptionV3 /opt/ml/model

RUN python3.8 -m pip install -r requirements.txt -t .

CMD ["app.lambda_handler"]

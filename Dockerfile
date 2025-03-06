FROM python:3.11-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements-deploy.txt

CMD ["python3", "app-flask-cite.py"]
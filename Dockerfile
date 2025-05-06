FROM python:3.11.1-slim-buster
WORKDIR /app
COPY . /app/

RUN apt-get update -y && apt-get install -y 

RUN pip install -r requirements.txt
CMD ["python", "app.py"]
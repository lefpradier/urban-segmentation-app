# backend/Dockerfile

FROM python:3.10.6

COPY deployment/frontend .

WORKDIR .

RUN pip install -r requirements.txt

EXPOSE 80

CMD ["python", "app.py"]
